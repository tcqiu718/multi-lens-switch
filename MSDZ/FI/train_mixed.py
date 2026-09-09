"""Fine-tune an FI model with original multi-frame and generated triplet data.

Triplets can be discovered from either of these layouts::

    triplet_root/train/scene_001/{000.png,001.png,002.png}

or the unmodified demo_paper output layout::

    triplet_root/scene_001/input/{wide.png,tele.png}
    triplet_root/scene_001/results/full_result.png

For explicit paths or per-scene timesteps, pass a CSV containing the columns
``img0,gt,img1,timestep``. Relative paths are resolved from the CSV directory.
"""

import argparse
import csv
import faulthandler
import math
import os
import random
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


faulthandler.enable(all_threads=True)
cv2.setNumThreads(1)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class OriginalSequence:
    name: str
    frames: tuple


@dataclass(frozen=True)
class Triplet:
    name: str
    img0: Path
    gt: Path
    img1: Path
    timestep: float


def natural_key(path):
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", Path(path).name)
    ]


def direct_images(directory):
    return tuple(
        sorted(
            (
                path
                for path in Path(directory).iterdir()
                if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
            ),
            key=natural_key,
        )
    )


def train_subdirectory(root):
    root = Path(root).expanduser().resolve()
    candidate = root / "train"
    return candidate if candidate.is_dir() else root


def discover_original_sequences(root):
    base = train_subdirectory(root)
    if not base.is_dir():
        raise FileNotFoundError("Original training directory not found: {}".format(base))

    sequences = []
    for directory in sorted((path for path in base.iterdir() if path.is_dir())):
        frames = direct_images(directory)
        if not frames:
            continue
        if len(frames) < 3:
            raise ValueError(
                "Original sequence '{}' needs at least 3 images, found {}.".format(
                    directory, len(frames)
                )
            )
        sequences.append(OriginalSequence(directory.name, frames))
    if not sequences:
        raise ValueError("No original multi-frame sequences found under {}".format(base))
    return tuple(sequences)


def validate_timestep(value, name):
    timestep = float(value)
    if not 0.0 < timestep < 1.0:
        raise ValueError(
            "Triplet '{}' timestep must be between 0 and 1, got {}.".format(
                name, timestep
            )
        )
    return timestep


def resolve_manifest_path(value, manifest_dir):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_dir / path
    return path.resolve()


def load_triplet_manifest(path, default_timestep):
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError("Triplet manifest not found: {}".format(manifest_path))

    triplets = []
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"img0", "gt", "img1"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                "Triplet manifest is missing columns: {}".format(
                    ", ".join(sorted(missing))
                )
            )
        for row_index, row in enumerate(reader, start=2):
            name = (row.get("scene") or row.get("name") or "row_{}".format(row_index)).strip()
            value = row.get("timestep", "").strip()
            timestep = validate_timestep(
                default_timestep if not value else value,
                name,
            )
            paths = {
                key: resolve_manifest_path(row[key], manifest_path.parent)
                for key in required
            }
            for key, image_path in paths.items():
                if not image_path.is_file():
                    raise FileNotFoundError(
                        "Manifest row {} {} image not found: {}".format(
                            row_index, key, image_path
                        )
                    )
            triplets.append(
                Triplet(name, paths["img0"], paths["gt"], paths["img1"], timestep)
            )
    if not triplets:
        raise ValueError("Triplet manifest contains no data rows: {}".format(manifest_path))
    return tuple(triplets)


def discover_triplets(root, default_timestep):
    base = train_subdirectory(root)
    if not base.is_dir():
        raise FileNotFoundError("Triplet directory not found: {}".format(base))

    triplets = []
    seen_directories = set()

    for gt_path in sorted(base.rglob("full_result.png")):
        if gt_path.parent.name.casefold() != "results":
            continue
        group = gt_path.parent.parent
        img0 = group / "input" / "wide.png"
        img1 = group / "input" / "tele.png"
        if img0.is_file() and img1.is_file():
            relative_name = str(group.relative_to(base)).replace(os.sep, "/")
            triplets.append(
                Triplet(
                    relative_name,
                    img0.resolve(),
                    gt_path.resolve(),
                    img1.resolve(),
                    validate_timestep(default_timestep, relative_name),
                )
            )
            seen_directories.add(group.resolve())

    candidate_directories = [base]
    candidate_directories.extend(path for path in base.iterdir() if path.is_dir())
    for directory in candidate_directories:
        if directory.resolve() in seen_directories:
            continue
        images = direct_images(directory)
        if len(images) == 3:
            triplets.append(
                Triplet(
                    directory.name,
                    images[0].resolve(),
                    images[1].resolve(),
                    images[2].resolve(),
                    validate_timestep(default_timestep, directory.name),
                )
            )
        elif images:
            raise ValueError(
                "Triplet directory '{}' must contain exactly 3 direct images, found {}. "
                "Use --triplet_manifest when a directory contains other images.".format(
                    directory, len(images)
                )
            )

    if not triplets:
        raise ValueError("No generated triplets found under {}".format(base))
    return tuple(triplets)


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Failed to read image: {}".format(path))
    return image


def center_zoom(image, scale):
    if scale < 1.0:
        raise ValueError("Start-frame zoom scale must be at least 1.0, got {}".format(scale))
    if math.isclose(scale, 1.0, rel_tol=1e-9, abs_tol=1e-12):
        return image

    height, width = image.shape[:2]
    resized = cv2.resize(
        image,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_CUBIC,
    )
    y0 = (resized.shape[0] - height) // 2
    x0 = (resized.shape[1] - width) // 2
    return resized[y0:y0 + height, x0:x0 + width]


def resize_images(images, train_size):
    if train_size is None:
        return images
    height, width = train_size
    return [
        cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        for image in images
    ]


class MixedInterpolationDataset(Dataset):
    def __init__(
        self,
        original_sequences,
        triplets,
        generated_probability,
        samples_per_epoch,
        original_start_scale,
        triplet_start_scale,
        train_size=None,
        augment=True,
    ):
        self.original_sequences = original_sequences
        self.triplets = triplets
        self.generated_probability = float(generated_probability)
        self.samples_per_epoch = int(samples_per_epoch)
        self.original_start_scale = float(original_start_scale)
        self.triplet_start_scale = float(triplet_start_scale)
        self.train_size = train_size
        self.augment = augment

    def __len__(self):
        return self.samples_per_epoch

    def _sample_original(self):
        sequence = random.choice(self.original_sequences)
        index = random.randint(1, len(sequence.frames) - 2)
        timestep = index / float(len(sequence.frames) - 1)
        return (
            read_image(sequence.frames[0]),
            read_image(sequence.frames[index]),
            read_image(sequence.frames[-1]),
            timestep,
            self.original_start_scale,
            0,
        )

    def _sample_triplet(self):
        triplet = random.choice(self.triplets)
        return (
            read_image(triplet.img0),
            read_image(triplet.gt),
            read_image(triplet.img1),
            triplet.timestep,
            self.triplet_start_scale,
            1,
        )

    def __getitem__(self, index):
        del index
        use_generated = random.random() < self.generated_probability
        img0, gt, img1, timestep, start_scale, source = (
            self._sample_triplet() if use_generated else self._sample_original()
        )

        img0 = center_zoom(img0, start_scale)
        img0, gt, img1 = resize_images([img0, gt, img1], self.train_size)
        if img0.shape != gt.shape or img0.shape != img1.shape:
            raise ValueError(
                "A training triplet must have equal image shapes, got {}, {}, {}. "
                "Use --train_size HEIGHT WIDTH to normalize them.".format(
                    img0.shape, gt.shape, img1.shape
                )
            )

        if self.augment:
            if random.random() < 0.5:
                img0, gt, img1 = [image[::-1, :, :] for image in (img0, gt, img1)]
            if random.random() < 0.5:
                img0, gt, img1 = [image[:, ::-1, :] for image in (img0, gt, img1)]

        tensors = [
            torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
            for image in (img0, img1, gt)
        ]
        images = torch.cat(tensors, dim=0)
        timestep_tensor = torch.tensor(timestep, dtype=torch.float32).reshape(1, 1, 1)
        return images, timestep_tensor, torch.tensor(source, dtype=torch.float32)


def checked_collate(batch):
    shapes = [tuple(item[0].shape) for item in batch]
    if any(shape != shapes[0] for shape in shapes[1:]):
        raise ValueError(
            "Images in one batch have different shapes: {}. Set --train_size HEIGHT WIDTH.".format(
                shapes
            )
        )
    return default_collate(batch)


def seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_model(name, model_rank):
    if name == "EDSC":
        from model.FI_models.EDSCVgg import Model
    elif name == "IFRNet":
        from model.FI_models.IFRNetVgg import Model
    elif name == "RIFE":
        from model.FI_models.RIFEVgg import Model
    elif name == "AMT":
        from model.FI_models.AMTVgg import Model
    elif name == "UPRNet":
        from model.FI_models.UPRNetVgg import Model
    elif name == "EMAVFI":
        from model.FI_models.EMAVFIVgg import Model
    else:
        raise ValueError("Unsupported FI model: {}".format(name))
    return Model(model_rank)


def flownet_module(model):
    return model.flownet.module if hasattr(model.flownet, "module") else model.flownet


def load_torch_file(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def resolve_pretrained_file(path, model_name, suffix):
    candidate = Path(path).expanduser().resolve()
    if candidate.is_file():
        return candidate
    if not candidate.is_dir():
        raise FileNotFoundError("Pretrained path not found: {}".format(candidate))

    stem = "{}_flownet".format(suffix) if suffix is not None else "flownet"
    extensions = (".pth", ".pkl") if model_name == "AMT" else (".pkl", ".pth")
    candidates = [candidate / (stem + extension) for extension in extensions]
    for weight_path in candidates:
        if weight_path.is_file():
            return weight_path
    raise FileNotFoundError(
        "No pretrained weights found. Checked: {}".format(
            ", ".join(str(weight_path) for weight_path in candidates)
        )
    )


def extract_model_state(checkpoint, weight_path):
    if not isinstance(checkpoint, dict):
        raise ValueError("Weights must contain a state dictionary: {}".format(weight_path))
    for key in ("model_state", "state_dict"):
        state = checkpoint.get(key)
        if isinstance(state, dict):
            return state
    return checkpoint


def load_initial_weights(model, path, model_name, suffix, device):
    weight_path = resolve_pretrained_file(path, model_name, suffix)
    checkpoint = load_torch_file(str(weight_path), device)
    state = extract_model_state(checkpoint, weight_path)
    normalized_state = {}
    for key, value in state.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            continue
        while key.startswith("module."):
            key = key[len("module."):]
        if "attn_mask" in key or key == "HW" or key.endswith(".HW"):
            continue
        normalized_state[key] = value
    if not normalized_state:
        raise ValueError("No model parameters found in {}".format(weight_path))

    try:
        flownet_module(model).load_state_dict(normalized_state, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            "Weights in '{}' do not match the selected {} model."
            .format(weight_path, model_name)
        ) from error
    return weight_path


def load_resume_checkpoint(model, path, model_name, device):
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Resume checkpoint not found: {}".format(checkpoint_path))
    checkpoint = load_torch_file(str(checkpoint_path), device)
    required = {"model_name", "model_state", "optimizer_state", "epoch", "step"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError("Checkpoint is missing keys: {}".format(", ".join(sorted(missing))))
    if checkpoint["model_name"] != model_name:
        raise ValueError(
            "Checkpoint model '{}' does not match '{}'.".format(
                checkpoint["model_name"], model_name
            )
        )
    flownet_module(model).load_state_dict(checkpoint["model_state"])
    model.optimG.load_state_dict(checkpoint["optimizer_state"])
    return int(checkpoint["epoch"]) + 1, int(checkpoint["step"])


def save_inference_weights(model, log_dir, epoch):
    bare_state = flownet_module(model).state_dict()
    wrapped_state = {"module." + key: value for key, value in bare_state.items()}
    numbered_path = os.path.join(log_dir, "{}_flownet.pkl".format(epoch))
    latest_path = os.path.join(log_dir, "flownet.pkl")
    torch.save(wrapped_state, numbered_path)
    torch.save(wrapped_state, latest_path)
    return numbered_path


def save_training_checkpoint(model, args, epoch, step):
    checkpoint = {
        "format_version": 1,
        "model_name": args.model,
        "model_state": flownet_module(model).state_dict(),
        "optimizer_state": model.optimG.state_dict(),
        "epoch": epoch,
        "step": step,
        "training_args": vars(args).copy(),
    }
    latest_path = os.path.join(args.log_dir, "checkpoint_latest.pth")
    temporary_path = latest_path + ".tmp"
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, latest_path)

    completed_epochs = epoch + 1
    archive_path = None
    if completed_epochs % args.save_every == 0 or completed_epochs == args.epoch:
        archive_path = os.path.join(
            args.log_dir, "checkpoint_epoch_{:04d}.pth".format(completed_epochs)
        )
        shutil.copyfile(latest_path, archive_path)
    return latest_path, archive_path


def learning_rate_at(step, total_steps, args):
    if args.warmup_steps > 0 and step < args.warmup_steps:
        return args.learning_rate * float(step + 1) / float(args.warmup_steps)
    cosine_steps = max(1, total_steps - args.warmup_steps)
    progress = min(max((step - args.warmup_steps) / float(cosine_steps), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.min_learning_rate + (args.learning_rate - args.min_learning_rate) * cosine


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune an FI model with multi-frame sequences and generated triplets"
    )
    parser.add_argument(
        "--model",
        default="RIFE",
        choices=("EDSC", "IFRNet", "RIFE", "AMT", "UPRNet", "EMAVFI"),
    )
    parser.add_argument(
        "--dataset_dir",
        required=True,
        help="original dataset root containing train/<sequence>/*.png",
    )
    parser.add_argument(
        "--triplet_dir",
        default="",
        help="generated triplet root; train/ is detected automatically",
    )
    parser.add_argument(
        "--triplet_manifest",
        default="",
        help="optional CSV with img0,gt,img1,timestep columns",
    )
    parser.add_argument(
        "--triplet_timestep",
        default=0.5,
        type=float,
        help="default timestep for generated triplets without CSV labels",
    )
    parser.add_argument(
        "--generated_probability",
        default=0.25,
        type=float,
        help="probability of drawing a generated triplet for each sample",
    )
    parser.add_argument("--samples_per_epoch", default=1000, type=int)
    parser.add_argument(
        "--original_start_scale",
        default=0.85 / 0.6,
        type=float,
        help="center zoom applied only to the original dataset's first frame",
    )
    parser.add_argument(
        "--triplet_start_scale",
        default=1.0,
        type=float,
        help="center zoom applied only to generated triplets' first frame",
    )
    parser.add_argument(
        "--train_size",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        default=None,
        help="optional common training resolution",
    )
    parser.add_argument("--no_augment", action="store_true")

    parser.add_argument("--log_dir", default="./ckpt/RIFE_mixed")
    parser.add_argument("--epoch", default=10, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--learning_rate", default=1.0e-5, type=float)
    parser.add_argument("--min_learning_rate", default=1.0e-6, type=float)
    parser.add_argument("--warmup_steps", default=100, type=int)
    parser.add_argument("--save_every", default=5, type=int)
    parser.add_argument("--tensorboard_every", default=10, type=int)
    parser.add_argument("--tensorboard_dir", default="")
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--gpu", default=0, type=int, help="CUDA device for direct python runs")

    parser.add_argument(
        "--init_mode",
        choices=("scratch", "pretrained"),
        default="pretrained",
    )
    parser.add_argument("--pretrained_root", default="./pretrained_dirs")
    parser.add_argument(
        "--pretrained_dir",
        default="",
        help="override pretrained directory; defaults to <pretrained_root>/<model>",
    )
    parser.add_argument(
        "--pretrained_suffix",
        default=None,
        help="load <suffix>_flownet.pkl instead of flownet.pkl",
    )
    parser.add_argument("--resume", default="", help="mixed-training .pth checkpoint")
    parser.add_argument(
        "--local_rank",
        "--local-rank",
        dest="local_rank",
        default=int(os.environ.get("LOCAL_RANK", 0)),
        type=int,
    )
    args = parser.parse_args()

    if not args.triplet_dir and not args.triplet_manifest:
        parser.error("provide --triplet_dir or --triplet_manifest")
    if not 0.0 <= args.generated_probability <= 1.0:
        parser.error("--generated_probability must be in [0, 1]")
    if not 0.0 < args.triplet_timestep < 1.0:
        parser.error("--triplet_timestep must be between 0 and 1")
    if args.samples_per_epoch <= 0 or args.epoch <= 0 or args.batch_size <= 0:
        parser.error("--samples_per_epoch, --epoch and --batch_size must be positive")
    if args.num_workers < 0 or args.warmup_steps < 0:
        parser.error("--num_workers and --warmup_steps must be non-negative")
    if args.save_every <= 0 or args.tensorboard_every <= 0:
        parser.error("--save_every and --tensorboard_every must be positive")
    if args.learning_rate <= 0 or args.min_learning_rate < 0:
        parser.error("learning rates must be positive (minimum may be zero)")
    if args.min_learning_rate > args.learning_rate:
        parser.error("--min_learning_rate cannot exceed --learning_rate")
    if args.original_start_scale < 1.0 or args.triplet_start_scale < 1.0:
        parser.error("start scales must be at least 1.0")
    if args.train_size is not None and any(value <= 0 for value in args.train_size):
        parser.error("--train_size values must be positive")
    if args.model == "IFRNet" and args.batch_size > 1:
        parser.error(
            "The current IFRNet wrapper only supports --batch_size 1 with variable timesteps"
        )
    return args


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("FI training requires a CUDA-capable PyTorch installation")

    launched = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if launched:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = args.local_rank
        model_rank = local_rank
    else:
        rank = 0
        world_size = 1
        local_rank = args.gpu
        model_rank = -1
    is_main = rank == 0
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    process_seed = args.seed + rank
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed_all(process_seed)
    torch.backends.cudnn.benchmark = True

    try:
        if is_main:
            print("Discovering original and generated training data", flush=True)
        original_sequences = discover_original_sequences(args.dataset_dir)
        triplets = (
            load_triplet_manifest(args.triplet_manifest, args.triplet_timestep)
            if args.triplet_manifest
            else discover_triplets(args.triplet_dir, args.triplet_timestep)
        )
        dataset = MixedInterpolationDataset(
            original_sequences,
            triplets,
            args.generated_probability,
            args.samples_per_epoch,
            args.original_start_scale,
            args.triplet_start_scale,
            tuple(args.train_size) if args.train_size else None,
            augment=not args.no_augment,
        )
        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=args.seed,
            )
            if launched
            else None
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=seed_worker,
            collate_fn=checked_collate,
            persistent_workers=args.num_workers > 0,
        )
        if len(loader) == 0:
            raise ValueError("No full batches are available; reduce --batch_size")

        if is_main:
            print(
                "Found {} original sequences and {} generated triplets".format(
                    len(original_sequences), len(triplets)
                ),
                flush=True,
            )
            print(
                "Sampling {:.1f}% original and {:.1f}% generated data".format(
                    100.0 * (1.0 - args.generated_probability),
                    100.0 * args.generated_probability,
                ),
                flush=True,
            )
            print("Initializing {} model".format(args.model), flush=True)

        model = build_model(args.model, model_rank)
        start_epoch = 0
        step = 0
        if args.resume:
            start_epoch, step = load_resume_checkpoint(
                model, args.resume, args.model, device
            )
            if is_main:
                print(
                    "Resumed at epoch {} and step {} from {}".format(
                        start_epoch, step, args.resume
                    ),
                    flush=True,
                )
        elif args.init_mode == "pretrained":
            pretrained_dir = args.pretrained_dir or os.path.join(
                args.pretrained_root, args.model
            )
            weight_path = load_initial_weights(
                model,
                pretrained_dir,
                args.model,
                args.pretrained_suffix,
                device,
            )
            if is_main:
                print("Loaded pretrained weights from {}".format(weight_path))
        elif is_main:
            print("Training from random initialization")

        if start_epoch >= args.epoch:
            raise ValueError(
                "Checkpoint has completed {} epochs; --epoch must be larger.".format(
                    start_epoch
                )
            )

        os.makedirs(args.log_dir, exist_ok=True)
        writer = None
        if is_main:
            tensorboard_dir = args.tensorboard_dir or os.path.join(
                args.log_dir, "tensorboard"
            )
            writer_kwargs = {"log_dir": tensorboard_dir}
            if step > 0:
                writer_kwargs["purge_step"] = step
            writer = SummaryWriter(**writer_kwargs)
            print("TensorBoard directory: {}".format(os.path.abspath(tensorboard_dir)))

        total_steps = args.epoch * len(loader)
        try:
            for epoch in range(start_epoch, args.epoch):
                if sampler is not None:
                    sampler.set_epoch(epoch)
                epoch_l1 = 0.0
                epoch_vgg = 0.0
                epoch_generated = 0.0
                epoch_samples = 0
                progress = tqdm(
                    loader,
                    total=len(loader),
                    desc="Epoch {}/{}".format(epoch + 1, args.epoch),
                    disable=not is_main,
                    dynamic_ncols=True,
                    mininterval=0.5,
                    unit="iter",
                )
                timestamp = time.time()
                for images, timestep, source in progress:
                    data_seconds = time.time() - timestamp
                    images = images.to(device, non_blocking=True).float() / 255.0
                    timestep = timestep.to(device, non_blocking=True)
                    source = source.to(device, non_blocking=True)
                    img_pair = images[:, :6]
                    gt = images[:, 6:9]
                    learning_rate = learning_rate_at(step, total_steps, args)
                    train_start = time.time()
                    _, info = model.update(
                        img_pair,
                        gt,
                        timestep=timestep,
                        learning_rate=learning_rate,
                        training=True,
                    )
                    train_seconds = time.time() - train_start

                    if is_main:
                        loss_l1 = info["loss_l1"].detach().item()
                        loss_vgg = info["loss_vgg"].detach().item()
                        generated_fraction = source.mean().item()
                        epoch_l1 += loss_l1
                        epoch_vgg += loss_vgg
                        epoch_generated += source.sum().item()
                        epoch_samples += source.numel()
                        progress.set_postfix(
                            l1="{:.4e}".format(loss_l1),
                            vgg="{:.4e}".format(loss_vgg),
                            generated="{:.0f}%".format(100.0 * generated_fraction),
                            t="{:.3f}".format(timestep.mean().item()),
                            lr="{:.2e}".format(learning_rate),
                        )
                        if writer is not None and step % args.tensorboard_every == 0:
                            writer.add_scalar("train/loss_l1", loss_l1, step)
                            writer.add_scalar("train/loss_vgg", loss_vgg, step)
                            writer.add_scalar("train/learning_rate", learning_rate, step)
                            writer.add_scalar("train/generated_fraction", generated_fraction, step)
                            writer.add_scalar("train/timestep_mean", timestep.mean().item(), step)
                            writer.add_scalar("time/data_seconds", data_seconds, step)
                            writer.add_scalar("time/train_seconds", train_seconds, step)
                    step += 1
                    timestamp = time.time()

                if is_main:
                    completed_epochs = epoch + 1
                    if writer is not None:
                        writer.add_scalar("epoch/loss_l1", epoch_l1 / len(loader), step)
                        writer.add_scalar("epoch/loss_vgg", epoch_vgg / len(loader), step)
                        writer.add_scalar(
                            "epoch/generated_fraction",
                            epoch_generated / max(1, epoch_samples),
                            step,
                        )
                        writer.flush()

                    latest_path, archive_path = save_training_checkpoint(
                        model, args, epoch, step
                    )
                    print("Saved latest checkpoint to {}".format(latest_path))
                    if archive_path is not None:
                        inference_path = save_inference_weights(model, args.log_dir, epoch)
                        print("Saved checkpoint to {}".format(archive_path))
                        print("Saved inference weights to {}".format(inference_path))
                if launched:
                    dist.barrier()
        finally:
            if writer is not None:
                writer.close()
    finally:
        if launched and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
