import os
import cv2
import math
import time
import torch
import torch.distributed as dist
import numpy as np
import random
import argparse
import shutil
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from model.FI_models.EDSCVgg import Model as EDSC
from model.FI_models.IFRNetVgg import Model as IFRNet
from model.FI_models.RIFEVgg import Model as RIFE
from model.FI_models.AMTVgg import Model as AMT
from model.FI_models.UPRNetVgg import Model as UPRNet
from model.FI_models.EMAVFIVgg import Model as EMAVFI

from dataset import *
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset import DZDataset

device = torch.device("cuda")


def get_learning_rate(step):
    if step < 2000:
        mul = step / 2000.
        return 3e-4 * mul
    else:
        mul = np.cos((step - 2000) / (args.epoch * args.step_per_epoch - 2000.) * math.pi) * 0.5 + 0.5
        return (3e-4 - 3e-6) * mul + 3e-6


def get_flownet_module(model):
    return model.flownet.module if hasattr(model.flownet, "module") else model.flownet


def load_torch_checkpoint(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save_training_checkpoint(model, log_dir, epoch, step, local_rank):
    if local_rank != 0:
        return

    checkpoint = {
        "format_version": 1,
        "model_name": args.model,
        "model_state": get_flownet_module(model).state_dict(),
        "optimizer_state": model.optimG.state_dict(),
        "epoch": epoch,
        "step": step,
        "training_args": vars(args).copy(),
    }

    latest_path = os.path.join(log_dir, "checkpoint_latest.pth")
    latest_temp_path = latest_path + ".tmp"
    torch.save(checkpoint, latest_temp_path)
    os.replace(latest_temp_path, latest_path)
    print("Saved latest training checkpoint to {}".format(latest_path))

    completed_epochs = epoch + 1
    if completed_epochs % args.save_every == 0 or completed_epochs == args.epoch:
        archive_path = os.path.join(
            log_dir, "checkpoint_epoch_{:04d}.pth".format(completed_epochs)
        )
        shutil.copyfile(latest_path, archive_path)
        print("Saved training checkpoint to {}".format(archive_path))


def resume_training_checkpoint(model, checkpoint_path, local_rank):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError("Resume checkpoint not found: {}".format(checkpoint_path))

    checkpoint_device = torch.device("cuda", local_rank)
    checkpoint = load_torch_checkpoint(checkpoint_path, map_location=checkpoint_device)
    required_keys = {"model_name", "model_state", "optimizer_state", "epoch", "step"}
    missing_keys = required_keys.difference(checkpoint)
    if missing_keys:
        raise ValueError(
            "Checkpoint {} is missing keys: {}".format(
                checkpoint_path, ", ".join(sorted(missing_keys))
            )
        )
    if checkpoint["model_name"] != args.model:
        raise ValueError(
            "Checkpoint model '{}' does not match --model '{}'.".format(
                checkpoint["model_name"], args.model
            )
        )

    get_flownet_module(model).load_state_dict(checkpoint["model_state"])
    model.optimG.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = int(checkpoint["epoch"]) + 1
    step = int(checkpoint["step"])

    if local_rank == 0:
        saved_args = checkpoint.get("training_args", {})
        for key in ("world_size", "batch_size", "epoch"):
            saved_value = saved_args.get(key)
            current_value = getattr(args, key)
            if saved_value is not None and saved_value != current_value:
                print(
                    "Warning: checkpoint {}={} but current {}={}; "
                    "the learning-rate schedule may change.".format(
                        key, saved_value, key, current_value
                    )
                )
        print(
            "Resumed {} from {} completed epochs (step {}) using {}".format(
                args.model, start_epoch, step, checkpoint_path
            )
        )
    return start_epoch, step

def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    
    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)

def train(model, data_root, log_dir, local_rank, start_epoch=0, step=0, writer=None):
    dataset = DZDataset('train', data_root=data_root)
    sampler = DistributedSampler(dataset)
    train_data = DataLoader(dataset, batch_size=args.batch_size, num_workers=2, pin_memory=True, drop_last=True, sampler=sampler)
    args.step_per_epoch = train_data.__len__()
    if start_epoch >= args.epoch:
        raise ValueError(
            "Checkpoint already completed {} epochs; --epoch must be greater than {}.".format(
                start_epoch, start_epoch
            )
        )

    for epoch in range(start_epoch, args.epoch):
        sampler.set_epoch(epoch)
        time_stamp = time.time()
        epoch_loss_l1 = 0.0
        epoch_loss_vgg = 0.0
        progress_bar = tqdm(
            enumerate(train_data),
            total=args.step_per_epoch,
            desc="Epoch {}/{}".format(epoch + 1, args.epoch),
            disable=local_rank != 0,
            dynamic_ncols=True,
            mininterval=0.5,
            unit="iter",
        )
        for i, data in progress_bar:
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            data_gpu, timestep = data

            b, t, c, h, w = data_gpu.shape
            data_gpu = data_gpu.view(-1, c, h, w)
            timestep = timestep.view(-1, timestep.shape[-3], timestep.shape[-2], timestep.shape[-1])

            data_gpu = data_gpu.to(device, non_blocking=True) / 255.
            timestep = timestep.to(device, non_blocking=True)
            imgs = data_gpu[:, :6]
            gt = data_gpu[:, 6:9]
            learning_rate = get_learning_rate(step) * args.world_size / 4
            _, info = model.update(imgs, gt, timestep=timestep, learning_rate=learning_rate, training=True) # pass timestep if you are training RIFEm
            
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()

            if local_rank == 0:
                loss_l1 = info['loss_l1'].detach().item()
                loss_vgg = info['loss_vgg'].detach().item()
                epoch_loss_l1 += loss_l1
                epoch_loss_vgg += loss_vgg
                progress_bar.set_postfix(
                    l1="{:.4e}".format(loss_l1),
                    vgg="{:.4e}".format(loss_vgg),
                    lr="{:.2e}".format(learning_rate),
                    data="{:.2f}s".format(data_time_interval),
                    train="{:.2f}s".format(train_time_interval),
                )
                if writer is not None and step % args.tensorboard_every == 0:
                    writer.add_scalar("train/loss_l1", loss_l1, step)
                    writer.add_scalar("train/loss_vgg", loss_vgg, step)
                    writer.add_scalar("train/learning_rate", learning_rate, step)
                    writer.add_scalar("time/data_seconds", data_time_interval, step)
                    writer.add_scalar("time/train_seconds", train_time_interval, step)
            step += 1
        completed_epochs = epoch + 1
        if writer is not None:
            writer.add_scalar(
                "epoch/loss_l1", epoch_loss_l1 / args.step_per_epoch, step
            )
            writer.add_scalar(
                "epoch/loss_vgg", epoch_loss_vgg / args.step_per_epoch, step
            )
            writer.add_scalar("epoch/completed", completed_epochs, step)
            writer.flush()
        should_archive = completed_epochs % args.save_every == 0 or completed_epochs == args.epoch
        if should_archive:
            model.save_model(log_dir, local_rank, suffix=str(epoch))
            if local_rank == 0:
                inference_path = os.path.join(log_dir, "{}_flownet.pkl".format(epoch))
                print(
                    "[Epoch {}/{}] Saved inference weights to {}".format(
                        completed_epochs, args.epoch, inference_path
                    )
                )
        save_training_checkpoint(model, log_dir, epoch, step, local_rank)
        dist.barrier()

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="train FI model")
    parser.add_argument(
        "--model",
        type=str,
        default="RIFE",
        choices=("EDSC", "IFRNet", "RIFE", "AMT", "UPRNet", "EMAVFI"),
        help="FI model",
    )
    parser.add_argument("--log_dir", type=str, default="./ckpt/RIFE_finetuned", help="log path")
    parser.add_argument("--dataset_dir", type=str, default="../dataset/DCSZ_dataset/DCSZ_syn", help="train data path")
    parser.add_argument(
        "--init_mode",
        choices=("scratch", "pretrained"),
        default="scratch",
        help="initialize the FI network randomly or load its original pretrained weights",
    )
    parser.add_argument(
        "--pretrained_root",
        type=str,
        default="./pretrained_dirs",
        help="root containing one pretrained subdirectory per FI model",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="full .pth checkpoint path; --epoch remains the total target epoch count",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=5,
        help="archive a numbered .pth checkpoint every N completed epochs",
    )
    parser.add_argument(
        "--tensorboard_dir",
        type=str,
        default="",
        help="TensorBoard event directory; defaults to <log_dir>/tensorboard",
    )
    parser.add_argument(
        "--tensorboard_every",
        type=int,
        default=10,
        help="write iteration metrics every N optimizer steps",
    )
    parser.add_argument('--epoch', default=100, type=int, help='total target epoch count')
    parser.add_argument('--batch_size', default=1, type=int, help='minibatch size')
    parser.add_argument(
        '--world_size',
        default=int(os.environ.get("WORLD_SIZE", 1)),
        type=int,
        help='world size',
    )
    parser.add_argument(
        '--local_rank',
        '--local-rank',
        dest='local_rank',
        default=int(os.environ.get("LOCAL_RANK", 0)),
        type=int,
        help='local rank',
    )
    args = parser.parse_args()

    if args.save_every <= 0:
        parser.error("--save_every must be greater than 0")
    if args.tensorboard_every <= 0:
        parser.error("--tensorboard_every must be greater than 0")

    torch.distributed.init_process_group(backend="nccl", world_size=args.world_size)

    torch.cuda.set_device(args.local_rank)
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    model_classes = {
        "EDSC": EDSC,
        "IFRNet": IFRNet,
        "RIFE": RIFE,
        "AMT": AMT,
        "UPRNet": UPRNet,
        "EMAVFI": EMAVFI,
    }
    model = model_classes[args.model](args.local_rank)

    start_epoch = 0
    step = 0
    if args.resume:
        start_epoch, step = resume_training_checkpoint(model, args.resume, args.local_rank)
    elif args.init_mode == "pretrained":
        pretrained_dir = os.path.join(args.pretrained_root, args.model)
        model.load_pretrained_model(pretrained_dir, rank=args.local_rank)
        if args.local_rank == 0:
            print("Loaded pretrained {} weights from {}".format(args.model, pretrained_dir))
    elif args.local_rank == 0:
        print("Training {} from random initialization".format(args.model))

    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    data_root = args.dataset_dir
    writer = None
    if args.local_rank == 0:
        tensorboard_dir = args.tensorboard_dir or os.path.join(log_dir, "tensorboard")
        writer_options = {"log_dir": tensorboard_dir}
        if start_epoch > 0:
            writer_options["purge_step"] = step
        writer = SummaryWriter(**writer_options)
        print("TensorBoard events will be saved to {}".format(os.path.abspath(tensorboard_dir)))

    try:
        train(
            model,
            data_root,
            log_dir,
            args.local_rank,
            start_epoch=start_epoch,
            step=step,
            writer=writer,
        )
    finally:
        if writer is not None:
            writer.close()
        
