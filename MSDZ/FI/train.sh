#!/bin/sh
set -eu

# Select with: sh train.sh <mode>
# Models: EDSC, IFRNet, RIFE, AMT, UPRNet, EMAVFI
MODE="${1:-single-scratch}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-RIFE}"
DATASET_DIR="${DATASET_DIR:-../dataset/DCSZ_dataset/DCSZ_syn}"
LOG_DIR="${LOG_DIR:-./ckpt/${MODEL}_scratch}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SAVE_EVERY="${SAVE_EVERY:-5}"
TENSORBOARD_EVERY="${TENSORBOARD_EVERY:-10}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-}"
MASTER_PORT="${MASTER_PORT:-29502}"
SINGLE_GPU="${SINGLE_GPU:-0}"
MULTI_GPUS="${MULTI_GPUS:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
PRETRAINED_ROOT="${PRETRAINED_ROOT:-./pretrained_dirs}"
RESUME_PATH="${RESUME_PATH:-${LOG_DIR}/checkpoint_latest.pth}"

run_modern() {
    visible_gpus="$1"
    process_count="$2"
    shift 2

    CUDA_VISIBLE_DEVICES="$visible_gpus" "$PYTHON_BIN" -m torch.distributed.run \
        --nproc-per-node="$process_count" \
        --master-port="$MASTER_PORT" \
        train.py \
        --model "$MODEL" \
        --dataset_dir "$DATASET_DIR" \
        --log_dir "$LOG_DIR" \
        --epoch "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" \
        --save_every "$SAVE_EVERY" \
        --tensorboard_every "$TENSORBOARD_EVERY" \
        --tensorboard_dir "$TENSORBOARD_DIR" \
        --world_size "$process_count" \
        "$@"
}

run_legacy_single() {
    CUDA_VISIBLE_DEVICES="$SINGLE_GPU" "$PYTHON_BIN" -m torch.distributed.launch \
        --nproc_per_node=1 \
        --master_port="$MASTER_PORT" \
        train.py \
        --model "$MODEL" \
        --init_mode scratch \
        --dataset_dir "$DATASET_DIR" \
        --log_dir "$LOG_DIR" \
        --epoch "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" \
        --save_every "$SAVE_EVERY" \
        --tensorboard_every "$TENSORBOARD_EVERY" \
        --tensorboard_dir "$TENSORBOARD_DIR" \
        --world_size 1
}

require_resume_checkpoint() {
    if [ ! -f "$RESUME_PATH" ]; then
        echo "Resume checkpoint not found: $RESUME_PATH" >&2
        exit 1
    fi
}

case "$MODE" in
    single-scratch)
        run_modern "$SINGLE_GPU" 1 --init_mode scratch
        ;;
    single-resume)
        require_resume_checkpoint
        run_modern "$SINGLE_GPU" 1 --resume "$RESUME_PATH"
        ;;
    single-pretrained)
        run_modern "$SINGLE_GPU" 1 \
            --init_mode pretrained \
            --pretrained_root "$PRETRAINED_ROOT"
        ;;
    multi-scratch)
        run_modern "$MULTI_GPUS" "$NPROC_PER_NODE" --init_mode scratch
        ;;
    multi-resume)
        require_resume_checkpoint
        run_modern "$MULTI_GPUS" "$NPROC_PER_NODE" --resume "$RESUME_PATH"
        ;;
    legacy-single-scratch)
        run_legacy_single
        ;;
    *)
        echo "Usage: sh train.sh <mode>"
        echo ""
        echo "Modes:"
        echo "  single-scratch        Single-GPU random initialization (default)"
        echo "  single-resume         Single-GPU resume from checkpoint_latest.pth"
        echo "  single-pretrained     Single-GPU pretrained fine-tuning"
        echo "  multi-scratch         Multi-GPU random initialization"
        echo "  multi-resume          Multi-GPU resume"
        echo "  legacy-single-scratch Legacy launcher fallback"
        echo ""
        echo "Common overrides:"
        echo "  DATASET_DIR=/data/fi LOG_DIR=./ckpt/run01 EPOCHS=50 NUM_WORKERS=0 sh train.sh single-scratch"
        echo "  RESUME_PATH=./ckpt/run01/checkpoint_epoch_0020.pth sh train.sh single-resume"
        exit 2
        ;;
esac
