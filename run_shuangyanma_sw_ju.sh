#!/bin/bash

# 创建日志文件名（包含时间戳）
LOG_FILE="./output_100epochs_42_bs128_sw_ju_bulk_dro/train_$(date +%Y%m%d_%H%M%S).log"

# 确保输出目录存在
mkdir -p ./output_100epochs_42_bs128_sw_ju_bulk_dro

echo "=========================================="
echo "开始细粒度注意力模型训练"
echo "日志文件: $LOG_FILE"
echo "=========================================="


export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0

nohup python train_with_cross_modal_attention.py \
    --root_dir /public/home/ghzhang/crysmmnet-main/dataset \
    --dataset jarvis \
    --property bulk_modulus_kv \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    --batch_size 64 \
    --epochs 300 \
    --learning_rate 5e-4 \
    --weight_decay 1e-3 \
    --warmup_steps 2000 \
    --alignn_layers 4 \
    --gcn_layers 4 \
    --hidden_features 256 \
    --graph_dropout 0.35 \
    --use_cross_modal True \
    --cross_modal_num_heads 2 \
    --use_middle_fusion True \
    --middle_fusion_layers 4 \
    --use_fine_grained_attention True \
    --middle_fusion_dropout 0.35 \
    --fine_grained_hidden_dim 256 \
    --fine_grained_num_heads 8 \
    --fine_grained_dropout 0.35 \
    --fine_grained_use_projection True \
    --early_stopping_patience 150 \
    --output_dir ./output_100epochs_42_bs128_sw_ju_bulk_dro \
    --num_workers 24 \
    --random_seed 42 \
    > "$LOG_FILE" 2>&1 &

echo "=========================================="
echo "训练已在后台启动，PID: $!"
echo "使用以下命令查看进度:"
echo "  tail -f $LOG_FILE"
