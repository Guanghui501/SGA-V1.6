#!/bin/bash

# 定义要训练的属性列表
PROPERTIES=("bulk_modulus_kv" "shear_modulus_gv")

# 定义随机种子列表
RANDOM_SEEDS=(42 7 123)

# 定义GPU设备
CUDA_DEVICE=3

echo "=========================================="
echo "开始批量训练任务"
echo "属性: ${PROPERTIES[@]}"
echo "随机种子: ${RANDOM_SEEDS[@]}"
echo "总任务数: $((${#PROPERTIES[@]} * ${#RANDOM_SEEDS[@]}))"
echo "=========================================="

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

# 循环遍历每个属性
for PROPERTY in "${PROPERTIES[@]}"; do
    # 循环遍历每个随机种子
    for SEED in "${RANDOM_SEEDS[@]}"; do
        # 创建输出目录名称
        OUTPUT_DIR="./output_100epochs_${SEED}_bs64_sw_ju_onlymiddle_${PROPERTY}"

        # 创建日志文件名（包含时间戳）
        LOG_FILE="${OUTPUT_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

        # 确保输出目录存在
        mkdir -p "$OUTPUT_DIR"

        echo ""
        echo "=========================================="
        echo "启动训练任务:"
        echo "  属性: $PROPERTY"
        echo "  随机种子: $SEED"
        echo "  输出目录: $OUTPUT_DIR"
        echo "  日志文件: $LOG_FILE"
        echo "=========================================="

        nohup python train_with_cross_modal_attention.py \
            --root_dir /public/home/ghzhang/crysmmnet-main/dataset \
            --dataset jarvis \
            --property "$PROPERTY" \
            --train_ratio 0.8 \
            --val_ratio 0.1 \
            --test_ratio 0.1 \
            --batch_size 64 \
            --epochs 100 \
            --learning_rate 5e-4 \
            --weight_decay 1e-3 \
            --warmup_steps 2000 \
            --alignn_layers 4 \
            --gcn_layers 4 \
            --hidden_features 256 \
            --graph_dropout 0.15 \
            --use_cross_modal False \
            --cross_modal_num_heads 2 \
            --use_middle_fusion True \
            --middle_fusion_layers 2 \
            --use_fine_grained_attention False \
            --middle_fusion_dropout 0.35 \
            --fine_grained_hidden_dim 256 \
            --fine_grained_num_heads 8 \
            --fine_grained_dropout 0.35 \
            --fine_grained_use_projection False \
            --early_stopping_patience 150 \
            --output_dir "$OUTPUT_DIR" \
            --num_workers 24 \
            --random_seed "$SEED" \
            > "$LOG_FILE" 2>&1 &

        TASK_PID=$!
        echo "任务已在后台启动，PID: $TASK_PID"
        echo "使用以下命令查看进度:"
        echo "  tail -f $LOG_FILE"

        # 等待几秒钟，避免同时启动太多任务
        sleep 5
    done
done

echo ""
echo "=========================================="
echo "所有训练任务已启动完成！"
echo "使用 'ps aux | grep train_with_cross_modal_attention' 查看运行中的任务"
echo "=========================================="
