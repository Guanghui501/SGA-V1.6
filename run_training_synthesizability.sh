#!/bin/bash
# 可合成性分类训练脚本
# 用于预测材料是否可合成（二分类任务）

# 创建日志文件名（包含时间戳）
LOG_FILE="./output_synthesizability_classification_8w4/train_$(date +%Y%m%d_%H%M%S).log"

# 确保输出目录存在
mkdir -p ./output_synthesizability_classification_8w4


echo "=========================================="
echo "开始可合成性分类模型训练"
echo "任务类型: 二分类"
echo "数据规模: 根据数据集确定"
echo "=========================================="


export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=2

# 检查数据集属性参数
PROPERTY=${1:-"classification"}  # 默认属性名，可通过参数修改

echo "预测属性: $PROPERTY"
echo ""

nohup python train_with_cross_modal_attention.py \
    --root_dir /public/home/ghzhang/crysmmnet-main/dataset \
    --dataset class \
    --property $PROPERTY \
    \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    \
    --batch_size 64 \
    --epochs 100 \
    --learning_rate 1e-3 \
    --weight_decay 5e-4 \
    --warmup_steps 2000 \
    \
    --alignn_layers 4 \
    --gcn_layers 4 \
    --hidden_features 256 \
    --graph_dropout 0.2 \
    \
    --use_middle_fusion True \
    --middle_fusion_layers 2,3 \
    --middle_fusion_hidden_dim 256 \
    --middle_fusion_num_heads 4 \
    --middle_fusion_dropout 0.15 \
    \
    --use_fine_grained_attention True \
    --fine_grained_hidden_dim 256 \
    --fine_grained_num_heads 8 \
    --fine_grained_dropout 0.25 \
    --fine_grained_use_projection True \
    \
    --use_cross_modal 1 \
    --cross_modal_hidden_dim 256 \
    --cross_modal_num_heads 4 \
    --cross_modal_dropout 0.15 \
    \
    --classification 1 \
    --classification_threshold 0.5 \
    \
    --output_dir ./output_synthesizability_classification_8w4 \
    --num_workers 24 \
    --random_seed 42 \
    > "$LOG_FILE" 2>&1 &

echo "=========================================="
echo "训练完成！"
echo "模型保存在: ./output_synthesizability_classification/"
echo "=========================================="
