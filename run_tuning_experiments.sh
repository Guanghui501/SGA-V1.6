#!/bin/bash
# 参数调优实验脚本:
# 基于训练曲线分析，针对过拟合和融合机制进行系统性调优

# ============================================================================
# 配置通用参数
# ============================================================================


export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=2

DATASET="jarvis"
PROPERTY="bulk_modulus_kv"
ROOT_DIR="/public/home/ghzhang/crysmmnet-main-2/dataset"
BASE_OUTPUT_DIR="./tuning_experiments-full-text"

BATCH_SIZE=128
EPOCHS=300
EARLY_STOPPING=100
NUM_WORKERS=24

# ============================================================================
# Phase 1: 增强正则化（解决过拟合）
# ============================================================================

echo "=========================================="
echo "Phase 1: 增强正则化实验"
echo "=========================================="

# 实验1.1: 保守的正则化增强
python train_with_cross_modal_attention.py \
    --dataset $DATASET \
    --property $PROPERTY \
    --root_dir $ROOT_DIR \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --early_stopping_patience $EARLY_STOPPING \
    --num_workers $NUM_WORKERS \
    --graph_dropout 0.2 \
    --weight_decay 0.001 \
    --cross_modal_dropout 0.15 \
    --use_cross_modal True \
    --use_middle_fusion True \
    --use_fine_grained_attention True \
    --fine_grained_use_projection True \
    --fine_grained_dropout 0.25 \
    --middle_fusion_dropout 0.15 \
    --learning_rate 0.0005 \
    --output_dir $BASE_OUTPUT_DIR/exp1_1_regularization_conservative \
    2>&1 | tee $BASE_OUTPUT_DIR/exp1_1_log.txt

echo "✅ 实验1.1完成"
echo ""

# 实验1.2: 激进的正则化增强
python train_with_cross_modal_attention.py \
    --dataset $DATASET \
    --property $PROPERTY \
    --root_dir $ROOT_DIR \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --early_stopping_patience $EARLY_STOPPING \
    --num_workers $NUM_WORKERS \
    --graph_dropout 0.25 \
    --use_cross_modal True \
    --use_fine_grained_attention True \
    --fine_grained_use_projection True \
    --use_middle_fusion True \
    --weight_decay 0.002 \
    --cross_modal_dropout 0.2 \
    --fine_grained_dropout 0.3 \
    --middle_fusion_dropout 0.2 \
    --learning_rate 0.0005 \
    --output_dir $BASE_OUTPUT_DIR/exp1_2_regularization_aggressive \
    2>&1 | tee $BASE_OUTPUT_DIR/exp1_2_log.txt

echo "✅ 实验1.2完成"
echo ""

# ============================================================================
# Phase 2: 增强融合机制
# ============================================================================

echo "=========================================="
echo "Phase 2: 增强融合机制实验"
echo "=========================================="

# 实验2.1: 双层融合（2,3）+ 增强容量
python train_with_cross_modal_attention.py \
    --dataset $DATASET \
    --property $PROPERTY \
    --root_dir $ROOT_DIR \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --early_stopping_patience $EARLY_STOPPING \
    --num_workers $NUM_WORKERS \
    --middle_fusion_layers "2,3" \
    --middle_fusion_hidden_dim 256 \
    --use_cross_modal True \
    --use_fine_grained_attention True \
    --use_middle_fusion True \
    --fine_grained_use_projection True \
    --middle_fusion_num_heads 4 \
    --middle_fusion_dropout 0.15 \
    --graph_dropout 0.2 \
    --weight_decay 0.001 \
    --cross_modal_dropout 0.15 \
    --fine_grained_dropout 0.25 \
    --learning_rate 0.0005 \
    --output_dir $BASE_OUTPUT_DIR/exp2_1_fusion_2_3 \
    2>&1 | tee $BASE_OUTPUT_DIR/exp2_1_log.txt

echo "✅ 实验2.1完成"
echo ""

# 实验2.2: 双层融合（1,2）+ 增强容量（更早融合）
python train_with_cross_modal_attention.py \
    --dataset $DATASET \
    --property $PROPERTY \
    --root_dir $ROOT_DIR \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --early_stopping_patience $EARLY_STOPPING \
    --num_workers $NUM_WORKERS \
    --middle_fusion_layers "1,2" \
    --use_cross_modal True \
    --use_fine_grained_attention True \
    --use_middle_fusion True \
    --fine_grained_use_projection True \
    --middle_fusion_hidden_dim 256 \
    --middle_fusion_num_heads 4 \
    --middle_fusion_dropout 0.15 \
    --graph_dropout 0.2 \
    --weight_decay 0.001 \
    --cross_modal_dropout 0.15 \
    --fine_grained_dropout 0.25 \
    --learning_rate 0.0005 \
    --output_dir $BASE_OUTPUT_DIR/exp2_2_fusion_1_2 \
    2>&1 | tee $BASE_OUTPUT_DIR/exp2_2_log.txt

echo "✅ 实验2.2完成"
echo ""

# 实验2.3: 三层融合（激进）
python train_with_cross_modal_attention.py \
    --dataset $DATASET \
    --property $PROPERTY \
    --root_dir $ROOT_DIR \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --early_stopping_patience $EARLY_STOPPING \
    --num_workers $NUM_WORKERS \
    --middle_fusion_layers "1,2,3" \
    --middle_fusion_hidden_dim 256 \
    --middle_fusion_num_heads 4 \
    --middle_fusion_dropout 0.15 \
    --graph_dropout 0.2 \
    --weight_decay 0.001 \
    --cross_modal_dropout 0.15 \
    --use_cross_modal True \
    --use_fine_grained_attention True \
    --fine_grained_use_projection True \
    --use_middle_fusion True \
    --fine_grained_dropout 0.25 \
    --learning_rate 0.0005 \
    --output_dir $BASE_OUTPUT_DIR/exp2_3_fusion_1_2_3 \
    2>&1 | tee $BASE_OUTPUT_DIR/exp2_3_log.txt

echo "✅ 实验2.3完成"
echo ""

# ============================================================================
# Phase 3: 减弱全局注意力（测试是否抵消融合效果）
# ============================================================================

echo "=========================================="
echo "Phase 3: 调整全局注意力实验"
echo "=========================================="

# 实验3.1: 减少注意力头数
python train_with_cross_modal_attention.py \
    --dataset $DATASET \
    --property $PROPERTY \
    --root_dir $ROOT_DIR \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --early_stopping_patience $EARLY_STOPPING \
    --num_workers $NUM_WORKERS \
    --middle_fusion_layers "2,3" \
    --middle_fusion_hidden_dim 256 \
    --middle_fusion_num_heads 4 \
    --use_cross_modal True \
    --use_fine_grained_attention True \
    --fine_grained_use_projection True \
    --use_middle_fusion True \
    --cross_modal_num_heads 2 \
    --graph_dropout 0.2 \
    --weight_decay 0.001 \
    --learning_rate 0.0005 \
    --output_dir $BASE_OUTPUT_DIR/exp3_1_weak_cross_attention \
    2>&1 | tee $BASE_OUTPUT_DIR/exp3_1_log.txt

echo "✅ 实验3.1完成"
echo ""

# 实验3.2: 关闭全局注意力（测试极端情况）
python train_with_cross_modal_attention.py \
    --dataset $DATASET \
    --property $PROPERTY \
    --root_dir $ROOT_DIR \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --early_stopping_patience $EARLY_STOPPING \
    --num_workers $NUM_WORKERS \
    --use_cross_modal True \
    --use_fine_grained_attention True \
    --fine_grained_use_projection True \
    --use_middle_fusion True \
    --middle_fusion_layers "2,3" \
    --middle_fusion_hidden_dim 256 \
    --middle_fusion_num_heads 4 \
    --use_cross_modal_attention 0 \
    --graph_dropout 0.2 \
    --weight_decay 0.001 \
    --learning_rate 0.0005 \
    --output_dir $BASE_OUTPUT_DIR/exp3_2_no_cross_attention \
    2>&1 | tee $BASE_OUTPUT_DIR/exp3_2_log.txt

echo "✅ 实验3.2完成"
echo ""

# ============================================================================
# Phase 4: Batch Size实验
# ============================================================================

echo "=========================================="
echo "Phase 4: Batch Size实验"
echo "=========================================="

# 实验4.1: 减小batch size
python train_with_cross_modal_attention.py \
    --dataset $DATASET \
    --property $PROPERTY \
    --root_dir $ROOT_DIR \
    --batch_size 64 \
    --epochs $EPOCHS \
    --early_stopping_patience $EARLY_STOPPING \
    --num_workers $NUM_WORKERS \
    --middle_fusion_layers "2,3" \
    --middle_fusion_hidden_dim 256 \
    --use_cross_modal True \
    --use_fine_grained_attention True \
    --use_middle_fusion True \
    --fine_grained_use_projection True \
    --middle_fusion_num_heads 4 \
    --graph_dropout 0.2 \
    --weight_decay 0.001 \
    --learning_rate 0.0005 \
    --output_dir $BASE_OUTPUT_DIR/exp4_1_batch_size_64 \
    2>&1 | tee $BASE_OUTPUT_DIR/exp4_1_log.txt

echo "✅ 实验4.1完成"
echo ""

# ============================================================================
# Phase 5: 最优组合（基于前面实验的最佳配置）
# ============================================================================

echo "=========================================="
echo "Phase 5: 最优组合实验"
echo "=========================================="

# 实验5.1: 综合最佳配置
python train_with_cross_modal_attention.py \
    --dataset $DATASET \
    --property $PROPERTY \
    --root_dir $ROOT_DIR \
    --batch_size 64 \
    --epochs $EPOCHS \
    --early_stopping_patience $EARLY_STOPPING \
    --num_workers $NUM_WORKERS \
    --middle_fusion_layers "2,3" \
    --middle_fusion_hidden_dim 256 \
    --middle_fusion_num_heads 4 \
    --middle_fusion_dropout 0.15 \
    --cross_modal_num_heads 2 \
    --graph_dropout 0.25 \
    --weight_decay 0.001 \
    --cross_modal_dropout 0.15 \
    --fine_grained_dropout 0.25 \
    --learning_rate 0.0005 \
    --use_cross_modal True \
    --use_fine_grained_attention True \
    --use_middle_fusion True \
    --fine_grained_use_projection True \
    --output_dir $BASE_OUTPUT_DIR/exp5_1_best_combined \
    2>&1 | tee $BASE_OUTPUT_DIR/exp5_1_log.txt

echo "✅ 实验5.1完成"
echo ""

# ============================================================================
# 生成实验对比报告
# ============================================================================

echo "=========================================="
echo "生成实验对比报告"
echo "=========================================="

python compare_experiments.py \
    --experiment_dirs $BASE_OUTPUT_DIR/exp* \
    --save_dir $BASE_OUTPUT_DIR/comparison_report

echo ""
echo "🎉 所有调优实验完成！"
echo ""
echo "实验结果保存在: $BASE_OUTPUT_DIR"
echo ""
echo "下一步："
echo "1. 查看各实验的训练曲线"
echo "2. 对比验证集MAE"
echo "3. 运行CKA分析查看融合效果"
echo "4. 选择最佳配置进行正式训练"
