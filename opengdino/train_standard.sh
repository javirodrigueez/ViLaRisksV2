GPU_NUM=$1
CFG=$2
DATASETS=$3
OUTPUT_DIR=$4
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# Change ``pretrain_model_path`` to use a different pretrain. 
# (e.g. GroundingDINO pretrain, DINO pretrain, Swin Transformer pretrain.)
# If you don't want to use any pretrained model, just ignore this parameter.

python main.py \
        --output_dir ${OUTPUT_DIR} \
        -c ${CFG} \
        --datasets ${DATASETS}  \
        --pretrain_model_path ./logs_train3/checkpoint_best_regular.pth \
        --options text_encoder_type=weights/bert-base-uncased
