export CUDA_VISIBLE_DEVICES=4,5,6,7

MODEL_DIR=./models/
DATA_DIR=../image_QMIA_v3/data/
BASE_ARCHITECTURE=resnet-50 #cifar-resnet-50 #
QMIA_ARCHITECTURE=convnext-tiny #facebook/convnext-large-224-22k-1k #cifar-resnet-18 #

# Set these variables
BASE_DATASET=imagenet-1k/0_16 #cinic10/0_16 #
ATTACK_DATASET=imagenet-1k/0_16 #cinic10/0_16 #
DROPPED_CLASSES=''

# Train base model
python train_base.py --dataset=$BASE_DATASET --architecture=$BASE_ARCHITECTURE --model_root=$MODEL_DIR --data_root=$DATA_DIR \
--batch_size=32 --scheduler=step --scheduler_step_gamma=0.2 --scheduler_step_fraction=0.3 --lr=0.1 --weight_decay=5e-4 --image_size=224 --epochs=200

# Train QMIA
python train_mia.py --attack_dataset=$ATTACK_DATASET --base_model_dataset=$BASE_DATASET --architecture=$QMIA_ARCHITECTURE --base_architecture=$BASE_ARCHITECTURE --model_root=$MODEL_DIR --data_root=$DATA_DIR \
--batch_size=16 --image_size=64 --epochs=10 --score_fn top_two_margin --loss_fn gaussian --cls_drop $DROPPED_CLASSES --rerun

# Evaluate performance
python evaluate_mia.py --attack_dataset=$ATTACK_DATASET --base_model_dataset=$BASE_DATASET --architecture=$QMIA_ARCHITECTURE --base_architecture=$BASE_ARCHITECTURE --model_root=$MODEL_DIR --data_root=$DATA_DIR \
--batch_size=128 --image_size=64 --score_fn top_two_margin --loss_fn gaussian --cls_drop $DROPPED_CLASSES --rerun