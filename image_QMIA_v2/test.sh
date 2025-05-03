export CUDA_VISIBLE_DEVICES=4,5,6,7

MODEL_DIR=./models/
DATA_DIR=./data/
BASE_ARCHITECTURE=cifar-resnet-50
QMIA_ARCHITECTURE=facebook/convnext-large-224-22k-1k #convnext-tiny #facebook/convnext-large-224-22k-1k #

# Set these variables
BASE_DATASET=cinic10/0_16 #imagenet-1k/0_16 #
ATTACK_DATASET=cinic10/0_16 #imagenet-1k/0_16 #
DROPPED_CLASSES=''

# Train base model
python train_base.py --dataset=$BASE_DATASET --scheduler=step --scheduler_step_gamma=0.2 --scheduler_step_fraction=0.3 --lr=0.1 --weight_decay=5e-4 --batch_size=1024 --image_size=32 --architecture=$BASE_ARCHITECTURE --epochs=100 --model_root=$MODEL_DIR --data_root=$DATA_DIR

# Train QMIA
python train_mia.py --attack_dataset=$ATTACK_DATASET --base_model_dataset=$BASE_DATASET --architecture=$QMIA_ARCHITECTURE --base_architecture=$BASE_ARCHITECTURE --batch_size=128 --image_size=224 --epochs=10 --score_fn top_two_margin --loss_fn gaussian --cls_drop $DROPPED_CLASSES

# Evaluate performance
python evaluate_mia.py --attack_dataset=$ATTACK_DATASET --base_model_dataset=$BASE_DATASET --architecture=$QMIA_ARCHITECTURE --base_architecture=$BASE_ARCHITECTURE --batch_size=512 --image_size=224 --score_fn top_two_margin --loss_fn gaussian --cls_drop $DROPPED_CLASSES