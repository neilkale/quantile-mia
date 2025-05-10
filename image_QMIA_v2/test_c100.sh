export CUDA_VISIBLE_DEVICES=4,5,6,7

MODEL_DIR=./models/
DATA_DIR=../image_QMIA_v3/data/
BASE_ARCHITECTURE=cifar-resnet-18 #resnet-50 #
QMIA_ARCHITECTURE=facebook/convnext-tiny-224 #facebook/convnext-large-224-22k-1k #cifar-resnet-18 #

# Set these variables
BASE_DATASET=cifar100/0_16 #imagenet-1k/0_16 #
ATTACK_DATASET=cifar100/0_16 #imagenet-1k/0_16 #
DROPPED_CLASSES=(0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95)

# Train base model
python train_base.py --dataset=$BASE_DATASET --architecture=$BASE_ARCHITECTURE --model_root=$MODEL_DIR --data_root=$DATA_DIR \
--batch_size=32 --scheduler=step --scheduler_step_gamma=0.2 --scheduler_step_fraction=0.3 --lr=0.1 --weight_decay=5e-4 --epochs=100

# Train QMIA
python train_mia.py --attack_dataset=$ATTACK_DATASET --base_model_dataset=$BASE_DATASET --architecture=$QMIA_ARCHITECTURE --base_architecture=$BASE_ARCHITECTURE --model_root=$MODEL_DIR --data_root=$DATA_DIR \
--batch_size=16 --image_size=224 --epochs=10 --score_fn true_margin --loss_fn gaussian --cls_drop "${DROPPED_CLASSES[@]}" --rerun

# Evaluate performance
python evaluate_mia.py --attack_dataset=$ATTACK_DATASET --base_model_dataset=$BASE_DATASET --architecture=$QMIA_ARCHITECTURE --base_architecture=$BASE_ARCHITECTURE --model_root=$MODEL_DIR --data_root=$DATA_DIR \
--batch_size=128 --image_size=224 --score_fn true_margin --loss_fn gaussian --cls_drop "${DROPPED_CLASSES[@]}" --checkpoint best_val_loss --rerun