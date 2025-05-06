export CUDA_VISIBLE_DEVICES=4,5,6,7

MODEL_DIR=./models/
DATA_DIR=../image_QMIA_v3/data/
BASE_ARCHITECTURE=cifar-resnet-50 #resnet-50 #
QMIA_ARCHITECTURE=facebook/convnext-large-224-22k-1k #cifar-resnet-18 #cifar-resnet-18 #convnext-tiny #

# Set these variables
BASE_DATASET=cinic10/0_16 #imagenet-1k/0_16 #
ATTACK_DATASET=cinic10/0_16 #imagenet-1k/0_16 #
DROPPED_CLASSES=(0 1 2 3 4 5 6 7 8)

# Evaluate performance
python evaluate_mia.py --attack_dataset=$ATTACK_DATASET --base_model_dataset=$BASE_DATASET --architecture=$QMIA_ARCHITECTURE --base_architecture=$BASE_ARCHITECTURE --batch_size=128 --image_size=224 --score_fn top_two_margin --loss_fn gaussian --cls_drop "${DROPPED_CLASSES[@]}"

# python explore_conv_maps.py --attack_dataset=$ATTACK_DATASET --base_model_dataset=$BASE_DATASET --architecture=$QMIA_ARCHITECTURE --base_architecture=$BASE_ARCHITECTURE --image_size=224 --score_fn top_two_margin --loss_fn gaussian --cls_drop $DROPPED_CLASSES