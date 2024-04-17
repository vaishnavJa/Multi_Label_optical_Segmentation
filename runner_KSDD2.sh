#!/bin/bash -l
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --export=NONE
#SBATCH --cluster=tinygpu
#SBATCH --job-name=MSsuface_test
unset SLURM_EXPORT_ENV
module load python/3.8-anaconda
python3 /home/hpc/iwfa/iwfa024h/Multi_Label_optical_Segmentation/train_net.py \
    --GPU=0 \
    --DATASET=KSDD2 \
    --RUN_NAME=F246_KSDD2\
    --DATASET_PATH= \
    --RESULTS_PATH=/home/woody/iwfa/iwfa024h/results2 \
    --SAVE_IMAGES=True \
    --DILATE=0 \
    --EPOCHS=20 \
    --LEARNING_RATE=1 \
    --DELTA_CLS_LOSS=0.1\
    --BATCH_SIZE=1 \
    --WEIGHTED_SEG_LOSS=False \
    --WEIGHTED_SEG_LOSS_P=2  \
    --WEIGHTED_SEG_LOSS_MAX=3  \
    --DYN_BALANCED_LOSS=True \
    --GRADIENT_ADJUSTMENT=True \
    --FREQUENCY_SAMPLING=False \
    --TRAIN_NUM=-1 \
    --NUM_SEGMENTED=246 \
    --USE_BEST_MODEL=True \
    --REPRODUCIBLE_RUN=True \
    --VALIDATE=True \
    --VALIDATE_ON_TEST=False \
    --FOLD=0 \
    --HYPERPARAM=False \
    --SEG_OUTSIZE=1 \
    --DEC_OUTSIZE=1 \
    --SPLIT_LOCATION=/home/hpc/iwfa/iwfa024h/Multi_Label_optical_Segmentation/splits/PA_M/split_280_D2.pyb \