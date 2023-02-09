#!/usr/bin/env bash

source EXPERIMENTS_ROOT.sh

run_COMIND_EXPERIMENTS()
{
    RESULTS_PATH=$1; shift
    SAVE_IMAGES=$1; shift
    GPUS=($@)


    train_single $SAVE_IMAGES KSDD2 $KSDD2_PATH N_0   $RESULTS_PATH 15 -1 0   50 0.01 1 1 True  2 3 False True  True  ${GPUS[0]} # Figure 9, Table 4-Row 13
    train_single $SAVE_IMAGES KSDD2 $KSDD2_PATH N_16  $RESULTS_PATH 15 -1 16  50 0.01 1 1 True  2 3 True  True  True  ${GPUS[0]} # Figure 9, Table 4-Row 12
    train_single $SAVE_IMAGES KSDD2 $KSDD2_PATH N_53  $RESULTS_PATH 15 -1 53  50 0.01 1 1 True  2 3 True  True  True  ${GPUS[0]} # Figure 9
    train_single $SAVE_IMAGES KSDD2 $KSDD2_PATH N_126 $RESULTS_PATH 15 -1 126 50 0.01 1 1 True  2 3 True  True  True  ${GPUS[0]} # Figure 9
    train_single $SAVE_IMAGES KSDD2 $KSDD2_PATH N_246 $RESULTS_PATH 15 -1 246 50 0.01 1 1 True  2 3 True  True  True  ${GPUS[0]} # Figure 9
    
}


# Space delimited list of GPU IDs which will be used for training
GPUS=(5 6 7)
if [ "${#GPUS[@]}" -eq 0 ]; then
  GPUS=(0)
  #GPUS=(0 1 2) # if more GPUs available
fi

run_COMIND_EXPERIMENTS ./results-comind True "${GPUS[@]}"


