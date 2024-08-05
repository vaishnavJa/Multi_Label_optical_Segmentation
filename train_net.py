
from end2end import End2End 
import os
from data.dataset_catalog import get_dataset
import argparse
from config import Config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
import time
import joblib


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--GPU', type=int, required=True, help="ID of GPU used for training/evaluation.")
    parser.add_argument('--RUN_NAME', type=str, required=True, help="Name of the run, used as directory name for storing results.")
    parser.add_argument('--DATASET', type=str, required=True, help="Which dataset to use.")
    parser.add_argument('--DATASET_PATH', type=str, required=True, help="Path to the dataset.")

    parser.add_argument('--EPOCHS', type=int, required=True, help="Number of training epochs.")

    parser.add_argument('--LEARNING_RATE', type=float, required=True, help="Learning rate.")
    parser.add_argument('--DELTA_CLS_LOSS', type=float, required=True, help="Weight delta for classification loss.")

    parser.add_argument('--BATCH_SIZE', type=int, required=True, help="Batch size for training.")

    parser.add_argument('--WEIGHTED_SEG_LOSS', type=str2bool, required=True, help="Whether to use weighted segmentation loss.")
    parser.add_argument('--WEIGHTED_SEG_LOSS_P', type=float, required=False, default=None, help="Degree of polynomial for weighted segmentation loss.")
    parser.add_argument('--WEIGHTED_SEG_LOSS_MAX', type=float, required=False, default=None, help="Scaling factor for weighted segmentation loss.")
    parser.add_argument('--DYN_BALANCED_LOSS', type=str2bool, required=True, help="Whether to use dynamically balanced loss.")
    parser.add_argument('--GRADIENT_ADJUSTMENT', type=str2bool, required=True, help="Whether to use gradient adjustment.")
    parser.add_argument('--FREQUENCY_SAMPLING', type=str2bool, required=False, help="Whether to use frequency-of-use based sampling.")

    parser.add_argument('--DILATE', type=int, required=False, default=None, help="Size of dilation kernel for labels")

    parser.add_argument('--FOLD', type=int, default=None, help="Which fold (KSDD) or class (DAGM) to train.")
    parser.add_argument('--TRAIN_NUM', type=int, default=None, help="Number of positive training samples for KSDD or STEEL.")
    parser.add_argument('--NUM_SEGMENTED', type=int, required=True, default=None, help="Number of segmented positive  samples.")
    parser.add_argument('--RESULTS_PATH', type=str, default=None, help="Directory to which results are saved.")

    parser.add_argument('--VALIDATE', type=str2bool, default=None, help="Whether to validate during training.")
    parser.add_argument('--VALIDATE_ON_TEST', type=str2bool, default=None, help="Whether to validate on test set.")
    parser.add_argument('--VALIDATION_N_EPOCHS', type=int, default=None, help="Number of epochs between consecutive validation runs.")
    parser.add_argument('--USE_BEST_MODEL', type=str2bool, default=None, help="Whether to use the best model according to validation metrics for evaluation.")

    parser.add_argument('--ON_DEMAND_READ', type=str2bool, default=None, help="Whether to use on-demand read of data from disk instead of storing it in memory.")
    parser.add_argument('--REPRODUCIBLE_RUN', type=str2bool, default=None, help="Whether to fix seeds and disable CUDA benchmark mode.")

    parser.add_argument('--MEMORY_FIT', type=int, default=None, help="How many images can be fitted in GPU memory.")
    parser.add_argument('--SAVE_IMAGES', type=str2bool, default=None, help="Save test images or not.")
    parser.add_argument('--HYPERPARAM', type=str2bool, required=True,  default=False, help="Whether running optuna.")
    parser.add_argument('--DEC_OUTSIZE', type=int, default=1, help="decision network output shape")
    parser.add_argument('--SEG_OUTSIZE', type=int, default=1, help="segmentation network output shape.")
    parser.add_argument('--DOWN_FACTOR', type=int, default=8, help="model dependent segmentation size reduction factor.")
    parser.add_argument('--SPLIT_LOCATION', type=str, default='', help="split location.")
    parser.add_argument('--MULTISEG', type=str2bool, default=False, help="split location.")
    parser.add_argument('--MULTIDEC', type=str2bool, default=False, help="split location.")
    parser.add_argument('--CLASSWEIGHTS', type=str2bool, default=False, help="split location.")
    parser.add_argument('--EARLYSTOP', type=int, default=False, help="split location.")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    
    # exit()
    args = parse_args()

    """KSDD2 training"""

    # param_dict = {"RUN_NAME":["N_246","N_126","N_53","N_16","N_0"],
    #                 "NUM_SEGMENTED":[246,126,53,16,0],
    #                 "DYN_BALANCED_LOSS":[True,True,True,True,False]}

    # for i in range(5):
    #     for key in param_dict.keys():

    #         vars(args)[key] = param_dict[key][i]
    #         # args.RESULTS_PATH = os.path.join("E:/AI/FAPS/code/Mixedsupervision/","results/KSDD2_",str(i))
    #     configuration = Config()
    #     configuration.merge_from_args(args)
    #     configuration.init_extra()

    #     end2end = End2End(cfg=configuration)
    #     end2end.train()
    #     device = end2end._get_device()
    #     model = end2end._get_model().to(device)
    #     end2end._set_results_path()
    #     optimizer = end2end._get_optimizer(model)
    #     loss_seg, loss_dec = end2end._get_loss(True), end2end._get_loss(False)    
    #     end2end.set_dec_gradient_multiplier(model, 0.0)
    #     end2end.eval(model, device, True, False, True)
            



    """hyper param selection """

    

    if args.HYPERPARAM:

        def objective(trial):

            # vars(args)['DILATE'] = trail.suggest_int('DILATE',1,17)
            # vars(args)['EPOCHS'] = trial.suggest_int('EPOCHS',70,100)
            vars(args)["LEARNING_RATE"] = trial.suggest_float("LEARNING_RATE",0.2,1.5)
            # vars(args)["DELTA_CLS_LOSS"] = trial.suggest_float("DELTA_CLS_LOSS",0,0.15)
            # vars(args)["WEIGHTED_SEG_LOSS"] = trial.suggest_categorical("WEIGHTED_SEG_LOSS", [True, False])
            # if vars(args)["WEIGHTED_SEG_LOSS"]:
                # vars(args)["WEIGHTED_SEG_LOSS"] = trial.suggest_int("WEIGHTED_SEG_LOSS_P",1,3)
                # vars(args)["WEIGHTED_SEG_LOSS"] = trial.suggest_int("WEIGHTED_SEG_LOSS_MAX",1,3)
            vars(args)["DYN_BALANCED_LOSS"] = trial.suggest_categorical("DYN_BALANCED_LOSS", [True, False])
            # vars(args)["GRADIENT_ADJUSTMENT"] = trial.suggest_categorical("GRADIENT_ADJUSTMENT", [True, False])
            # vars(args)["MULTISEG"] = trial.suggest_categorical("MULTISEG", [True, False])
            vars(args)["MULTIDEC"] = trial.suggest_categorical("MULTIDEC", [True, False])
            vars(args)["CLASSWEIGHTS"] = trial.suggest_categorical("CLASSWEIGHTS", [True, False])
            # vars(args)["FREQUENCY_SAMPLING"] = trail.suggest_categorical("FREQUENCY_SAMPLING", [True])
            
            # args.RESULTS_PATH = os.path.join("E:/AI/FAPS/code/Mixedsupervision/results",param,str(val))
            configuration = Config()
            configuration.merge_from_args(args)
            configuration.init_extra()
            configuration.trial = trial

            # configuration.IOU_THRESHOLD = 0.5
            end2end = End2End(cfg=configuration)
            fscore = end2end.train()
            print('********',fscore)
            return fscore
        
        SPLIT = 0
        # study = optuna.create_study(study_name=f"study_db_WS0",direction = 'maximize', storage=f"sqlite:///final_runs.db", load_if_exists=True)
        study = optuna.create_study(study_name=f"study_final_{SPLIT}_new",
                                    direction = 'maximize', 
                                    storage=f"sqlite:///final_runs.db", 
                                    load_if_exists=True,
                                    sampler=optuna.samplers.TPESampler(multivariate=True))
        study.optimize(objective, n_trials=4)
        # joblib.dump(study, study_filename)
        os.system('sbatch runner_hyper.sh')
        
        # def objective(trial):

        #     # vars(args)['DILATE'] = trail.suggest_int('DILATE',1,17)
        #     # vars(args)['EPOCHS'] = trial.suggest_categorical("EPOCHS", [1])
        #     # vars(args)["LEARNING_RATE"] = trial.suggest_categorical("LEARNING_RATE", [0.6468867589109444])
        #     # vars(args)["DELTA_CLS_LOSS"] = trial.suggest_categorical("DELTA_CLS_LOSS", [0.028373756707337477])
        #     # # vars(args)["WEIGHTED_SEG_LOSS"] = trail.suggest_categorical("WEIGHTED_SEG_LOSS", [True, False])
        #     # # if vars(args)["WEIGHTED_SEG_LOSS"]:
        #     # #     vars(args)["WEIGHTED_SEG_LOSS"] = trail.suggest_int("WEIGHTED_SEG_LOSS_P",1,3)
        #     # #     vars(args)["WEIGHTED_SEG_LOSS"] = trail.suggest_int("WEIGHTED_SEG_LOSS_MAX",1,3)
        #     # vars(args)["DYN_BALANCED_LOSS"] = trial.suggest_categorical("DYN_BALANCED_LOSS", [True])
        #     # vars(args)["GRADIENT_ADJUSTMENT"] = trial.suggest_categorical("GRADIENT_ADJUSTMENT", [False])
        #     # vars(args)["MULTISEG"] = trial.suggest_categorical("MULTISEG", [True])
        #     # vars(args)["MULTIDEC"] = trial.suggest_categorical("MULTIDEC", [False])
        #     # vars(args)["CLASSWEIGHTS"] = trial.suggest_categorical("CLASSWEIGHTS", [False])
        #     # vars(args)["FREQUENCY_SAMPLING"] = trail.suggest_categorical("FREQUENCY_SAMPLING", [True])
            
        #     # args.RESULTS_PATH = os.path.join("E:/AI/FAPS/code/Mixedsupervision/results",param,str(val))
        #     configuration = Config()
        #     configuration.merge_from_args(args)
        #     configuration.init_extra()
        #     configuration.trial = trial

        #     # configuration.IOU_THRESHOLD = 0.5
        #     end2end = End2End(cfg=configuration)
        #     fscore = end2end.train()
        #     print('********',fscore)
        #     return fscore
        
        

        # SPLIT = 65
        # # study_filename = 'study_12.pkl'
        # # if os.path.isfile(study_filename):
        # #     study = joblib.load(study_filename)
        # # else:
        # #     study = optuna.create_study(direction='maximize',pruner=optuna.pruners.MedianPruner(
        # #     n_startup_trials=10, n_warmup_steps=30, interval_steps=10))
        # study = optuna.create_study(study_name=f"studytest_{SPLIT}_db",direction = 'maximize', storage=f"sqlite:///studytest_{SPLIT}.db", load_if_exists=True)
        # study.optimize(objective, n_trials=1)
        # # joblib.dump(study, study_filename)
        # # os.system('sbatch runner_hyper.sh')


    else:
       
        """training model """
        
        start = time.time()
        configuration = Config()
        configuration.merge_from_args(args)
        configuration.init_extra()

        end2end = End2End(cfg=configuration)
        end2end.train()
        print(f"\n\ntotal time = {time.time() - start}")

        """IOU threshold selection"""

        configuration = Config()
        configuration.merge_from_args(args)
        configuration.init_extra()

        # THRESHOLDS = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
        THRESHOLDS = [0.5]


        for thresh in THRESHOLDS:

            configuration.IOU_THRESHOLD = thresh
            # configuration.RUN_NAME = f'{configuration.RUN_NAME }/THRESHOLD/{thresh}'
            end2end = End2End(cfg=configuration)
            device = end2end._get_device()
            model = end2end._get_model().to(device)
            end2end._set_results_path()
            optimizer = end2end._get_optimizer(model)
            loss_seg, loss_dec = end2end._get_loss(True), end2end._get_loss(False)    
            end2end.set_dec_gradient_multiplier(model, 0.0)
            end2end.threshold_selection(model, device, True, False, True, 'TEST',"TEST")
            end2end.threshold_selection(model, device, True, False, True, 'VAL',"VAL")
            end2end.threshold_selection(model, device, False, False, True, 'TRAIN',"TRAIN")

    
    




