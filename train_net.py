
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

    args = parser.parse_args()

    return args


if __name__ == '__main__':
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
            
    """training model """
    
    # start = time.time()
    # configuration = Config()
    # configuration.merge_from_args(args)
    # configuration.init_extra()

    # end2end = End2End(cfg=configuration)
    # end2end.train()
    # print(f"\n\ntotal time = {time.time() - start}")


    """hyper param selection """

    # def objective(trail):

    #     vars(args)['DILATE'] = trail.suggest_int('DILATE',1,17)
    #     # vars(args)['EPOCHS'] = trail.suggest_int('EPOCHS',50,150)
    #     vars(args)["LEARNING_RATE"] = trail.suggest_float("LEARNING_RATE",0.0001,1.5)
    #     vars(args)["DELTA_CLS_LOSS"] = trail.suggest_float("DELTA_CLS_LOSS",0,1)
    #     # vars(args)["WEIGHTED_SEG_LOSS"] = trail.suggest_categorical("WEIGHTED_SEG_LOSS", [True, False])
    #     # if vars(args)["WEIGHTED_SEG_LOSS"]:
    #     #     vars(args)["WEIGHTED_SEG_LOSS"] = trail.suggest_int("WEIGHTED_SEG_LOSS_P",0,1)
    #     #     vars(args)["WEIGHTED_SEG_LOSS"] = trail.suggest_int("WEIGHTED_SEG_LOSS_MAX",0,1)
        
    #     vars(args)["DYN_BALANCED_LOSS"] = trail.suggest_categorical("DYN_BALANCED_LOSS", [True, False])
    #     vars(args)["GRADIENT_ADJUSTMENT"] = trail.suggest_categorical("GRADIENT_ADJUSTMENT", [True, False])
    #     vars(args)["FREQUENCY_SAMPLING"] = trail.suggest_categorical("FREQUENCY_SAMPLING", [True, False])
        
    #     # args.RESULTS_PATH = os.path.join("E:/AI/FAPS/code/Mixedsupervision/results",param,str(val))
    #     configuration = Config()
    #     configuration.merge_from_args(args)
    #     configuration.init_extra()

    #     # configuration.IOU_THRESHOLD = 0.5
    #     end2end = End2End(cfg=configuration)
    #     loss_seg, loss_dec, total_loss, iou, _ = end2end.train(trail)
    #     return iou
    

    # study = optuna.create_study(direction='maximize')
    # joblib.dump(study, "studybefore.pkl")
    # study.optimize(objective, n_trials=20)
    # joblib.dump(study, "study.pkl")

   


    """hyper param selection """
    # RUN_NAME = args.RUN_NAME
    # params = ["WEIGHTED_SEG_LOSS_P"]
    # param_vals = {  
    #                 # "WEIGHTED_SEG_LOSS":[False],
    #                 # "DELTA_CLS_LOSS":[0.5,1],
    #                 "DYN_BALANCED_LOSS":[False],
    #                 # "GRADIENT_ADJUSTMENT":[False]
    #                 # "WEIGHTED_SEG_LOSS_MAX":[2,3,4]
    #              }
    # for param in param_vals.keys():
    #     for val in param_vals[param]:
            
    #         vars(args)[param] = val
    #         # if param == "WEIGHTED_SEG_LOSS_P":
    #             # vars(args)["WEIGHTED_SEG_LOSS_MAX"] = val+1
    #         print("{}/{}/{}".format(RUN_NAME,param,val))
    #         vars(args)['RUN_NAME'] = "{}/{}/{}".format(args.RUN_NAME,param,val)
    #         # args.RESULTS_PATH = os.path.join("E:/AI/FAPS/code/Mixedsupervision/results",param,str(val))
    #         configuration = Config()
    #         configuration.merge_from_args(args)
    #         configuration.init_extra()

    #         # configuration.IOU_THRESHOLD = 0.5
    #         end2end = End2End(cfg=configuration)
    #         end2end.train()

    """Weak supervision"""

        # # params = ["WEIGHTED_SEG_LOSS_P"]
    # segments = [288]
    # for seg in segments:
    #     vars(args)["RUN_NAME"] = f'WS{seg}'
    #     vars(args)["NUM_SEGMENTED"] = seg
    #     configuration = Config()
    #     configuration.merge_from_args(args)
    #     configuration.init_extra()

    #     end2end = End2End(cfg=configuration)
    #     end2end.train()

    """evaluation"""

    # configuration = Config()
    # configuration.merge_from_args(args)
    # configuration.init_extra()

    # end2end = End2End(cfg=configuration)
    # device = end2end._get_device()
    # model = end2end._get_model().to(device)
    # end2end._set_results_path()
    # optimizer = end2end._get_optimizer(model)
    # loss_seg, loss_dec = end2end._get_loss(True), end2end._get_loss(False)    
    # end2end.set_dec_gradient_multiplier(model, 0.0)
    # end2end.eval(model, device, True, False, True)

    # pd.read_csv('losses.csv')

    """IOU threshold selection"""

    configuration = Config()
    configuration.merge_from_args(args)
    configuration.init_extra()

    # THRESHOLDS = [0.1,0.2,0.3,0.4,0.5]
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
        end2end.threshold_selection(model, device, False, False, True, 'VAL_2')
        end2end.eval(model, device, False, False, True,'TEST_2')




