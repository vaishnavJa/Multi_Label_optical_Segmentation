

import os
import numpy as np
from data.dataset import Dataset
import glob
import pickle
from pathlib import Path,PureWindowsPath



def check_has_file(img_name,seg_paths):
    for s in seg_paths:
        seg_name = os.path.basename(os.path.normpath(s))
        seg_name = int(seg_name[:-4])
        if img_name == seg_name:
            return True,s
    return False,None

class PADataset(Dataset):
    def __init__(self, kind, cfg):
        super(PADataset, self).__init__(cfg.DATASET_PATH, cfg, kind)
        self.read_contents()

    def read_split(self):
        fn = f"PA/split_{self.cfg.NUM_SEGMENTED}.pyb"
        with open(f"splits/{fn}", "rb") as f:
            vals = pickle.load(f)
        return vals

    def read_contents(self):
        if not self.cfg.ON_DEMAND_READ:
            raise Exception("Need to implement eager loading!")

        pos_samples, neg_samples = [], []

        # [image, segmentation_mask, segmentation_loss_mask, is_segmented, image_path, segmentation_mask_path, sample_name]

        # pos
        if self.cfg.NUM_SEGMENTED in [280,143,52,15] :
            train_pos_path,test_pos_path,train_seg_path,test_seg_path,train_neg_path,test_neg_path,val_pos_path,val_seg_path,val_neg_path = self.read_split()
        else:
            train_pos_path,test_pos_path,train_seg_path,test_seg_path,train_neg_path,test_neg_path = self.read_split()

        if self.kind == 'TRAIN':
            img_paths = train_pos_path
            seg_paths = train_seg_path
            neg_paths = train_neg_path

        elif self.kind == 'VAL' and self.cfg.NUM_SEGMENTED in [280,143,52,15] : 
            img_paths = val_pos_path
            seg_paths = val_seg_path
            neg_paths = val_neg_path
        
        else:
            img_paths = test_pos_path
            seg_paths = test_seg_path
            neg_paths = test_neg_path



        for i in range(len(img_paths)):
            img_p = img_paths[i]
            sample_name = os.path.basename(os.path.normpath(img_p))
            sample_name = int(sample_name[:-4])
            is_seg,seg_p = check_has_file(sample_name,seg_paths)
            
            if is_seg:
                seg_p = str(PureWindowsPath(seg_p))
                pos_samples.append([None,None,None,is_seg,str(PureWindowsPath(img_p)),seg_p,sample_name])
            else:
                pos_samples.append([None,None,None,is_seg,str(PureWindowsPath(img_p)),seg_p,sample_name])
 
        for img_p in neg_paths:
            sample_name = os.path.basename(os.path.normpath(img_p))
            sample_name = int(sample_name[:-4])
            neg_samples.append([None,None,None,True,str(PureWindowsPath(img_p)),None,sample_name])


        self.pos_samples = pos_samples
        self.neg_samples = neg_samples

        self.num_pos = len(pos_samples)
        self.num_neg = len(neg_samples)
        self.len = 2 * len(pos_samples) if self.kind in ['TRAIN'] else len(pos_samples) + len(neg_samples)

        self.init_extra()
