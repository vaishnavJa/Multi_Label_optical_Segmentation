

import os
import numpy as np
from data.dataset import Dataset
import glob
import pickle
from pathlib import Path,PureWindowsPath


class PAMultiDataset(Dataset):
    def __init__(self, kind, cfg):
        super(PAMultiDataset, self).__init__(cfg.DATASET_PATH, cfg, kind)
        self.read_contents()

    def read_split(self):

        with open(self.cfg.SPLIT_LOCATION, "rb") as f:
            vals = pickle.load(f)
        return vals

    def read_contents(self):
        if not self.cfg.ON_DEMAND_READ:
            raise Exception("Need to implement eager loading!")

        pos_samples, neg_samples = [], []

        # [image, segmentation_mask, segmentation_loss_mask, is_segmented, image_path, segmentation_mask_path, sample_name]

        # pos
        train_pos_path,test_pos_path,train_seg_path,test_seg_path,train_neg_path,test_neg_path,train_y_vals,test_y_vals,val_pos_path,val_seg_path,val_neg_path,val_y_vals = self.read_split()

        if self.kind == 'TRAIN':
            img_paths = train_pos_path
            seg_paths = train_seg_path
            neg_paths = train_neg_path
            y_vals = train_y_vals

        elif self.kind == 'VAL' : 
            img_paths = val_pos_path
            seg_paths = val_seg_path
            neg_paths = val_neg_path
            y_vals = val_y_vals
        
        else:
            img_paths = test_pos_path
            seg_paths = test_seg_path
            neg_paths = test_neg_path
            y_vals = test_y_vals

        # print(y_vals.shape)

        # print(len(img_paths),len(seg_paths))
        # print(y_vals)

        for i in range(len(img_paths)):
            img_p = img_paths[i]
            sample_name = os.path.basename(os.path.normpath(img_p))
            sample_name = int(sample_name[:-4])
            is_seg,seg_p = 1 if len(seg_paths[i]) > 0 else 0, seg_paths[i]
            pos_samples.append([None,None,None,is_seg,img_p,seg_p,sample_name,y_vals[i]])

        
        # print(pos_samples)
 
        for img_p in neg_paths:
            sample_name = os.path.basename(os.path.normpath(img_p))
            sample_name = int(sample_name[:-4])
            neg_samples.append([None,None,None,True,img_p,None,sample_name,np.zeros(self.cfg.DEC_OUTSIZE)])
            # print(img_p)


        self.pos_samples = pos_samples
        self.neg_samples = neg_samples

        self.num_pos = len(pos_samples)
        self.num_neg = len(neg_samples)
        self.len = 2 * len(pos_samples) if self.kind in ['TRAIN'] else len(pos_samples) + len(neg_samples)
        print(self.kind,self.num_neg,self.num_pos)
        if self.num_neg == 0:
            self.len = self.num_pos
            

        self.init_extra()
