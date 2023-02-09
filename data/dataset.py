import cv2
import numpy as np
import torch
from scipy.ndimage.morphology import distance_transform_edt
from scipy.signal import convolve2d
from config import Config



class Dataset(torch.utils.data.Dataset):
    def __init__(self, path: str, cfg: Config, kind: str):
        super(Dataset, self).__init__()
        self.path: str = path
        self.cfg: Config = cfg
        self.kind: str = kind
        self.image_size: (int, int) = (self.cfg.INPUT_WIDTH, self.cfg.INPUT_HEIGHT)
        self.grayscale: bool = self.cfg.INPUT_CHANNELS == 1

        self.num_negatives_per_one_positive: int = 1
        self.frequency_sampling: bool = self.cfg.FREQUENCY_SAMPLING and self.kind == 'TRAIN'

    def init_extra(self):
        self.counter = 0
        self.neg_imgs_permutation = np.random.permutation(self.num_neg)

        self.neg_retrieval_freq = np.zeros(shape=self.num_neg)

    def __getitem__(self, index) -> (torch.Tensor, torch.Tensor, torch.Tensor, bool, str):

        if self.counter >= self.len:
            self.counter = 0
            if self.frequency_sampling:
                sample_probability = 1 - (self.neg_retrieval_freq / np.max(self.neg_retrieval_freq))
                sample_probability = sample_probability - np.median(sample_probability) + 1
                sample_probability = sample_probability ** (np.log(len(sample_probability)) * 4)
                sample_probability = sample_probability / np.sum(sample_probability)

                # use replace=False for to get only unique values
                self.neg_imgs_permutation = np.random.choice(range(self.num_neg),
                                                             size=self.num_negatives_per_one_positive * self.num_pos,
                                                             p=sample_probability,
                                                             replace=False)
            else:
                self.neg_imgs_permutation = np.random.permutation(self.num_neg)

        

        if self.kind == 'TRAIN':
            if index >= self.num_pos:
                ix = index % self.num_pos
                ix = self.neg_imgs_permutation[ix]
                item = self.neg_samples[ix]
                self.neg_retrieval_freq[ix] = self.neg_retrieval_freq[ix] + 1

            else:
                ix = index
                item = self.pos_samples[ix]
        else:
            if index < self.num_neg:
                ix = index
                item = self.neg_samples[ix]
            else:
                ix = index - self.num_neg
                item = self.pos_samples[ix]

        if self.cfg.DATASET == 'PA_M':
            image, seg_mask, seg_loss_mask, is_segmented, image_path, seg_mask_path, sample_name,y_val = item
        else:
            image, seg_mask, seg_loss_mask, is_segmented, image_path, seg_mask_path, sample_name = item
            y_val = None

        if self.cfg.ON_DEMAND_READ:  # STEEL only so far
            seg_loss_mask = 1
            if image_path == -1 or seg_mask_path == -1:
                raise Exception('For ON_DEMAND_READ image and seg_mask paths must be set in read_contents')
            img = self.read_img_resize(image_path, self.grayscale, self.image_size)
            if seg_mask_path is None:  # good sample
                seg_mask = np.zeros((self.cfg.NUM_CLASS,*self.image_size[::-1]))
            elif isinstance(seg_mask_path, list):
                seg_mask = self.rle_to_mask(seg_mask_path, self.image_size)
            else:
                seg_mask, _ = self.read_label_resize(seg_mask_path, self.image_size,self.cfg.DILATE,y_val)
                # cv2.imshow('show',seg_mask)
                # cv2.waitKey(0)
            if self.cfg.WEIGHTED_SEG_LOSS:
                if np.max(seg_mask) == np.min(seg_mask):  # good sample
                    seg_loss_mask = np.ones_like(seg_mask)
                else:
                    seg_loss_mask = self.distance_transform(seg_mask, self.cfg.WEIGHTED_SEG_LOSS_MAX, self.cfg.WEIGHTED_SEG_LOSS_P,y_val)

            if np.random.rand() > 0.5:
                img = img[:, ::-1, :]
                seg_mask = np.fliplr(seg_mask)
                if self.cfg.WEIGHTED_SEG_LOSS:
                    seg_loss_mask = np.fliplr(seg_loss_mask)

            image = self.to_tensor(img)
            seg_mask = self.to_tensor(self.downsize(seg_mask) ,True)
            
            # if self.cfg.DATASET == 'PA_M':
            #     y_val = torch.from_numpy(y_val)
            if self.cfg.WEIGHTED_SEG_LOSS:
                seg_loss_mask = self.to_tensor(self.downsize(seg_loss_mask, True))

        self.counter = self.counter + 1

        return image, seg_mask, seg_loss_mask, is_segmented, sample_name,y_val

    def __len__(self):
        return self.len

    def read_contents(self):
        pass

    def read_img_resize(self, path, grayscale, resize_dim) -> np.ndarray:

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
        if resize_dim is not None:
            img = cv2.resize(img, dsize=resize_dim)
        img = np.array(img, dtype=np.float32) / 255.0
        return img

    def read_label_resize(self, path, resize_dim, dilate=None,y_val=np.zeros(12)) -> (np.ndarray, bool):

        if self.cfg.DATASET == 'PA_M':
            if resize_dim is not None:
                mask = np.zeros((self.cfg.NUM_CLASS,*resize_dim[::-1]))
            else:
                mask = np.zeros(self.cfg.NUM_CLASS,*self.image_size)[0]
            for i in np.where(np.array(y_val) == 1)[0]: 
                j = i
                if i >= 3:
                    j = i+1
                path = f'{path[:-4]}_{i}.jpg'
                lbl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                lbl[lbl>10] = 255
                if dilate is not None and dilate > 1:
                    lbl = cv2.dilate(lbl, np.ones((dilate, dilate)))
                if resize_dim is not None:
                    lbl = cv2.resize(lbl, dsize=resize_dim)
                
                mask[i] = np.array((lbl / 255.0), dtype=np.float32)
            
            return mask, np.max(mask) > 0
        
        else :
            path = path[:-4]+'.jpg'
            lbl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            lbl[lbl>10] = 255
            if dilate is not None and dilate > 1:
                lbl = cv2.dilate(lbl, np.ones((dilate, dilate)))
            if resize_dim is not None:
                lbl = cv2.resize(lbl, dsize=resize_dim)
            return np.array((lbl / 255.0), dtype=np.float32), np.max(lbl) > 0

    def to_tensor(self, x, is_Seg = False) -> torch.Tensor:
        if x.dtype != np.float32:
            x = (x / 255.0).astype(np.float32)

        if len(x.shape) == 3 and not is_Seg:
            x = np.transpose(x, axes=( 2, 0, 1))
        elif not len(x.shape) == 3:
            x = np.expand_dims(x, axis=0)

        x = torch.from_numpy(x.copy())
        return x

    def distance_transform(self, mask: np.ndarray, max_val: float, p: float,y_val = np.zeros(12)) -> np.ndarray:
        
        
        h, w = mask.shape[:2]
        if self.cfg.DATASET == 'PA_M':
            h, w = mask.shape[1:3]      
            dst_trf = np.zeros_like(mask)

            for i in np.where(np.array(y_val) == 1)[0]:
            
                num_labels, labels = cv2.connectedComponents((mask[i] * 255.0).astype(np.uint8), connectivity=8)
                for idx in range(1, num_labels):
                    mask_roi= np.zeros((h, w))
                    k = labels == idx
                    mask_roi[k] = 255
                    dst_trf_roi = distance_transform_edt(mask_roi)
                    if dst_trf_roi.max() > 0:
                        dst_trf_roi = (dst_trf_roi / dst_trf_roi.max())
                        dst_trf_roi = (dst_trf_roi ** p) * max_val
                    dst_trf[i] += dst_trf_roi

                dst_trf[i][mask == 0] = 1

        else :

            h, w = mask.shape[:2]      
            dst_trf = np.zeros((h,w))
            
            num_labels, labels = cv2.connectedComponents((mask * 255.0).astype(np.uint8), connectivity=8)
            for idx in range(1, num_labels):
                mask_roi= np.zeros((h, w))
                k = labels == idx
                mask_roi[k] = 255
                dst_trf_roi = distance_transform_edt(mask_roi)
                if dst_trf_roi.max() > 0:
                    dst_trf_roi = (dst_trf_roi / dst_trf_roi.max())
                    dst_trf_roi = (dst_trf_roi ** p) * max_val
                dst_trf += dst_trf_roi

                dst_trf[mask == 0] = 1
        
        
        return np.array(dst_trf, dtype=np.float32)

    def downsize(self, image: np.ndarray, downsize_factor: int = 8) -> np.ndarray:
        img_t = torch.from_numpy(np.expand_dims(image, 0 if len(image.shape) == 3 else (0, 1)).astype(np.float32))
        img_t = torch.nn.ReflectionPad2d(padding=(downsize_factor))(img_t)
        image_np = torch.nn.AvgPool2d(kernel_size=2 * downsize_factor + 1, stride=downsize_factor)(img_t).detach().numpy()
        return image_np[0] if len(image.shape) == 3 else image_np[0, 0]

    def rle_to_mask(self, rle, image_size):
        if len(rle) % 2 != 0:
            raise Exception('Suspicious')

        w, h = image_size
        mask_label = np.zeros(w * h, dtype=np.float32)

        positions = rle[0::2]
        length = rle[1::2]
        for pos, le in zip(positions, length):
            mask_label[pos - 1:pos + le - 1] = 1
        mask = np.reshape(mask_label, (h, w), order='F').astype(np.uint8)
        return mask
