import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from models import SegDecNet
import numpy as np
import os
from torch import nn as nn
import torch
import utils
import pandas as pd
from data.dataset_catalog import get_dataset
import random
import cv2
from config import Config
from torch.utils.tensorboard import SummaryWriter
from functional import iou_score,get_stats
from sklearn.metrics import f1_score
from math import isnan
import optuna

LVL_ERROR = 10
LVL_INFO = 5
LVL_DEBUG = 1

LOG = 1  # Will log all mesages with lvl greater than this
SAVE_LOG = True

WRITE_TENSORBOARD = False

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class End2End:
    def __init__(self, cfg: Config):
        self.cfg: Config = cfg
        self.storage_path: str = os.path.join(self.cfg.RESULTS_PATH, self.cfg.DATASET)
        if self.cfg.REPRODUCIBLE_RUN:
            # self._log("Reproducible run, fixing all seeds to:1337", LVL_DEBUG)
            np.random.seed(1337)
            torch.manual_seed(1337)
            random.seed(1337)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _log(self, message, lvl=LVL_INFO):
        n_msg = f"{self.run_name} {message}"
        if lvl >= LOG:
            if True:
            # if not self.cfg.HYPERPARAM:
                print(n_msg)

    def train(self):
        self._set_results_path()
        self._create_results_dirs()
        self.print_run_params()
        if self.cfg.REPRODUCIBLE_RUN:
            self._log("Reproducible run, fixing all seeds to:1337", LVL_DEBUG)
            np.random.seed(1337)
            torch.manual_seed(1337)
            random.seed(1337)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        device = self._get_device()
        model = self._get_model().to(device)
        optimizer = self._get_optimizer(model)
        loss_seg, loss_dec = self._get_loss(True), self._get_loss(False)

        train_loader = get_dataset("TRAIN", self.cfg)
        validation_loader = get_dataset("VAL", self.cfg)

        tensorboard_writer = SummaryWriter(log_dir=self.tensorboard_path) if WRITE_TENSORBOARD else None
        train_results = self._train_model(device, model, train_loader, loss_seg, loss_dec, optimizer, validation_loader, tensorboard_writer)
        
        if self.cfg.HYPERPARAM:
            # exit()
            # print(train_results)
            return train_results[1]
        
        self._save_train_results(train_results)
        self._save_model(model)

        self.threshold_selection(model, device, self.cfg.SAVE_IMAGES, False, False,dataset='VAL')

        self._save_params()

    def eval(self, model, device, save_images, plot_seg, reload_final,prefix=''):
        # print(model.volume_lr_multiplier_layer)
        self.reload_model(model, reload_final)
        test_loader = get_dataset("TEST", self.cfg)
        self.eval_model(device, model, test_loader, save_folder=self.outputs_path, save_images=save_images, is_validation=False, plot_seg=plot_seg, prefix=prefix)

    def threshold_selection(self, model, device, save_images, plot_seg, reload_final,prefix = '',dataset='VAL',reduction = 'mean'):
        # print(model.volume_lr_multiplier_layer)
        self.reload_model(model, reload_final)
        test_loader = get_dataset(dataset, self.cfg)
        # print(test_loader)
        self.eval_model(device, model, test_loader, save_folder=self.outputs_path, save_images=save_images, is_validation=False, plot_seg=plot_seg, prefix=prefix,reduction=reduction)
    
    def training_iteration(self, data, device, model, criterion_seg, criterion_dec, optimizer, weight_loss_seg, weight_loss_dec,
                           tensorboard_writer, iter_index,isVal = False):
        images, seg_masks, seg_loss_masks, is_segmented, _ , y_val = data

        batch_size = self.cfg.BATCH_SIZE
        memory_fit = self.cfg.MEMORY_FIT  # Not supported yet for >1
        # class_weights =torch.FloatTensor([0.09, 0.02 , 0.08, 0.08, 0.07, 0.05, 0.15, 0.3 , 0.18, 1.  , 0.03])
        # class_weights = torch.ones((1,self.cfg.SEG_OUTSIZE,self.cfg.INPUT_HEIGHT, self.cfg.INPUT_WIDTH)).to(self._get_device())
        class_weights = torch.ones((1,self.cfg.SEG_OUTSIZE,self.cfg.INPUT_HEIGHT//self.cfg.DOWN_FACTOR, self.cfg.INPUT_WIDTH//self.cfg.DOWN_FACTOR)).to(self._get_device())
        # class_weights = class_weights.view(1, self.cfg.SEG_OUTSIZE, 1,1).expand(-1, -1,  self.cfg.INPUT_HEIGHT//self.cfg.DOWN_FACTOR, self.cfg.INPUT_WIDTH//self.cfg.DOWN_FACTOR).to(self._get_device())

        num_subiters = int(batch_size / memory_fit)
        total_loss = 0
        total_correct = 0

        if not isVal:
            optimizer.zero_grad()

        total_loss_seg = 0
        total_loss_dec = 0
        ious = [0]

        for sub_iter in range(num_subiters):
            # print(images.shape)
            images_ = images[sub_iter * memory_fit:(sub_iter + 1) * memory_fit, :, :, :].to(device)
            seg_masks_ = seg_masks[sub_iter * memory_fit:(sub_iter + 1) * memory_fit, :, :, :].to(device)
            if self.cfg.WEIGHTED_SEG_LOSS:
                seg_loss_masks_ = seg_loss_masks[sub_iter * memory_fit:(sub_iter + 1) * memory_fit, :, :, :].to(device)
            is_pos_ = seg_masks_.max().reshape((memory_fit, 1)).to(device)
            if self.cfg.DATASET == 'PA_M':
                y_val_ = y_val.reshape((memory_fit, self.cfg.DEC_OUTSIZE)).float().to(device)

            if tensorboard_writer is not None and iter_index % 100 == 0:
                tensorboard_writer.add_image(f"{iter_index}/image", images_[0, :, :, :])
                tensorboard_writer.add_image(f"{iter_index}/seg_mask", seg_masks[0, :, :, :])
                tensorboard_writer.add_image(f"{iter_index}/seg_loss_mask", seg_loss_masks_[0, :, :, :])

            decision, output_seg_mask = model(images_)

            if is_segmented[sub_iter]:

                loss_seg = criterion_seg(output_seg_mask, seg_masks_) * class_weights
                
                if self.cfg.WEIGHTED_SEG_LOSS:
                    loss_seg = torch.mean(loss_seg * seg_loss_masks_)
                else:
                    loss_seg = torch.mean(loss_seg)
                    
                if self.cfg.DATASET == 'PA_M':
                    loss_dec = torch.mean(criterion_dec(decision, y_val_))
                else:
                    loss_dec = torch.mean(criterion_dec(decision, is_pos_))

                total_loss_seg += loss_seg.item()
                total_loss_dec += loss_dec.item()
                # total_correct += np.sum((decision.numpy() > 0.0))
                total_correct += 1
                loss = weight_loss_seg * loss_seg + weight_loss_dec * loss_dec
            else:
                if self.cfg.DATASET == 'PA_M':
                    loss_dec = torch.mean(criterion_dec(decision, y_val_))
                else:
                    loss_dec = torch.mean(criterion_dec(decision, is_pos_))
                total_loss_dec += loss_dec.item()

                # total_correct += (decision > 0.0).item() == is_pos_.item()
                total_correct += 1
                loss = weight_loss_dec * loss_dec
            total_loss += loss.item()

            # iou_metric = utils.iou_pytorch(output_seg_mask,seg_masks_,self.cfg.IOU_THRESHOLD,reduction='mean').item()

            # tp, fp, fn, tn = get_stats(output_seg_mask,seg_masks_, mode='multilabel', threshold=self.cfg.IOU_THRESHOLD)
            # iou_metric = iou_score(tp, fp, fn, tn, reduction="micro")
            if not isVal:
                ious.append(1)

                loss.backward()

        # Backward and optimize
        if not isVal:
            optimizer.step()
            optimizer.zero_grad()

        if isVal:
            return total_loss_seg, total_loss_dec, total_loss, np.mean(ious),decision
        else:
            return total_loss_seg, total_loss_dec, total_loss, np.mean(ious)

    def _train_model(self, device, model, train_loader, criterion_seg, criterion_dec, optimizer, validation_set, tensorboard_writer,trail=None):
        losses = []
        validation_data = []
        max_validation = -1
        validation_step = self.cfg.VALIDATION_N_EPOCHS
        # validation_loss_list = [np.inf,np.inf,np.inf]
        es_seg = EarlyStopper(patience=5,min_delta=0)
        es_dec = EarlyStopper(patience=5,min_delta=0)

        num_epochs = self.cfg.EPOCHS
        samples_per_epoch = len(train_loader) * self.cfg.BATCH_SIZE
        val_samples_per_epoch = len(validation_set) * self.cfg.BATCH_SIZE
        
        self.set_dec_gradient_multiplier(model, 0.0)

        # self.reload_model(model)

        for epoch in range(num_epochs):
            if epoch % 5 == 0:
                self._save_model(model, f"ep_{epoch:02}.pth")

            model.train()

            weight_loss_seg, weight_loss_dec = self.get_loss_weights(epoch)
            dec_gradient_multiplier = self.get_dec_gradient_multiplier(epoch)
            self.set_dec_gradient_multiplier(model, dec_gradient_multiplier)

            epoch_loss_seg, epoch_loss_dec, epoch_loss = 0, 0, 0
            epoch_correct = 0
            epoch_ious = [0]
            from timeit import default_timer as timer

            time_acc = 0
            start = timer()
            for iter_index, (data) in enumerate(train_loader):
                start_1 = timer()
                curr_loss_seg, curr_loss_dec, curr_loss, iou = self.training_iteration(data, device, model,
                                                                                           criterion_seg,
                                                                                           criterion_dec,
                                                                                           optimizer, weight_loss_seg,
                                                                                           weight_loss_dec,
                                                                                           tensorboard_writer, (epoch * samples_per_epoch + iter_index))

                end_1 = timer()
                time_acc = time_acc + (end_1 - start_1)

                epoch_loss_seg += curr_loss_seg
                epoch_loss_dec += curr_loss_dec
                epoch_loss += curr_loss

                epoch_ious.append(iou)

                if isnan(epoch_loss):
                    if self.cfg.HYPERPARAM:
                        return -1,0
                    else:
                        self._log("loss in Nan")
                        return None,-1

            end = timer()

            epoch_loss_seg = epoch_loss_seg / samples_per_epoch
            epoch_loss_dec = epoch_loss_dec / samples_per_epoch
            epoch_loss = epoch_loss / samples_per_epoch
            losses.append((epoch_loss_seg, epoch_loss_dec, epoch_loss,np.mean(epoch_ious) ,epoch))

            self._log(
                f"Epoch {epoch + 1}/{num_epochs} ==> avg_loss_seg={epoch_loss_seg:.5f}, avg_loss_dec={epoch_loss_dec:.5f}, avg_loss={epoch_loss:.5f}, correct={epoch_correct}/{samples_per_epoch}, in {end - start:.2f}s/epoch (fwd/bck in {time_acc:.2f}s/epoch)")

        
            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar("Loss/Train/segmentation", epoch_loss_seg, epoch)
                tensorboard_writer.add_scalar("Loss/Train/classification", epoch_loss_dec, epoch)
                tensorboard_writer.add_scalar("Loss/Train/joined", epoch_loss, epoch)
                tensorboard_writer.add_scalar("Accuracy/Train/", epoch_correct / samples_per_epoch, epoch)

        
            
            if True:

                if self.cfg.VALIDATE and (epoch % 5 == 0 or epoch == num_epochs - 1)  :

                    model.eval()

                    epoch_loss_seg, epoch_loss_dec, epoch_loss = 0, 0, 0
                    epoch_correct = 0
                    epoch_ious = [0]
                    val_true = []
                    val_pred = []

                    for data in validation_set:
                        val_true.append(data[-1].numpy().reshape(-1))
                        curr_loss_seg, curr_loss_dec, curr_loss, iou,decision = self.training_iteration(data, device, model,
                                                                                                criterion_seg,
                                                                                                criterion_dec,
                                                                                                optimizer, weight_loss_seg,
                                                                                                weight_loss_dec,
                                                                                                tensorboard_writer, (epoch * samples_per_epoch + iter_index),True)
                        out = nn.Sigmoid()(decision)
                        val_pred.append(out.detach().cpu().numpy().reshape(-1))
                        # print(out)
                        # self.eval_model(device, model, validation_set, None, False, True, False)
                        

                        

                        epoch_loss_seg += curr_loss_seg
                        epoch_loss_dec += curr_loss_dec
                        epoch_loss += curr_loss

                        epoch_ious.append(iou)

                        if isnan(epoch_loss):
                            if self.cfg.HYPERPARAM:
                                return -1,0
                            else:
                                self._log("val loss in Nan")
                                return None,-1

                    v_prediction = np.array(val_pred)
                    
                    v_ground_truth = np.array(val_true)
                    # print(v_prediction[:2,:])
                    # print(v_ground_truth[:2,:])
                    # self.eval_model(device, model, validation_set, None, False, True, False)

                    fscore = []
                    for thresh in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]:
                        # print(res[:,0])
                        fscore.append(f1_score(v_ground_truth,v_prediction>=thresh,average='macro'))
                    self._log(f's0 {fscore}')
                    fscore = np.max(fscore)
                    
                    
                    epoch_loss_seg = epoch_loss_seg / val_samples_per_epoch
                    epoch_loss_dec = epoch_loss_dec / val_samples_per_epoch
                    epoch_loss = epoch_loss / val_samples_per_epoch

                    if self.cfg.HYPERPARAM:
                        
                        # self.cfg.trial.report(fscore, epoch)

                        # if self.cfg.trial.should_prune():
                        #     raise optuna.TrialPruned()

                        if es_seg.early_stop(epoch_loss_seg) and es_dec.early_stop(epoch_loss_dec):
                            print(f'Early stop at {epoch}')
                            return 0,fscore

                        continue
                    
                    validation_data.append((epoch_loss_seg, epoch_loss_dec, epoch_loss,np.mean(epoch_ious) ,epoch, fscore))
                    self._log(f'validation fscore = {fscore},thresh = {np.argmax(fscore)}, \n"Epoch {epoch + 1}/{num_epochs} ==> avg_loss_seg={epoch_loss_seg:.5f}, avg_loss_dec={epoch_loss_dec:.5f}, avg_loss={epoch_loss:.5f}')
                    
                    

                    # if self.cfg.VALIDATE and (epoch % validation_step == 0 or epoch == num_epochs - 1)  :
                    #     _ , test = self.eval_model(device, model, validation_set, None, False, True, False)

            
            # if self.cfg.HYPERPARAM:

            #     if self.cfg.VALIDATE and (epoch % validation_step == 0 or epoch == num_epochs - 1)  :
            #         _ , validation_data = self.eval_model(device, model, validation_set, None, False, True, False)

                    # if es_seg.early_stop(epoch_loss_seg) and es_dec.early_stop(epoch_loss_dec):
                    #         print(f'Early stop at {epoch}')
                    #         return 0,fscore

                    # self.cfg.trial.report(validation_data, epoch)

                    # if self.cfg.trial.should_prune():
                    #     raise optuna.TrialPruned()


                    
        if self.cfg.HYPERPARAM:
            return 0,fscore
        
        return losses, validation_data

    def eval_model(self, device, model, eval_loader, save_folder, save_images, is_validation, plot_seg, prefix='',reduction = 'mean'):
        model.eval()

        dsize = self.cfg.INPUT_WIDTH, self.cfg.INPUT_HEIGHT

        v_prediction = []
        v_ground_truth = []

        res = []
        colors = np.array([(50, 69, 57), (170, 98, 195), (112, 168, 172), (112, 215, 214), (181, 72, 24), (22, 132, 221), (167, 228, 28), (124, 33, 204), (16, 0, 241), (72, 117, 136), (70, 113, 190), (149, 89, 6), (255, 255, 255)])
        # print('inside',eval_loader)
        for data_point in eval_loader:
            image, seg_mask, seg_loss_mask, _, sample_name,y_val = data_point
            image, seg_mask, y_val = image.to(device), seg_mask.to(device),y_val.reshape(1,self.cfg.DEC_OUTSIZE)
            prediction, pred_seg = model(image)
            pred_seg = nn.Sigmoid()(pred_seg)
            prediction = nn.Sigmoid()(prediction)

            # print('\n\n\n')
            # print(prediction)
            # exit()

            # if True:
            #     pred_seg[0][prediction.reshape(-1)<0.6] = torch.zeros((pred_seg.shape[2:])).to(device)

            # if is_pos:
            iou_metric = utils.iou_pytorch(pred_seg,seg_mask,self.cfg.IOU_THRESHOLD,reduction='none').cpu().numpy()
            iou_metric_mean = utils.iou_pytorch(pred_seg,seg_mask,self.cfg.IOU_THRESHOLD,reduction='mean').cpu().numpy()
            # tp, fp, fn, tn = get_stats(pred_seg,seg_mask.int(), mode='multilabel', threshold=self.cfg.IOU_THRESHOLD,num_classes=self.cfg.NUM_CLASS)
            # # iou_metric = iou_score(tp, fp, fn, tn, reduction="micro").item()
            # iou_metric = iou_score(tp, fp, fn, tn, reduction="none").cpu().numpy()
            # iou_metric = 0

            prediction = prediction.detach().cpu().numpy()
            image = image.detach().cpu().numpy()
            pred_seg =pred_seg.detach().cpu().numpy()
            # pred_seg_img = torch.argmax(pred_seg, dim=1)
            # pred_seg_img = pred_seg_img.detach().cpu().numpy()
           
            # seg_mask_img = torch.argmax(seg_mask, dim=1)
            
            seg_mask = seg_mask.detach().cpu().numpy()
            seg_loss_mask = seg_loss_mask.detach().cpu().numpy()
            
            # seg_mask_img = seg_mask_img.detach().cpu().numpy()

            y_val = y_val.detach().cpu().numpy().reshape(-1)
            if self.cfg.DATASET != 'PA_M':
                y_val = (seg_mask.max() > 0).reshape(-1).astype(np.int32)
            sample_name = sample_name.item()

            # predictions.append(prediction)
            # ground_truths.append(y_val)

            res.append((prediction.reshape(-1),y_val, sample_name,iou_metric_mean,iou_metric.reshape(-1)))
            # res.append((prediction.reshape(-1),y_val, sample_name,iou_metric))

            
            if save_images:
                image = cv2.resize(np.transpose(image[0, :, :, :], (1, 2, 0)), dsize)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # np.transpose(seg_mask_img, (1, 2, 0))
                seg_mask = np.transpose(seg_mask[0], (1, 2, 0))
                ground_label = np.zeros((*seg_mask.shape[:-1],3))#np.transpose(seg_mask[0], (1, 2, 0))*np.array([[0,0,0]])
                
                pred_seg = np.transpose(pred_seg[0], (1, 2, 0))
                pred_seg = np.where(pred_seg < self.cfg.IOU_THRESHOLD ,0,pred_seg)
                pred_label = np.zeros((*seg_mask.shape[:-1],3))

                for i in range(ground_label.shape[0]):
                    for j in range(ground_label.shape[1]):
                        if seg_mask[i,j].max() != 0:
                            label = seg_mask[i,j].argmax()
                            ground_label[i,j] = colors[label]
                        if pred_seg[i,j].max() != 0:
                            label = pred_seg[i,j].argmax()
                            pred_label[i,j] = colors[label]

                    
                ground_label = cv2.resize(ground_label, seg_mask.shape[::-1][1:],interpolation = cv2.INTER_NEAREST).astype(np.uint8)
                # ground_label = cv2.cvtColor(ground_label, cv2.COLOR_RGB2BGR)  
                pred_label = cv2.resize(pred_label,seg_mask.shape[::-1][1:],interpolation = cv2.INTER_NEAREST).astype(np.uint8) 

                utils.plot_sample(sample_name, image, pred_label, ground_label, save_folder, decision=prediction, plot_seg=plot_seg,threshold=self.cfg.IOU_THRESHOLD)

            
                
            v_prediction.append(prediction.reshape(-1))
            v_ground_truth.append(y_val)


        
        v_prediction = np.array(v_prediction)
        # print('\n\n\n')
        # print(v_prediction[:2,:])
        # print(v_ground_truth[:2,:])
        # exit()
        v_ground_truth = np.array(v_ground_truth)
        # print(v_ground_truth[:2,:])
        # exit()
        res = np.array(res)
        iou_m = np.mean(res[:,3])
        fscore = []
        for thresh in [0.2,0.3,0.4,0.5,0.6,0.7,0.8]:
            # print(res[:,0])
            fscore.append(f1_score(v_ground_truth,v_prediction>=thresh,average='macro'))
        # fscore = np.max(fscore)
        self._log(f's1 validation fscore = {fscore},thresh = {np.argmax(fscore)}')
        fscore = np.max(fscore)
            

        
            # return iou_m, fscore
            
        
        utils.evaluate_metrics(res, self.run_path, self.run_name,self.cfg.IOU_THRESHOLD, prefix)

    def get_dec_gradient_multiplier(self,epoch):
        if self.cfg.GRADIENT_ADJUSTMENT:
            grad_m = 0
        else:
            grad_m = 1

        if self.cfg.GRADIENT_ADJUSTMENT:
            grad_m = 0 if epoch<70 else 1

        self._log(f"Returning dec_gradient_multiplier {grad_m}", LVL_DEBUG)
        return grad_m

    def set_dec_gradient_multiplier(self, model, multiplier):
        model.set_gradient_multipliers(multiplier)

    def get_loss_weights(self, epoch):
        total_epochs = float(self.cfg.EPOCHS)
        # total_epochs = 70

        if self.cfg.DYN_BALANCED_LOSS:
            seg_loss_weight = 1 - (epoch / total_epochs)
            dec_loss_weight = self.cfg.DELTA_CLS_LOSS * (epoch / total_epochs)
        else:
            seg_loss_weight = 1
            dec_loss_weight = self.cfg.DELTA_CLS_LOSS
        
        self._log(f"Returning seg_loss_weight {seg_loss_weight} and dec_loss_weight {dec_loss_weight}", LVL_DEBUG)
        return seg_loss_weight, dec_loss_weight

    def reload_model(self, model, load_final=False):
        self.cfg.USE_BEST_MODEL = False
        if self.cfg.USE_BEST_MODEL:
            path = os.path.join(self.model_path, "best_state_dict.pth")
            model.load_state_dict(torch.load(path))
            self._log(f"Loading model state from {path}")
        elif load_final:
            path = os.path.join(self.model_path, "final_state_dict.pth")
            model.load_state_dict(torch.load(path))
            self._log(f"Loading model state from {path}")
        else:
            self._log("Keeping same model state")

    def _save_params(self):
        params = self.cfg.get_as_dict()
        import json

        with open(os.path.join(self.run_path,'run_params.json'), 'w') as fp:
            json.dump(params, fp)

    def _save_train_results(self, results):
        losses, validation_data = results
        ls, ld, l, iou,le  = map(list, zip(*losses))
        if self.cfg.VALIDATE:
            v_ls, v_ld, v_l, v_iou,v_e,v_fscore = map(list, zip(*validation_data))
        # plt.plot(le, l, label="Loss", color="red")
        plt.figure(figsize=(15,15))
        plt.plot(le, ls, label="Loss seg",color = 'blue')
        plt.plot(le, ld, label="Loss dec",color = 'red')
        plt.plot(le, l, label="Loss total",color = 'yellow')

        if self.cfg.VALIDATE:
            plt.plot(v_e,v_ld,label="val loss dec", color="orange")
            plt.plot(v_e,v_ls,label="val loss seg", color="brown")
            plt.plot(v_e,v_l,label="val loss ", color="brown")
        plt.ylim((0, 1))
        plt.xlabel("Epochs")
        plt.legend()
        plt.savefig(os.path.join(self.run_path, "loss_IOU"), dpi=200)

        # plt.plot(le, ld, label="Loss dec")
        plt.figure(figsize=(15,15))

        if self.cfg.VALIDATE:
            
            plt.plot(v_e,v_fscore,label="val fscore", color="violet")

        plt.ylim((0, 1))
        plt.xlabel("Epochs")
        plt.legend()
        plt.savefig(os.path.join(self.run_path, "metric"), dpi=200)


        df_loss = pd.DataFrame(data={"loss_seg": ls, "loss_dec": ld, "loss": l, "IOU" :iou, "epoch": le})
        df_loss.to_csv(os.path.join(self.run_path, "losses.csv"), index=False)

        if self.cfg.VALIDATE:
            # v, vf, ve = map(list, zip(*validation_data))
            df_loss = pd.DataFrame(data={"loss_seg": ls, "loss_dec": ld, "loss": l, "epoch": le})
            df_loss.to_csv(os.path.join(self.run_path, "losses.csv"), index=False)

    def _save_model(self, model, name="final_state_dict.pth"):
        output_name = os.path.join(self.model_path, name)
        self._log(f"Saving current model state to {output_name}")
        if os.path.exists(output_name):
            os.remove(output_name)

        torch.save(model.state_dict(), output_name)

    def _get_optimizer(self, model):
        return torch.optim.SGD(model.parameters(), self.cfg.LEARNING_RATE)

    def _get_loss(self, is_seg):

        if is_seg:
            # weights =torch.from_numpy(np.ones((1,self.cfg.SEG_OUTSIZE,self.cfg.INPUT_HEIGHT, self.cfg.INPUT_WIDTH))) 
            weights =torch.from_numpy(np.ones((1,self.cfg.SEG_OUTSIZE,self.cfg.INPUT_HEIGHT//self.cfg.DOWN_FACTOR, self.cfg.INPUT_WIDTH//self.cfg.DOWN_FACTOR))) * 7  if self.cfg.CLASSWEIGHTS else None

            return nn.BCEWithLogitsLoss(pos_weight=weights,reduction='none').to(self._get_device())
        weights =torch.from_numpy(np.ones(self.cfg.DEC_OUTSIZE)) * 7  if self.cfg.CLASSWEIGHTS else None
        return nn.BCEWithLogitsLoss(pos_weight=weights,reduction='none').to(self._get_device())

    def _get_device(self):
        return f"cuda:{self.cfg.GPU}"
        # return f"cpu"

    def _set_results_path(self):
        self.run_name = f"{self.cfg.RUN_NAME}_FOLD_{self.cfg.FOLD}" if self.cfg.DATASET in ["KSDD", "DAGM"] else self.cfg.RUN_NAME

        results_path = os.path.join(self.cfg.RESULTS_PATH, self.cfg.DATASET)
        self.tensorboard_path = os.path.join(results_path, "tensorboard", self.run_name)

        run_path = os.path.join(results_path, self.cfg.RUN_NAME)
        if self.cfg.DATASET in ["KSDD", "DAGM"]:
            run_path = os.path.join(run_path, f"FOLD_{self.cfg.FOLD}")

        self._log(f"Executing run with path {run_path}")

        self.run_path = run_path
        self.model_path = os.path.join(run_path, "models")
        self.outputs_path = os.path.join(run_path, "test_outputs")

    def _create_results_dirs(self):
        list(map(utils.create_folder, [self.run_path, self.model_path, self.outputs_path, ]))

    def _get_model(self):
        seg_net = SegDecNet(self._get_device(), self.cfg.INPUT_WIDTH, self.cfg.INPUT_HEIGHT, self.cfg.INPUT_CHANNELS,self.cfg.DATASET,self.cfg)
        return seg_net

    def print_run_params(self):
        for l in sorted(map(lambda e: e[0] + "::" + str(e[1]) + "\n", self.cfg.get_as_dict().items())):
            k, v = l.split("::")
            self._log(f"{k:25s} : {str(v.strip())}")
