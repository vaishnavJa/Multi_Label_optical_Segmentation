import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc,f1_score
import pandas as pd
import os
import errno
import pickle
import cv2
import torch

NUM_CLASS = 12


def create_folder(folder, exist_ok=True):
    try:
        os.makedirs(folder)
    except OSError as e:
        if e.errno != errno.EEXIST or not exist_ok:
            raise


def calc_confusion_mat(D, Y):
    FP = (D != Y) & (Y.astype(np.bool) == False)
    FN = (D != Y) & (Y.astype(np.bool) == True)
    TN = (D == Y) & (Y.astype(np.bool) == False)
    TP = (D == Y) & (Y.astype(np.bool) == True)

    return FP, FN, TN, TP


def plot_sample(image_name, image, segmentation, label, save_dir, decision=None, blur=True, plot_seg=False, threshold = 0.5):
    plt.figure()
    plt.clf()
    plt.subplot(1,4,1)
    plt.xticks([])
    plt.yticks([])
    plt.title('Input image')
    if image.shape[0] < image.shape[1]:
        image = np.transpose(image, axes=[1, 0, 2])
    if segmentation.shape[0] < segmentation.shape[1]:
        segmentation = np.transpose(segmentation,axes=[1,0,2])
        label = np.transpose(label,axes=[1, 0, 2])
    if image.shape[2] == 1:
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(image)

    # colors = np.array([(50, 69, 57), (170, 98, 195), (222, 222, 55), (112, 168, 172), (112, 215, 214), (181, 72, 24), (22, 132, 221), (167, 228, 28), (124, 33, 204), (16, 0, 241), (72, 117, 136), (70, 113, 190), (149, 89, 6), (255, 255, 255)])
    plt.subplot(1, 4, 2)
    plt.xticks([])
    plt.yticks([])
    plt.title('Groundtruth')
    plt.imshow(label)

    plt.subplot(1, 4, 3)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(segmentation)
    if decision is None:
        plt.title('Output')
    else:
        plt.title(f"Output: {np.argmax(decision):.5f}")
    # display max
    # vmax_value = max(1, np.max(segmentation))
    

    # plt.subplot(1, 4, 4)
    # plt.xticks([])
    # plt.yticks([])
    # plt.title('Output scaled')
    # if blur:
    #     normed = segmentation / segmentation.max()
    #     blured = cv2.blur(normed, (32, 32))
    #     plt.imshow((blured / blured.max() * 255).astype(np.uint8), cmap="jet")
    # else:
    #     plt.imshow((segmentation / segmentation.max() * 255).astype(np.uint8), cmap="jet")

    out_prefix = '{:.3f}_'.format(np.argmax(decision)) if decision is not None else ''

    plt.savefig(f"{save_dir}/{image_name}.jpg", bbox_inches='tight', dpi=300)
    plt.close()

    if plot_seg:
        jet_seg = cv2.applyColorMap((segmentation * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(f"{save_dir}/{out_prefix}_segmentation_{image_name}.png", jet_seg)


SMOOTH = 1e-6

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor,threshold = 0.5,reduction = 'none'):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(0)  # BATCH x 1 x H x W => BATCH x H x W
    # print(outputs.shape,labels.shape)
    labels = labels.squeeze(0)  # BATCH x 1 x H x W => BATCH x H x W
    # exit()
    outputs = outputs> threshold
    labels = labels > 0
    if reduction != 'none':
        intersection = (outputs & labels).float().sum() 
        union = (outputs | labels).float().sum()
    else:
        intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
        union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    # print(intersection.item(),union.item(),iou.item())
    
    # thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return iou

# def iou_numpy(outputs: torch.Tensor, labels: torch.Tensor,threshold = 0.5):
#     # You can comment out this line if you are passing tensors of equal shape
#     # But if you are passing output from UNet or something it will most probably
#     # be with the BATCH x 1 x H x W shape
#     # outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
#     outputs = outputs> threshold
#     labels = labels > 0.5
#     intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
#     union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
    
#     iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
#     # thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
#     return iou.mean()


def evaluate_metrics(samples, results_path, run_name, threshold = 0.5, prefix=''):
    samples = np.array(samples)

    img_names = samples[:, 2]
    predictions = samples[:, 0]
    labels = samples[:, 1]
    iou_metric_mean = samples[:,3]
    iou_metric = samples[:,4]
    fscore = [f1_score(y_true,y_pred>threshold,average='weighted') for y_true,y_pred in zip(labels,predictions)] 

    # metrics = get_metrics(labels, predictions)

    df = pd.DataFrame(
        data={'prediction': [','.join(str(round(i,2)) for i in j) for j in predictions],
                'IOU':iou_metric_mean,
                'IOU_label':[','.join(str(round(i,2)) for i in j) for j in iou_metric],
                # 'decision': [','.join(str(i) for i in j) for j in predictions],
                'ground_truth': [','.join(str(i) for i in j) for j in labels],
                'fscore' : fscore,
                'img_name': img_names})
    df.to_csv(os.path.join(results_path, f'{prefix}_results{threshold*10}.csv'), index=False)

    # print(
    #     f'{run_name} EVAL IOU={np.mean(iou_metric):f}, and AP={metrics["AP"]:f}, w/ best thr={metrics["best_thr"]:f} at f-m={metrics["best_f_measure"]:.3f} and FP={sum(metrics["FP"]):d}, FN={sum(metrics["FN"]):d}')

    # with open(os.path.join(results_path, 'metrics.pkl'), 'wb') as f:
    #     pickle.dump(metrics, f)
        # f.close()

    # plt.figure(1)
    # plt.clf()
    # plt.plot(metrics['recall'], metrics['precision'])
    # plt.title('Average Precision=%.4f' % metrics['AP'])
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.savefig(f"{results_path}/precision-recall.pdf", bbox_inches='tight')

    # plt.figure(1)
    # plt.clf()
    # plt.plot(metrics['FPR'], metrics['TPR'])
    # plt.title('AUC=%.4f' % metrics['AUC'])
    # plt.xlabel('False positive rate')
    # plt.ylabel('True positive rate')
    # plt.savefig(f"{results_path}/ROC.pdf", bbox_inches='tight')


def get_metrics(labels, predictions):
    metrics = {}
    
    precision, recall, thresholds = precision_recall_curve(labels, predictions)
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['thresholds'] = thresholds
    f_measures = 2 * np.multiply(recall, precision) / (recall + precision + 1e-8)
    metrics['f_measures'] = f_measures
    ix_best = np.argmax(f_measures)
    metrics['ix_best'] = ix_best
    best_f_measure = f_measures[ix_best]
    metrics['best_f_measure'] = best_f_measure
    best_thr = thresholds[ix_best]
    metrics['best_thr'] = best_thr
    FPR, TPR, _ = roc_curve(labels, predictions)
    metrics['FPR'] = FPR
    metrics['TPR'] = TPR
    AUC = auc(FPR, TPR)
    metrics['AUC'] = AUC
    AP = auc(recall, precision)
    metrics['AP'] = AP
    decisions = predictions >= best_thr
    metrics['decisions'] = decisions
    FP, FN, TN, TP = calc_confusion_mat(decisions, labels)
    metrics['FP'] = FP
    metrics['FN'] = FN
    metrics['TN'] = TN
    metrics['TP'] = TP
    metrics['accuracy'] = (sum(TP) + sum(TN)) / (sum(TP) + sum(TN) + sum(FP) + sum(FN))
    return metrics
