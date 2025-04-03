import numpy as np
import torch

def get_onehot_data(gt_onehot, pred, class_num):
    
    # Perform calculation on a batch
    matrix             = np.zeros((3, class_num))
    n_gt_pixels_array  = np.zeros(class_num)

    # Calculate tp (true positives), fp (false positives), fn (false negatives) per class
    for i in range(class_num):
        # pred shape: (N, H, W)
        # gt shape:   (N, H, W), binary array where 0 denotes negative and 1 denotes positive
        class_pred = pred[:, i, :, :]
        class_gt   = gt_onehot[:, i, :, :]

        pred_flat  = class_pred.contiguous().view(-1, )  # shape: (N * H * W, )
        gt_flat    = class_gt.contiguous().view(-1, )    # shape: (N * H * W, )

        tp = torch.sum(gt_flat * pred_flat)
        fp = torch.sum(pred_flat) - tp
        fn = torch.sum(gt_flat) - tp
        
        # Counts numbers of pixels of each class in groundtruth
        n_gt_pixels_array[i] = np.sum(np.asanyarray(class_gt.cpu().detach().numpy()) == 1)

        matrix[:, i] = tp.item(), fp.item(), fn.item()

    return matrix, n_gt_pixels_array


def calculate_multi_metrics(gt, pred, class_num, average = False, eps = 1e-10, ignore_background = True):
    
    # Calculate metrics in multi-class segmentation
    matrix, n_gt_pixels_array = get_onehot_data(gt, pred, class_num)
    
    # Ignore background void for metrics
    if ignore_background == True:
        matrix             = matrix[:, 1:]
        n_gt_pixels_array  = n_gt_pixels_array[1:]
        class_num          = class_num - 1

    # 1. Pixel accuracy (PA)
    # Pixel accuracy measures how many pixels are predicted correctly
    # Ratio between the amount of adequately classified pixels and the total number of pixels of that class
    # pixel_acc = TP / (pixels of that class) = (TP + TN) / (TP + TN + FP + FN)
    pixel_acc = (matrix[0] + eps) / (n_gt_pixels_array + 1e5 * eps)
    
    # 2. Intersection over Union, or Jaccard coefficient
    # IoU = TP / (TP + FP + FN)
    IoU       = (matrix[0] + eps) / (matrix[0] + matrix[1] + matrix[2] + 1e5 * eps)

    # 3. Dice score, Dice-Sørensen coefficient, or F1-score 
    # Dice evaluates the overlap rate of prediction results and ground truth
    # It is also an "harmonious" balance between precision and recall 
    # dice = (2 * TP) / (2 * TP + FP + FN)
    dice      = (2 * matrix[0] + eps) / (2 * matrix[0] + matrix[1] + matrix[2] + 1e5 * eps)

    # 4. Precision
    # Precision describes the purity of positive detections relative to the ground truth
    # precision = TP / (TP + FP)
    precision = (matrix[0] + eps) / (matrix[0] + matrix[1] + 1e5 * eps)

    # 5. Recall
    # Completeness of positive predictions relative to the ground truth
    # Positive results divided by pixels that should have been identified as positive
    # recall = TP / (TP + FN)
    recall    = (matrix[0] + eps) / (matrix[0] + matrix[2] + 1e5 * eps)

    # Check if all tp, fp and fn of each class are all zero
    # In that case, that class is not presented in that img
    # and the class doesn't need to be considered in the average
    for i in range(class_num):
        metrics_zeros = not np.any(matrix[:, i])
        if(metrics_zeros):
            # For individual metrics 
            pixel_acc[i]   = np.nan
            IoU[i]         = np.nan
            dice[i]        = np.nan
            precision[i]   = np.nan
            recall[i]      = np.nan

    if average:
        # For average metrics, we remove nan value from list
        # to consider one value less for N
        pixel_acc_avg  = pixel_acc[~np.isnan(pixel_acc)]
        IoU_avg        = IoU[~np.isnan(IoU)]
        dice_avg       = dice[~np.isnan(dice)]
        precision_avg  = precision[~np.isnan(precision)]
        recall_avg     = recall[~np.isnan(recall)]

        # Calculating average
        pixel_acc      = np.average(pixel_acc_avg)
        IoU            = np.average(IoU_avg)
        dice           = np.average(dice_avg)
        precision      = np.average(precision_avg)
        recall         = np.average(recall_avg)

    return pixel_acc, IoU, dice, precision, recall
