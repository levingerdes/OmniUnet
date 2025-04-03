import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import metrics
import utils.img_handler
import torch.nn as nn

def evaluate(net, dataloader, device, masks_pred_size, dataset_str):
    criterion   = nn.CrossEntropyLoss()

    num_val_batches  = len(dataloader)
    pixel_acc_score  = 0 
    IoU_score        = 0
    val_dice_score   = 0
    precision_score  = 0 
    recall_score     = 0
    validation_loss_total  = 0

    net.eval()
    # iterate over the validation set
    for batch in tqdm(dataloader, total = num_val_batches, desc = 'Validation round', unit = 'batch', leave = False):
        image, mask_true = batch['image'], batch['mask']

        image     = image.to(device     = device, dtype = torch.float32)
        mask_true = mask_true.to(device = device, dtype = torch.long)
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            if dataset_str == 'T' or dataset_str == 'D':
                mask_pred, _ = net(image, image_type = dataset_str)
            else:
                mask_pred, _ = net(image)

            mask_pred   = F.interpolate(mask_pred, size = masks_pred_size, mode = 'bicubic', align_corners = False)
            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
            else:
                mask_pred  = F.one_hot(mask_pred.argmax(dim = 1), net.n_classes).permute(0, 3, 1, 2).float()
                
            # Compute the dice score, ignoring background
            pixel_acc, IoU, dice, precision, recall = metrics.calculate_multi_metrics(mask_true, mask_pred, net.n_classes, average = True)


            # Validation dice loss
            val_dice_loss    = 1 - dice
            # Criterion should be with logits, before softmax
            validation_loss  = criterion(mask_pred, mask_true) + val_dice_loss
            
            # General metrics
            pixel_acc_score  = pixel_acc_score  + pixel_acc
            IoU_score        = IoU_score        + IoU
            val_dice_score   = val_dice_score   + dice
            precision_score  = precision_score  + precision
            recall_score     = recall_score     + recall
            validation_loss_total =  validation_loss_total + validation_loss 

    net.train()
    
    result_pixel_acc_score  = pixel_acc_score  / max(num_val_batches, 1)
    result_IoU_score        = IoU_score        / max(num_val_batches, 1)
    result_val_dice_score   = val_dice_score   / max(num_val_batches, 1)
    result_precision_score  = precision_score  / max(num_val_batches, 1)
    result_recall_score     = recall_score     / max(num_val_batches, 1)
    result_validation_loss  = validation_loss_total / max(num_val_batches, 1)

    return result_pixel_acc_score, result_IoU_score, result_val_dice_score, result_precision_score, result_recall_score, result_validation_loss


def evaluate_missing_modality(net, dataloader, device, masks_pred_size, dataset_str):
    criterion   = nn.CrossEntropyLoss()

    num_val_batches  = len(dataloader)
    pixel_acc_score  = 0 
    IoU_score        = 0
    val_dice_score   = 0
    precision_score  = 0 
    recall_score     = 0
    validation_loss_total  = 0

    net.eval()
    # iterate over the validation set
    for batch in tqdm(dataloader, total = num_val_batches, desc = 'Validation round', unit = 'batch', leave = False):
        image, mask_true = batch['image'], batch['mask']
        image, missing_case = utils.img_handler.missing_modality_selector(image)

        image     = image.to(device     = device, dtype = torch.float32)
        mask_true = mask_true.to(device = device, dtype = torch.long)
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            if missing_case == 'T' or missing_case == 'D':
                mask_pred, _ = net(image, image_type = missing_case)
            else:
                mask_pred, _ = net(image)

            mask_pred   = F.interpolate(mask_pred, size = masks_pred_size, mode = 'bicubic', align_corners = False)
            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
            else:
                mask_pred  = F.one_hot(mask_pred.argmax(dim = 1), net.n_classes).permute(0, 3, 1, 2).float()
                
            # Compute the dice score, ignoring background
            pixel_acc, IoU, dice, precision, recall = metrics.calculate_multi_metrics(mask_true, mask_pred, net.n_classes, average = True)


            # Validation dice loss
            val_dice_loss    = 1 - dice
            # Criterion should be with logits, before softmax
            validation_loss  = criterion(mask_pred, mask_true) + val_dice_loss
            
            # General metrics
            pixel_acc_score  = pixel_acc_score  + pixel_acc
            IoU_score        = IoU_score        + IoU
            val_dice_score   = val_dice_score   + dice
            precision_score  = precision_score  + precision
            recall_score     = recall_score     + recall
            validation_loss_total =  validation_loss_total + validation_loss 

    net.train()
    
    result_pixel_acc_score  = pixel_acc_score  / max(num_val_batches, 1)
    result_IoU_score        = IoU_score        / max(num_val_batches, 1)
    result_val_dice_score   = val_dice_score   / max(num_val_batches, 1)
    result_precision_score  = precision_score  / max(num_val_batches, 1)
    result_recall_score     = recall_score     / max(num_val_batches, 1)
    result_validation_loss  = validation_loss_total / max(num_val_batches, 1)

    return result_pixel_acc_score, result_IoU_score, result_val_dice_score, result_precision_score, result_recall_score, result_validation_loss

