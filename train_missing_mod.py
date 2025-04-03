import pathlib, argparse, logging, torch, random, math, wandb, os
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.sampler
import numpy as np
import utils.evaluate_net
import utils.data_loading
import utils.img_handler
import utils.metrics
import datetime

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from torch import optim
from torch.utils.data import DataLoader, ConcatDataset, random_split
from tqdm import tqdm
from model.omniunet_model import OmniUnet
from model import unet_decoder

import warnings
warnings.filterwarnings("ignore")

THIS_FOLDER     = os.path.dirname(os.path.abspath(__file__))
LAENTIEC_CLASS  = os.path.join(THIS_FOLDER, 'classes/LAENTIEC')
RUGD_CLASS      = os.path.join(THIS_FOLDER, 'classes/RUGD')

class MultiBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset_a, dataset_b, batch_size): 
        a_len           = dataset_a.__len__()
        ab_len          = a_len + dataset_b.__len__()
        self.a_indices  = list(range(a_len))
        self.b_indices  = list(range(a_len, ab_len))
        self.batch_size = batch_size
        self.length     = math.floor(ab_len / batch_size) # Drop_last equivalent
    
    def __iter__(self):
        random.shuffle(self.a_indices)
        random.shuffle(self.b_indices)
        a_batches   = self.chunk(self.a_indices, self.batch_size)
        b_batches   = self.chunk(self.b_indices, self.batch_size)
        all_batches = list(a_batches + b_batches)
        all_batches = [batch.tolist() for batch in all_batches]
        random.shuffle(all_batches)
        return iter(all_batches)

    def __len__(self):
        return (len(self.a_indices) + len(self.b_indices)) // self.batch_size
    
    def chunk(self, indices, size):
        return torch.split(torch.tensor(indices), size)


def split_dataset(dataset, val_percent):
    # Split into train / validation partitions
    n_val   = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator = torch.Generator().manual_seed(0))

    return train_set, val_set


def train_net(net, device, epochs: int = 5,  batch_size:  int = 1,    learning_rate: float = 1e-5,
              val_percent: float = 0.1, save_checkpoint: bool = True, amp: bool = False, dataset_str: str = '', 
              concat = False, dir_class = LAENTIEC_CLASS):
    
    if concat == True:
        # 1. Create dataset
        dataset_a, masks_pred_size, dir_checkpoint =  utils.data_loading.create_dataset("rgbdt")
        dataset_b, masks_pred_size, dir_checkpoint =  utils.data_loading.create_dataset(dataset_str)

        dataset_a_train, dataset_a_val = split_dataset(dataset_a, val_percent)
        dataset_b_train, dataset_b_val = split_dataset(dataset_b, val_percent)

        train_concat_dataset = ConcatDataset((dataset_a_train, dataset_b_train))
        val_concat_dataset   = ConcatDataset((dataset_a_val, dataset_b_val))

        train_batch_sampler = MultiBatchSampler(dataset_a_train, dataset_b_train, batch_size)
        val_batch_sampler   = MultiBatchSampler(dataset_a_val, dataset_b_val, batch_size)

        n_train = len(train_concat_dataset)
        n_val   = len(val_concat_dataset)

        # 3. Create data loaders
        loader_args  = dict(pin_memory = True, num_workers = 4)
        train_loader = DataLoader(train_concat_dataset, batch_sampler = train_batch_sampler, **loader_args)
        val_loader   = DataLoader(val_concat_dataset,   batch_sampler = val_batch_sampler, **loader_args)
    
    elif concat == False:
        # 1. Create dataset
        dataset, masks_pred_size, dir_checkpoint = utils.data_loading.create_dataset(dataset_str)
        
        # 2. Split into train / validation partitions
        n_val   = int(len(dataset) * val_percent)
        n_train = len(dataset) - n_val
        train_set, val_set = random_split(dataset, [n_train, n_val], generator = torch.Generator().manual_seed(0))

        # 3. Create data loaders
        loader_args  = dict(batch_size = batch_size, num_workers = 4, pin_memory = True)
        train_loader = DataLoader(train_set, shuffle = True, **loader_args)
        val_loader   = DataLoader(val_set, shuffle = True, drop_last = True, **loader_args)

        # Getting filenames (IDs) for training and validation set
        train_imgs = [train_set.dataset.get_indexed_filename(i_) for i_ in train_set.indices]
        valid_imgs = [val_set.dataset.get_indexed_filename(i_) for i_ in val_set.indices]
        
        # Creating textfiles for each set
        current_time       = datetime.datetime.now()
        validation_img_str = "validation_" + dataset_str + "_" + current_time.strftime('%Y-%m-%d_%H-%M-%S') + ".txt"
        training_img_str   = "training_" + dataset_str + "_" + current_time.strftime('%Y-%m-%d_%H-%M-%S') + ".txt"
        utils.img_handler.create_list_txt(validation_img_str, valid_imgs)
        utils.img_handler.create_list_txt(training_img_str, train_imgs)

    # (Initialize logging)
    experiment = wandb.init(project = 'OmniUNET')
    experiment_config = dict(epochs = epochs, batch_size = batch_size, learning_rate = learning_rate,
                            val_percent = val_percent, save_checkpoint = save_checkpoint, amp = amp)
    experiment.config.update(experiment_config)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler
    optimizer   = optim.RMSprop(net.parameters(), lr = learning_rate, weight_decay = 1e-8, momentum = 0.9)
    scheduler   = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience = 40)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled = amp)
    criterion   = nn.CrossEntropyLoss()


    global_step = 0
    # 5. Begin training
    for epoch in range(1, epochs + 1):
        net.train()
        epoch_loss = 0
        with tqdm(total = n_train, desc = f'Epoch {epoch}/{epochs}', unit = 'img') as pbar:
            for batch in train_loader:
                images     = batch['image']
                true_masks = batch['mask']
                image_id   = batch['image_id']
                
                # Training multimodality
                images, missing_case = utils.img_handler.missing_modality_selector(images)

                images     = images.to(device = device, dtype = torch.float32)
                true_masks = true_masks.to(device = device, dtype = torch.long)

                with torch.cuda.amp.autocast(enabled = amp):
                    if missing_case == 'T' or missing_case == 'D':
                        masks_pred, _ = net(images, image_type = missing_case)
                    else:
                        masks_pred, _ = net(images)
                        
                    masks_pred       = F.interpolate(masks_pred, size = masks_pred_size, mode = 'bicubic', align_corners = False)
                    mask_true_onehot = F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float()
                    mask_pred_onehot = F.softmax(masks_pred, dim = 1).float()
                    pixel_acc, IoU, dice_score, precision, recall = utils.metrics.calculate_multi_metrics(mask_true_onehot, mask_pred_onehot, net.n_classes, average = True)

                    # Validation dice loss
                    val_dice_loss    = 1 - dice_score
                    # Criterion should be with logits, before softmax
                    loss             = criterion(masks_pred, true_masks) + val_dice_loss
                    
                optimizer.zero_grad(set_to_none = True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss  += loss.item()
                experiment.log({
                    "training": {
                        'train loss': loss.item(),
                        'step': global_step,
                        'epoch': epoch
                        }
                    })
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                
                # Evaluation round
                division_step = (n_train // (10 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        # Some metrics may be innacurate for large batches
                        pixel_acc_score, IoU_score, val_dice_score, precision_score, recall_score, validation_loss = utils.evaluate_net.evaluate_missing_modality(net, val_loader, device, masks_pred_size, dataset_str)
                        scheduler.step(val_dice_score)
                        
                        in_channels = images.shape[1]
                        
                        debug_list = []
                        image_name = str(image_id[0])
                        
                        if missing_case == 'T':
                            ### Input thermal Image
                            thermal_tensor = images[0,0,...]
                            thermal_numpy  = thermal_tensor.cpu().detach().numpy()
                            wandb_thermal  = wandb.Image(thermal_numpy, caption = "Thermal image")
                            debug_list.append(wandb_thermal)

                        elif missing_case == 'D':
                            ### Input depth Image
                            depth_tensor = images[0,0,...]
                            depth_numpy  = depth_tensor.cpu().detach().numpy()
                            wandb_depth  = wandb.Image(depth_numpy, caption = "Depth image")
                            debug_list.append(wandb_depth)

                        else:
                        ### Input RGB Image
                            Rgb_tensor  = images[0,0:3,...].permute(1, 2, 0)
                            Rgb_numpy   = Rgb_tensor.cpu().detach().numpy()
                            Rgb_uint8   = Rgb_numpy * 255 # original is float 0-1
                            Rgb_uint8   = Rgb_uint8.astype(np.uint8)
                            rgb_caption = "RGB image, id: " + image_name
                            wandb_rgb   = wandb.Image(Rgb_uint8, caption = rgb_caption)
                            debug_list.append(wandb_rgb)

                            if in_channels > 3: 
                                
                                ### Input depth Image
                                depth_tensor = images[0,3,...]
                                depth_numpy  = depth_tensor.cpu().detach().numpy()
                                wandb_depth  = wandb.Image(depth_numpy, caption = "Depth image")
                                debug_list.append(wandb_depth)

                                if in_channels == 5:
                                    
                                    ### Input thermal Image
                                    thermal_tensor = images[0,4,...]
                                    thermal_numpy  = thermal_tensor.cpu().detach().numpy()
                                    wandb_thermal  = wandb.Image(thermal_numpy, caption = "Thermal image")
                                    debug_list.append(wandb_thermal)
                       
                    
                        classes_list, class_color_list  = utils.img_handler.parse_class_file(dir_class)
                        ### Groundtruth mask
                        true_mask        = true_masks[0].float().cpu()
                        PIL_true_mask    = utils.img_handler.plot_mask_and_legend(image_name, true_mask, classes_list, class_color_list)
                        wandb_true_mask  = wandb.Image(PIL_true_mask, caption = "Groundtruth")

                        ### Predicted mask
                        pred_mask        = torch.softmax(masks_pred, dim = 1).argmax(dim = 1)[0].float().cpu()
                        PIL_pred_mask    = utils.img_handler.plot_mask_and_legend(image_name, pred_mask, classes_list, class_color_list)
                        wandb_pred_mask  = wandb.Image(PIL_pred_mask, caption = "Predicted mask")

                        print("\n")
                        logging.info('Pixel accuracy (%):      {}'.format(str(round(pixel_acc_score * 100, 2))))
                        logging.info('Val. Dice score (%):     {}'.format(str(round(val_dice_score  * 100, 2))))
                        logging.info('Precision accuracy (%):  {}'.format(str(round(precision_score * 100, 2))))
                        logging.info('Recall (%):              {}'.format(str(round(recall_score    * 100, 2))))
                        logging.info('Number of entrance channels: {}'.format(in_channels))

                        experiment.log({
                            'validation': {
                                'learning rate'  : optimizer.param_groups[0]['lr'],
                                'validation loss': validation_loss,
                                'validation Dice': val_dice_score,
                                'pixel accuracy' : pixel_acc_score,
                                'precision'      : precision_score,
                                'recall'         : recall_score,
                            },
                            'imgs'           : debug_list,
                            'masks'          : [wandb_true_mask, wandb_pred_mask],
                        })

        if save_checkpoint:
            pathlib.Path(dir_checkpoint).mkdir(parents = True, exist_ok = True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description = 'Train the OmniUnet on images and target masks')
    parser.add_argument('--epochs',           '-e', type = int,   default = 5,     help = 'Number of epochs')
    parser.add_argument('--batch-size',       '-b', type = int,   default = 1,     dest = 'batch_size', help = 'Batch size')
    parser.add_argument('--learning-rate',    '-l', type = float, default = 1e-5,  dest ='lr', help = 'Learning rate')
    parser.add_argument('--load',             '-f', type = str,   default = '',    help='Load model from a .pth file')
    parser.add_argument('--scale',            '-s', type = float, default = 0.5,   help='Downscaling factor of the images')
    parser.add_argument('--validation',       '-v', type = float, default = 10.0,  dest='val',  help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--classes',          '-c', type = int,   default = 2,     help='Number of classes')
    parser.add_argument('--dataset',          '-d', type = str,   default = '',    help='Dataset to use')
    parser.add_argument('--max_channels',     '-m', type = int,   default = '3',   help='Max channels')
    parser.add_argument('--pretrain_classes', '-p', type = int,   default = '6',   help='Pretrained classes')
    parser.add_argument('--gpu',              '-u', type = str,   default = '0',   help='Gpu to use')
    parser.add_argument('--amp',         action = 'store_true', default = True, help='Use mixed precision')
    parser.add_argument('--bilinear',    action = 'store_true', default = False, help='Use bilinear upsampling')

    return parser.parse_args()


if __name__ == '__main__':
   
    args   = get_args()
    cuda_  = "cuda:" + args.gpu
    device = torch.device(cuda_ if torch.cuda.is_available() else 'cpu')

    logging.basicConfig(level = logging.INFO, format = '%(levelname)s: %(message)s')
    logging.info(f'Using device {device}')

    # Defining dataset type
    depthmode         = "summed_tokens"
    input_dataset_str = args.dataset.upper()

    if input_dataset_str    == 'RGBDT':
        channels = 5
        dir_class = LAENTIEC_CLASS

    if args.load:
        net = OmniUnet(max_channels = args.max_channels, n_classes = args.pretrain_classes, bilinear = args.bilinear, depth_mode = depthmode)
        logging.info(f'Network: {net.n_classes} output channels (classes)')

        # Load the model
        loaded_model  = torch.load(args.load, map_location = device)
        net.load_state_dict(loaded_model, strict = True)
        
        # Changing classes to new training
        net.n_channels = channels
        net.n_classes  = args.classes
        n_ftrs         = net.outc.in_channels
        
        # Last layer converted to the one needed for training
        net.outc       = unet_decoder.OutConv(n_ftrs, args.classes)

        logging.info(f'Model loaded from {args.load}')
    else:
        net = OmniUnet(max_channels = args.max_channels, n_classes = args.classes, bilinear = args.bilinear, depth_mode = depthmode)

    try:
        net.to(device = device)
        train_net(net = net, epochs = args.epochs, batch_size = args.batch_size, learning_rate = args.lr,
                  device = device, val_percent = args.val / 100, amp = args.amp, dataset_str = input_dataset_str, 
                  concat = False, dir_class = dir_class)
        
    except KeyboardInterrupt:
        torch.save(net.state_dict(), './checkpoints/INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise
