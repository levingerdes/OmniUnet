import argparse, logging, os, sys, torch
import numpy as np
import torch.nn.functional as F
import utils.data_loading as dataloading
import utils.img_handler as img_handler
import warnings
import utils.metrics
import matplotlib.pyplot as plt
import matplotlib
# To remove memory leaks around 27 iterations
matplotlib.use('Agg')

from matplotlib.colors import ListedColormap
import pandas as pd
from os import listdir
from os.path import isfile, join
import time

warnings.filterwarnings("ignore") 
from model.omniunet_model import OmniUnet

THIS_FOLDER     = os.path.dirname(os.path.abspath(__file__))
LAENTIEC_CLASS  = os.path.join(THIS_FOLDER, 'classes/LAENTIEC')
RUGD_CLASS      = os.path.join(THIS_FOLDER, 'classes/RUGD')
BASEPROD_CLASS  = os.path.join(THIS_FOLDER, 'classes/BASEPROD')

validation_list ="validation_RGBDT_2024-05-29_10-33-15.txt"

# Names of output folders
prediction_folder              = "predictions/"
true_masks_folder              = "true_masks/"
legend_prediction_folder       = "legend_masks/"
comp_prediction_folder         = "comparative_predictions/"
comp_input_prediction_folder   = "comparative_input_predictions/"
metrics_folder                 = "metrics/"
article_folder                 = "article_logs/"

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model',             '-m', default = 'MODEL.pth', metavar = 'FILE', help = 'Specify the file in which the model is stored')
    parser.add_argument('--rgb_folder',        '-i', type = str,   default = '',    help = 'Filenames of input images')
    parser.add_argument('--scale',             '-s', type = float, default = 1,     help = 'Scale factor for the input images')
    parser.add_argument('--bilinear',                type = bool,  default = False, help = 'Use bilinear upsampling')
    parser.add_argument('--classes',           '-c', type = int,   default = 2,     help = 'Number of classes')
    parser.add_argument('--dataset',           '-d', type = str,   default = '',    help = 'Dataset name')
    parser.add_argument('--depth_folder',      '-p', type = str,   default = '',    help = 'Depth filename')
    parser.add_argument('--thermal_folder',    '-T', type = str,   default = '',    help = 'Thermal filename')
    parser.add_argument('--mask_true_folder',  '-g', type = str,   default = '',    help = 'Mask true filename')
    parser.add_argument('--remove_background', '-r', default = True,  action = 'store_true', help = 'Remove void class for metrics')
    parser.add_argument('--output_folder', '-o', type = str,   default = 'output_files/',    help = 'Output folder')
    parser.add_argument('--file_list', '-l', default = False, action = 'store_true', help = 'File list to verify')

    return parser.parse_args()


if __name__ == '__main__':
    
    args      = get_args()
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.basicConfig(level = logging.INFO, format = '%(levelname)s: %(message)s')

    # Defining dataset type
    depthmode         = "summed_tokens"
    input_dataset_str = args.dataset.upper()

    # Check if output folder exists
    img_handler.create_folder(args.output_folder)

    # Creating output folders
    output_prediction_folder        =  args.output_folder + prediction_folder  
    output_true_masks_folder        =  args.output_folder + true_masks_folder  
    output_legend_prediction_folder =  args.output_folder + legend_prediction_folder  
    output_comp_prediction_folder   =  args.output_folder + comp_prediction_folder
    output_metrics_folder           =  args.output_folder + metrics_folder  
    output_input_prediction_folder  =  args.output_folder + comp_input_prediction_folder  
    img_handler.create_folder(output_prediction_folder)
    img_handler.create_folder(output_true_masks_folder)
    img_handler.create_folder(output_legend_prediction_folder)
    img_handler.create_folder(output_comp_prediction_folder)
    img_handler.create_folder(output_metrics_folder)
    img_handler.create_folder(output_input_prediction_folder)

    # Saving for article
    output_article_folder =  args.output_folder + article_folder
    img_handler.create_folder(output_article_folder)

    # Searching files
    if input_dataset_str   == 'RGBDT':
        channels        = 5
        dir_class       = LAENTIEC_CLASS
        rgb_files       = [os.path.join(args.rgb_folder, f) for f in sorted(listdir(args.rgb_folder)) if isfile(join(args.rgb_folder, f))]
        depth_files     = [os.path.join(args.depth_folder, f) for f in sorted(listdir(args.depth_folder)) if isfile(join(args.depth_folder, f))]
        thermal_files   = [os.path.join(args.thermal_folder, f) for f in sorted(listdir(args.thermal_folder)) if isfile(join(args.thermal_folder, f))]
        mask_true_files = [os.path.join(args.mask_true_folder, f) for f in sorted(listdir(args.mask_true_folder)) if isfile(join(args.mask_true_folder, f))]
        
        # Search in validation list if presented
        if args.file_list:
            rgb_files       = sorted(img_handler.matching_id(validation_list, rgb_files))
            depth_files     = sorted(img_handler.matching_id(validation_list, depth_files))
            thermal_files   = sorted(img_handler.matching_id(validation_list, thermal_files))
            mask_true_files = sorted(img_handler.matching_id(validation_list, mask_true_files))

    elif input_dataset_str == 'RGBT':
        channels        = 4
        dir_class       = BASEPROD_CLASS
        rgb_files       = [os.path.join(args.rgb_folder, f) for f in sorted(listdir(args.rgb_folder)) if isfile(join(args.rgb_folder, f))]
        thermal_files   = [os.path.join(args.thermal_folder, f) for f in sorted(listdir(args.thermal_folder)) if isfile(join(args.thermal_folder, f))]
        mask_true_files = [os.path.join(args.mask_true_folder, f) for f in sorted(listdir(args.mask_true_folder)) if isfile(join(args.mask_true_folder, f))]
        
        # Search in validation list if presented
        if args.file_list:
            rgb_files       = sorted(img_handler.matching_id(validation_list, rgb_files))
            depth_files     = sorted(img_handler.matching_id(validation_list, depth_files))
            thermal_files   = sorted(img_handler.matching_id(validation_list, thermal_files))
            mask_true_files = sorted(img_handler.matching_id(validation_list, mask_true_files))

    elif input_dataset_str == 'RGBD':
        channels        = 4
        dir_class       = BASEPROD_CLASS
        rgb_files       = [os.path.join(args.rgb_folder, f) for f in sorted(listdir(args.rgb_folder)) if isfile(join(args.rgb_folder, f))]
        depth_files     = [os.path.join(args.depth_folder, f) for f in sorted(listdir(args.depth_folder)) if isfile(join(args.depth_folder, f))]
        mask_true_files = [os.path.join(args.mask_true_folder, f) for f in sorted(listdir(args.mask_true_folder)) if isfile(join(args.mask_true_folder, f))]
        
        # Search in validation list if presented
        if args.file_list:
            rgb_files       = sorted(img_handler.matching_id(validation_list, rgb_files))
            depth_files     = sorted(img_handler.matching_id(validation_list, depth_files))
            mask_true_files = sorted(img_handler.matching_id(validation_list, mask_true_files))

    elif input_dataset_str == 'RGB':
        channels        = 3
        dir_class       = LAENTIEC_CLASS
        rgb_files       = [os.path.join(args.rgb_folder, f) for f in sorted(listdir(args.rgb_folder)) if isfile(join(args.rgb_folder, f))]
        mask_true_files = [os.path.join(args.mask_true_folder, f) for f in sorted(listdir(args.mask_true_folder)) if isfile(join(args.mask_true_folder, f))]
        
        # Search in validation list if presented
        if args.file_list:
            rgb_files       = sorted(img_handler.matching_id(validation_list, rgb_files))
            mask_true_files = sorted(img_handler.matching_id(validation_list, mask_true_files))
    
    elif input_dataset_str == 'RUGD':
        channels        = 3
        dir_class       = RUGD_CLASS
        rgb_files       = [os.path.join(args.rgb_folder, f) for f in sorted(listdir(args.rgb_folder)) if isfile(join(args.rgb_folder, f))]
        mask_true_files = [os.path.join(args.mask_true_folder, f) for f in sorted(listdir(args.mask_true_folder)) if isfile(join(args.mask_true_folder, f))]
        
        # Search in validation list if presented
        if args.file_list:
            rgb_files       = sorted(img_handler.matching_id(validation_list, rgb_files))
            mask_true_files = sorted(img_handler.matching_id(validation_list, mask_true_files))

    elif input_dataset_str == 'T':
        channels        = 1
        dir_class       = BASEPROD_CLASS
        thermal_files   = [os.path.join(args.thermal_folder, f) for f in sorted(listdir(args.thermal_folder)) if isfile(join(args.thermal_folder, f))]
        mask_true_files = [os.path.join(args.mask_true_folder, f) for f in sorted(listdir(args.mask_true_folder)) if isfile(join(args.mask_true_folder, f))]

        # Search in validation list if presented
        if args.file_list:
            thermal_files   = sorted(img_handler.matching_id(validation_list, thermal_files))
            mask_true_files = sorted(img_handler.matching_id(validation_list, mask_true_files))

    elif input_dataset_str == 'D':
        channels        = 1
        dir_class       = BASEPROD_CLASS
        depth_files     = [os.path.join(args.depth_folder, f) for f in sorted(listdir(args.depth_folder)) if isfile(join(args.depth_folder, f))]
        mask_true_files = [os.path.join(args.mask_true_folder, f) for f in sorted(listdir(args.mask_true_folder)) if isfile(join(args.mask_true_folder, f))]
        
        # Search in validation list if presented
        if args.file_list:
            depth_files   = sorted(img_handler.matching_id(validation_list, depth_files))
            mask_true_files = sorted(img_handler.matching_id(validation_list, mask_true_files))

    
    else:
        logging.error(f"Specify correct dataset type")
        sys.exit()

    rgb_files       = [os.path.join(args.rgb_folder, f) for f in sorted(listdir(args.rgb_folder)) if isfile(join(args.rgb_folder, f))]
    depth_files     = [os.path.join(args.depth_folder, f) for f in sorted(listdir(args.depth_folder)) if isfile(join(args.depth_folder, f))]
    thermal_files   = [os.path.join(args.thermal_folder, f) for f in sorted(listdir(args.thermal_folder)) if isfile(join(args.thermal_folder, f))]
    mask_true_files = [os.path.join(args.mask_true_folder, f) for f in sorted(listdir(args.mask_true_folder)) if isfile(join(args.mask_true_folder, f))]
    
    # Search in validation list if presented
    if args.file_list:
        rgb_files       = sorted(img_handler.matching_id(validation_list, rgb_files))
        depth_files     = sorted(img_handler.matching_id(validation_list, depth_files))
        thermal_files   = sorted(img_handler.matching_id(validation_list, thermal_files))
        mask_true_files = sorted(img_handler.matching_id(validation_list, mask_true_files))


    ## Loading the network model
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net = OmniUnet(max_channels = 5, n_classes = args.classes, bilinear = args.bilinear, depth_mode = depthmode)
    net.to(device = device)
    loaded_model  = torch.load(args.model, map_location = "cpu")
    net.load_state_dict(loaded_model, strict = True)
    net.eval()

    logging.info(f'Network: {net.n_classes} output channels (classes)')
    logging.info('Model loaded!')
    logging.info(f'Input image with channels: {input_dataset_str}')
  
    # Creating empty class arrays for metrics
    metric_classes_num = net.n_classes
    if(args.remove_background):
        metric_classes_num = metric_classes_num - 1
    
    # Array to count how much each class appears in masks
    classes_counter = np.zeros(net.n_classes)

    # Average inference time
    avg_inference_time  = 0
    
    # Empty zero array to calculate sum
    pixel_acc_class_sum = np.zeros(metric_classes_num)
    IoU_class_sum       = np.zeros(metric_classes_num)
    dice_class_sum      = np.zeros(metric_classes_num)
    precision_class_sum = np.zeros(metric_classes_num)
    recall_class_sum    = np.zeros(metric_classes_num)

    # Empty zero array with ocurrence of classes
    pixel_acc_class_num = np.zeros(metric_classes_num)
    IoU_class_num       = np.zeros(metric_classes_num)
    dice_class_num      = np.zeros(metric_classes_num)
    precision_class_num = np.zeros(metric_classes_num)
    recall_class_num    = np.zeros(metric_classes_num)

    # Defining number of images
    N_images = len(mask_true_files)

    # Creating metrics .csv
    csv_file_name   = output_metrics_folder + "multi.csv"
    logging.info(f'Creating .csv of metrics at : {csv_file_name}')
    df_title        = pd.DataFrame(columns = ["Number of images:", N_images, "Type:", input_dataset_str])
    empty_df        = pd.DataFrame([[np.nan] * len(df_title.columns)], columns = df_title.columns)
    df_final        = pd.concat([df_title, empty_df], ignore_index = True)
    df_final.to_csv(csv_file_name, index = False)

    ## Preparing each image
    for i in range(N_images):
        logging.info(f'Predicting image number: {i} of {N_images}...')

        if input_dataset_str   == 'RGBDT':
            input_rgb     = dataloading.load_img(rgb_files[i])
            input_depth   = dataloading.load_img(depth_files[i])
            input_thermal = dataloading.load_img(thermal_files[i])
            mask_true     = dataloading.load_img(mask_true_files[i])
            input_img     = dataloading.RgbdtDataset.build_rgbdt(input_rgb, input_depth, input_thermal)

            logging.info(f'Input RGB image: {rgb_files[i]}')
            logging.info(f'Input DEPTH image: {depth_files[i]}')
            logging.info(f'Input THERMAL image: {thermal_files[i]}')
            logging.info(f'Input MASK image: {mask_true_files[i]}')
        
        elif input_dataset_str == 'RGBT':
            input_rgb     = dataloading.load_img(rgb_files[i])
            input_thermal = dataloading.load_img(thermal_files[i])
            mask_true     = dataloading.load_img(mask_true_files[i])
            input_img     = dataloading.RgbtDataset.build_rgbt(input_rgb, input_thermal)

            logging.info(f'Input RGB image: {rgb_files[i]}')
            logging.info(f'Input THERMAL image: {thermal_files[i]}')
            logging.info(f'Input MASK image: {mask_true_files[i]}')
            
        elif input_dataset_str == 'RGBD':
            input_rgb     = dataloading.load_img(rgb_files[i])
            input_depth   = dataloading.load_img(depth_files[i])
            mask_true     = dataloading.load_img(mask_true_files[i])
            input_img     = dataloading.RgbdDataset.build_rgbd(input_rgb, input_depth)
            
            logging.info(f'Input RGB image: {rgb_files[i]}')
            logging.info(f'Input DEPTH image: {depth_files[i]}')
            logging.info(f'Input MASK image: {mask_true_files[i]}')

        elif input_dataset_str == 'T':
            input_thermal = dataloading.load_img(thermal_files[i])
            input_img     = dataloading.TDataset.build_t(input_thermal)
            mask_true     = dataloading.load_img(mask_true_files[i])

            logging.info(f'Input THERMAL image: {thermal_files[i]}')
            logging.info(f'Input MASK image: {mask_true_files[i]}')

        elif input_dataset_str == 'D':
            input_depth   = dataloading.load_img(depth_files[i])
            input_img     = dataloading.DDataset.build_d(input_depth)
            mask_true     = dataloading.load_img(mask_true_files[i])

            logging.info(f'Input DEPTH image: {depth_files[i]}')
            logging.info(f'Input MASK image: {mask_true_files[i]}')

        elif input_dataset_str == 'RGB' or input_dataset_str == 'RUGD':
            input_rgb     = dataloading.load_img(rgb_files[i])
            processed_img = dataloading.preprocess_img(input_rgb, args.scale, is_mask = False)
            input_img     = torch.from_numpy(processed_img)
            input_img     = input_img.unsqueeze(1)
            input_img     = input_img.unsqueeze(0)
            mask_true     = dataloading.load_img(mask_true_files[i])
            
            logging.info(f'Input RGB image: {rgb_files[i]}')
            logging.info(f'Input MASK image: {mask_true_files[i]}')
        
        else:
            logging.error(f"Specify correct dataset type")
            sys.exit()

        
        img = input_img.to(device = device, dtype = torch.float32)
        
        with torch.no_grad():
            inference_start    = time.time() * 1000 # millisecons
            if input_dataset_str == 'T' or input_dataset_str == 'D':
                mask_pred, omnivore_only = net(img, image_type = input_dataset_str)
            else:
                mask_pred, omnivore_only = net(img)
            inference_end      = time.time() * 1000
            elapsed_inference  = inference_end - inference_start
            avg_inference_time = avg_inference_time + elapsed_inference
            mask_pred_size     = (input_img.shape[3], input_img.shape[4])

            logging.info(f'Prediction mask interpolated size: {mask_pred_size}')
            logging.info(f'Image number {i} took {round(elapsed_inference, 2)} ms to be predicted.')

        # Producing prediction mask
        mask_pred_interp   = F.interpolate(mask_pred, size = mask_pred_size, mode = 'bicubic', align_corners = False)
        mask_pred          = torch.softmax(mask_pred_interp, dim = 1).argmax(dim = 1)[0].float().cpu()
        
        # Creating one_hot (N binary channels for classes) mask to calculate metrics  
        mask_ndarray       = np.asarray(mask_true)
        mask_true_tensor   = torch.as_tensor(mask_ndarray).long()
        mask_true_tensor   = mask_true_tensor.to(device = device, dtype = torch.long)
        mask_true_tensor   = mask_true_tensor.unsqueeze(0)
        mask_true_one_hot  = F.one_hot(mask_true_tensor, net.n_classes).permute(0, 3, 1, 2).float().cpu()
        mask_pred_one_hot  = F.one_hot(mask_pred_interp.argmax(dim = 1), net.n_classes).permute(0, 3, 1, 2).float().cpu()
        
        # Saving fig of pred mask
        classes_list, class_color_list  = img_handler.parse_class_file(dir_class)
        input_file_id                   = img_handler.extract_img_id(mask_true_files[i])
        legend_mask                     = img_handler.plot_mask_and_legend(input_file_id, mask_pred, classes_list, class_color_list)
    
        # Saving standalone and legend image
        save_pred_fig_dir        = output_prediction_folder + input_file_id + "_out.png"
        save_legend_fig_dir      = output_legend_prediction_folder + input_file_id + "_legend_out.png"
        save_mask_true_fig_dir   = output_true_masks_folder + input_file_id + "_mask_true.png"
        save_mask_comp_fig_dir   = output_comp_prediction_folder + input_file_id + "_mask_comp.png"
        save_input_comp_fig_dir  = output_input_prediction_folder + input_file_id + "_input_mask_comp.png"

        # Save the different outputs
        cmap_color_list          = ListedColormap(class_color_list)

        # With RGB by default
        if input_dataset_str == 'RGB' or input_dataset_str == 'RGBD' or input_dataset_str == 'RGBDT' or input_dataset_str == 'RUGD':
            comp_mask            = img_handler.plot_mask_comparison(input_file_id, mask_true, mask_pred, classes_list, class_color_list)
            legend_mask.save(save_legend_fig_dir)
            comp_mask.save(save_mask_comp_fig_dir)
            plt.imsave(save_pred_fig_dir, mask_pred, cmap = cmap_color_list, origin = 'upper', dpi = 300)
            plt.imsave(save_mask_true_fig_dir, mask_ndarray, cmap = cmap_color_list, origin = 'upper', dpi = 300)

            if input_dataset_str == 'RGBDT':
                depth_matrix     = input_img[:,3,:,:,:].squeeze(0).squeeze(0)
                thermal_matrix   = input_img[:,4,:,:,:].squeeze(0).squeeze(0)
                input_comp_mask  = img_handler.plot_mask_comparison_with_input(input_file_id, mask_true, mask_pred, input_rgb, thermal_matrix, 
                                                                                classes_list, class_color_list)
                input_comp_mask.save(save_input_comp_fig_dir)
            elif input_dataset_str == 'RGBD':
                depth_matrix     = input_img[:,3,:,:,:].squeeze(0).squeeze(0)
            
        elif input_dataset_str == 'T':
            thermal_matrix   = input_img[:,0,:,:,:].squeeze(0).squeeze(0)
            input_thermal =np.genfromtxt(thermal_files[i], delimiter = ',')
            input_rgb     = dataloading.load_img(rgb_files[i])
            comp_mask            = img_handler.plot_mask_comparison(input_file_id, mask_true, mask_pred, classes_list, class_color_list)
            input_comp_mask  = img_handler.plot_mask_comparison_with_input(input_file_id, mask_true, mask_pred, input_rgb, input_thermal, 
                                                                                classes_list, class_color_list)
            input_comp_mask.save(save_input_comp_fig_dir)
            legend_mask.save(save_legend_fig_dir)
            comp_mask.save(save_mask_comp_fig_dir)
        
        elif input_dataset_str == 'D':
            depth_matrix     = input_img[:,0,:,:,:].squeeze(0).squeeze(0)
            input_thermal =np.genfromtxt(thermal_files[i], delimiter = ',')
            input_rgb     = dataloading.load_img(rgb_files[i])
            comp_mask            = img_handler.plot_mask_comparison(input_file_id, mask_true, mask_pred, classes_list, class_color_list)
            input_comp_mask  = img_handler.plot_mask_comparison_with_input(input_file_id, mask_true, mask_pred, input_rgb, input_thermal, 
                                                                                classes_list, class_color_list)
            input_comp_mask.save(save_input_comp_fig_dir)
            legend_mask.save(save_legend_fig_dir)
            comp_mask.save(save_mask_comp_fig_dir)
    

        # Saving for article
        # Create folder for this image id
        folder_image_id = output_article_folder + input_file_id + "/"
        plot_n_classes  = len(class_color_list) - 1
        img_handler.create_folder(folder_image_id)
        
        # Creating folder names
        article_pred_str    = folder_image_id + input_file_id + "_pred.png"
        article_true_str    = folder_image_id + input_file_id + "_true.png"
        article_rgb_str     = folder_image_id + input_file_id + "_rgb.png"
        article_thermal_str = folder_image_id + input_file_id + "_thermal.png"
        article_depth_str   = folder_image_id + input_file_id + "_depth.png"

        # Saving standalone images
        # Saving prediction masks and rgb for all cases
        plt.imsave(article_pred_str, mask_pred,    cmap = cmap_color_list, origin = 'upper', dpi = 300, vmin = 0, vmax = plot_n_classes)
        plt.imsave(article_true_str, mask_ndarray, cmap = cmap_color_list, origin = 'upper', dpi = 300, vmin = 0, vmax = plot_n_classes)
        if input_dataset_str   == 'RGB':
            input_rgb.save(article_rgb_str)

        if input_dataset_str   == 'RGBDT':
            plt.imsave(article_depth_str, depth_matrix, cmap = "gray", origin = 'upper', dpi = 300, vmin = 0)
            plt.imsave(article_thermal_str, thermal_matrix, origin = 'upper', dpi = 300, cmap = 'plasma', vmin = 0)
            input_rgb.save(article_rgb_str)
         
        elif input_dataset_str == 'RGBD':
            plt.imsave(article_depth_str, depth_matrix, cmap = "gray", origin = 'upper', dpi = 300, vmin = 0)
            input_rgb.save(article_rgb_str)

        elif input_dataset_str == 'T':
            plt.imsave(article_thermal_str, thermal_matrix, origin = 'upper', dpi = 300, cmap = 'plasma', vmin = 0)
        
        elif input_dataset_str == 'D':
            plt.imsave(article_depth_str, depth_matrix, cmap = "gray", origin = 'upper', dpi = 300, vmin = 0)

        logging.info(f'Original output image saved at: {save_pred_fig_dir}')
        logging.info(f'Output image with legend saved at: {save_legend_fig_dir}')
        logging.info(f'Output comparative image with legend saved at: {save_mask_comp_fig_dir}')
        logging.info(f'Output comparative image with inputs saved at: {save_input_comp_fig_dir}')

        # Calculating average metrics
        avg_pixel_acc, avg_IoU, avg_dice, avg_precision, avg_recall = utils.metrics.calculate_multi_metrics(mask_true_one_hot, mask_pred_one_hot, 
                                                                                        net.n_classes, average = True)
        logging.info(f'''Calculating metrics:
            Pixel accuracy (%):   {str(round(avg_pixel_acc * 100, 2))}
            IoU (%):              {str(round(avg_IoU * 100, 2))}
            Dice coeff. (%):      {str(round(avg_dice * 100, 2))}
            Precision (%):        {str(round(avg_precision * 100, 2))}
            Recall (%):           {str(round(avg_recall * 100, 2))}
            ''')
        
        # Calculating per class metrics
        pixel_acc, IoU, dice, precision, recall = utils.metrics.calculate_multi_metrics(mask_true_one_hot, mask_pred_one_hot, 
                                                                                   net.n_classes, average = False)
        # Consider nan as zero 
        pixel_acc_class_sum   =  np.nansum(np.stack((pixel_acc_class_sum, pixel_acc)), axis = 0)
        IoU_class_sum         =  np.nansum(np.stack((IoU_class_sum, IoU)), axis = 0)
        dice_class_sum        =  np.nansum(np.stack((dice_class_sum, dice)), axis = 0)
        precision_class_sum   =  np.nansum(np.stack((precision_class_sum, precision)), axis = 0)
        recall_class_sum      =  np.nansum(np.stack((recall_class_sum, recall)), axis = 0)

        # Create an array where Nan is zero and a number is 1
        pixel_acc_class_mask  = 1 * ~np.isnan(pixel_acc)
        IoU_class_mask        = 1 * ~np.isnan(IoU)
        dice_class_mask       = 1 * ~np.isnan(dice)
        precision_class_mask  = 1 * ~np.isnan(precision)
        recall_class_mask     = 1 * ~np.isnan(recall)

        # Add to create array with class ocurrences 
        pixel_acc_class_num   = np.add(pixel_acc_class_num, pixel_acc_class_mask).astype(int)
        IoU_class_num         = np.add(IoU_class_num, IoU_class_mask).astype(int)
        dice_class_num        = np.add(dice_class_num, pixel_acc_class_mask).astype(int)
        precision_class_num   = np.add(precision_class_num, pixel_acc_class_mask).astype(int)
        recall_class_num      = np.add(recall_class_num, pixel_acc_class_mask).astype(int)

        # Checking if background class is included
        metrics_classes_list = classes_list
        if(args.remove_background):
            metrics_classes_list = classes_list[1:]
            
        # Creating dataframe
        # Metrics per classs
        data_per_class = {
            'Class name':         metrics_classes_list,
            'Pixel accuracy (%)': np.round(pixel_acc * 100, decimals = 2),
            'IoU (%)':            np.round(IoU * 100, decimals = 2),
            'Dice coeff. (%)':    np.round(dice * 100, decimals = 2),
            'Precision (%)':      np.round(precision * 100, decimals = 2),
            'Recall (%)':         np.round(recall * 100, decimals = 2)
        }
        
        # Average of all classes
        avg_data = {
            'Class name':         "Average",
            'Pixel accuracy (%)': [round(avg_pixel_acc * 100, 2)],
            'IoU (%)':            [round(avg_IoU * 100, 2)],
            'Dice coeff. (%)':    [round(avg_dice * 100, 2)],
            'Precision (%)':      [round(avg_precision * 100, 2)],
            'Recall (%)':         [round(avg_recall * 100, 2)]
            }
        
        # Creating Title row
        df_title        = pd.DataFrame(columns = ["Image Number:", i, "Image ID:", input_file_id, "Type:", input_dataset_str])
        df_title.to_csv(csv_file_name, index = False, mode = "a")

        # Filling with Data
        df1        = pd.DataFrame(data_per_class)
        df2        = pd.DataFrame(avg_data)
        empty_df   = pd.DataFrame([[np.nan] * len(df1.columns)], columns = df1.columns)
        df_final   = pd.concat([df1, df2, empty_df], ignore_index = True)
        df_final.to_csv(csv_file_name, index = False, mode = "a")

        # Counter of hom much each class appears in true masks 
        true_mask_matrix = np.asarray(mask_true)
        values_true      = np.unique(true_mask_matrix.ravel()).astype(int)
        mask_classes     = np.zeros(net.n_classes)
        mask_classes[values_true] = 1
        classes_counter = classes_counter + mask_classes


    ## Calculating average of metrics per class
    pixel_acc_class_avg    =  np.divide(pixel_acc_class_sum, pixel_acc_class_num)
    IoU_class_avg          =  np.divide(IoU_class_sum, IoU_class_num)
    dice_class_avg         =  np.divide(dice_class_sum, dice_class_num)
    precision_class_avg    =  np.divide(precision_class_sum, precision_class_num)
    recall_class_avg       =  np.divide(recall_class_sum, recall_class_num)

    avg_per_class_data = {
        'Class name':         metrics_classes_list,
        'Pixel accuracy (%)': np.round(pixel_acc_class_avg * 100, decimals = 2),
        'IoU (%)':            np.round(IoU_class_avg * 100, decimals = 2),
        'Dice coeff. (%)':    np.round(dice_class_avg * 100, decimals = 2),
        'Precision (%)':      np.round(precision_class_avg * 100, decimals = 2),
        'Recall (%)':         np.round(recall_class_avg * 100, decimals = 2)
    }

    logging.info(f'''Average metrics per class:
        Class:                {metrics_classes_list}
        Pixel accuracy (%):   {np.round(pixel_acc_class_avg * 100, decimals = 2)}
        IoU (%):              {np.round(IoU_class_avg * 100, decimals = 2)}
        Dice coeff. (%):      {np.round(dice_class_avg * 100, decimals = 2)}
        Precision (%):        {np.round(precision_class_avg * 100, decimals = 2)}
        Recall (%):           {np.round(recall_class_avg * 100, decimals = 2)}
    ''')

    # Filling with Avg. Metrics per class
    # Creating Title row
    df_title        = pd.DataFrame(columns = ["Average metrics per class"])
    df_title.to_csv(csv_file_name, index = False, mode = "a")
    df1             = pd.DataFrame(avg_per_class_data)
    df1.to_csv(csv_file_name, index = False, mode = "a")

    logging.info(f'''True masks per class:
        Class:                {classes_list}
        Class ocurrences:     {classes_counter}
    ''')

    # Logging inference time
    avg_inference_time =  avg_inference_time / N_images
    logging.info(f'''Average inference time:  {avg_inference_time}''')