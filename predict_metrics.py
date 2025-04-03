import argparse, logging, os, sys, torch
import numpy as np
import torch.nn.functional as F
import utils.data_loading as dataloading
import utils.img_handler as img_handler
import warnings
import utils.metrics
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd

warnings.filterwarnings("ignore") 
from model.omniunet_model import OmniUnet

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
dir_class   = os.path.join(THIS_FOLDER, 'classes/LAENTIEC')

# Names of output folders
prediction_folder              = "predictions/"
true_masks_folder              = "true_masks/"
legend_prediction_folder       = "legend_masks/"
comp_prediction_folder         = "comparative_predictions/"
comp_input_prediction_folder   = "comparative_input_predictions/"
metrics_folder                 = "metrics/"

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model',         '-m', default = 'MODEL.pth', metavar = 'FILE', help = 'Specify the file in which the model is stored')
    parser.add_argument('--rgb_input',     '-i', type = str,   default = '',    help = 'Filenames of input images')
    parser.add_argument('--scale',         '-s', type = float, default = 1,     help = 'Scale factor for the input images')
    parser.add_argument('--bilinear',            type = bool,  default = False, help = 'Use bilinear upsampling')
    parser.add_argument('--classes',       '-c', type = int,   default = 2,     help = 'Number of classes')
    parser.add_argument('--dataset',       '-d', type = str,   default = '',    help = 'Dataset name')
    parser.add_argument('--depth_input',   '-p', type = str,   default = '',    help = 'Depth filename')
    parser.add_argument('--thermal_input', '-T', type = str,   default = '',    help = 'Thermal filename')
    parser.add_argument('--mask_true',     '-g', type = str,   default = '',    help = 'Mask true filename')
    parser.add_argument('--remove_background', '-r', type = bool,  default = True,  help = 'Remove void class for metrics')
    parser.add_argument('--output_folder', '-o', type = str,   default = 'output_files/',    help = 'Output folder')

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

    if input_dataset_str   == 'RGBDT':
        channels      = 5
        input_rgb     = dataloading.load_img(args.rgb_input)
        input_depth   = dataloading.load_img(args.depth_input)
        input_thermal = dataloading.load_img(args.thermal_input)
        mask_true     = dataloading.load_img(args.mask_true)
        input_img     = dataloading.RgbdtDataset.build_rgbdt(input_rgb, input_depth, input_thermal)
    
    elif input_dataset_str == 'RGBT':
        channels      = 4
        input_rgb     = dataloading.load_img(args.rgb_input)
        input_thermal = dataloading.load_img(args.thermal_input)
        mask_true     = dataloading.load_img(args.mask_true)
        input_img     = dataloading.RgbtDataset.build_rgbt(input_rgb, input_thermal)
        
    elif input_dataset_str == 'RGBD':
        channels      = 4
        input_rgb     = dataloading.load_img(args.rgb_input)
        input_depth   = dataloading.load_img(args.depth_input)
        mask_true     = dataloading.load_img(args.mask_true)
        input_img     = dataloading.RgbdDataset.build_rgbd(input_rgb, input_depth)

    elif input_dataset_str == 'RGB':
        channels      = 3
        input_rgb     = dataloading.load_img(args.rgb_input)
        processed_img = dataloading.preprocess_img(input_rgb, args.scale_factor, is_mask = False)
        input_img     = torch.from_numpy(processed_img)
        input_img     = input_img.unsqueeze(0)
        mask_true     = dataloading.load_img(args.mask_true)
    
    else:
        logging.error(f"Specify correct dataset type")
        sys.exit()
    
    logging.info(f'Input image with channels: {input_dataset_str}')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net = OmniUnet(max_channels = 5, n_classes = args.classes, bilinear = args.bilinear, depth_mode = depthmode)
    net.to(device = device)
    loaded_model  = torch.load(args.model, map_location = "cpu")
    net.load_state_dict(loaded_model, strict = True)
    
    logging.info(f'Network: {net.n_classes} output channels (classes)')
    logging.info('Model loaded!')
    logging.info(f'Predicting image: {args.rgb_input}...')
    
    net.eval()
    img = input_img.to(device = device, dtype = torch.float32)
    
    with torch.no_grad():
        mask_pred, omnivore_only = net(img)
        mask_pred_size = (input_img.shape[3], input_img.shape[4])
        logging.info(f'Prediction mask interpolated size: {mask_pred_size}')

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
    input_file_id                   = img_handler.extract_img_id(args.rgb_input)
    legend_mask                     = img_handler.plot_mask_and_legend(input_file_id, mask_pred, classes_list, class_color_list)

    # Saving standalone and legend image
    save_pred_fig_dir        = output_prediction_folder + input_file_id + "_out.png"
    save_legend_fig_dir      = output_legend_prediction_folder + input_file_id + "_legend_out.png"
    save_mask_true_fig_dir   = output_true_masks_folder + input_file_id + "_mask_true.png"
    save_mask_comp_fig_dir   = output_comp_prediction_folder + input_file_id + "_mask_comp.png"
    save_input_comp_fig_dir  = output_input_prediction_folder + input_file_id + "_input_mask_comp.png"

    # Save the different outputs
    cmap_color_list          = ListedColormap(class_color_list)
    comp_mask                = img_handler.plot_mask_comparison(input_file_id, mask_true, mask_pred, classes_list, class_color_list)
    thermal_csv              = img_handler.loadCsv(args.thermal_input, 720, 1280)
    input_comp_mask          = img_handler.plot_mask_comparison_with_input(input_file_id, mask_true, mask_pred, input_rgb, thermal_csv, 
                                                                           classes_list, class_color_list)
    plt.imsave(save_pred_fig_dir, mask_pred, cmap = cmap_color_list, origin = 'upper', dpi = 300)
    plt.imsave(save_mask_true_fig_dir, mask_ndarray, cmap = cmap_color_list, origin = 'upper', dpi = 300)
    legend_mask.save(save_legend_fig_dir)
    input_comp_mask.save(save_input_comp_fig_dir)
    comp_mask.save(save_mask_comp_fig_dir)

    logging.info(f'Original output image saved at: {save_pred_fig_dir}')
    logging.info(f'Output image with legend saved at: {save_legend_fig_dir}')
    logging.info(f'Output comparative image with legend saved at: {save_mask_comp_fig_dir}')
    logging.info(f'Output comparative image with inputs saved at: {save_input_comp_fig_dir}')

    # Calculating average metrics
    avg_pixel_acc, avg_IoU, avg_dice, avg_precision, avg_recall = utils.metrics.calculate_multi_metrics(mask_true_one_hot, mask_pred_one_hot, 
                                                                                    net.n_classes, average = True)
    logging.info(f'''Calculating metrics:
        Pixel accuracy (%):   {str(round(avg_pixel_acc * 100, 2))}
        Dice coeff. (%):      {str(round(avg_dice * 100, 2))}
        Precision (%):        {str(round(avg_precision * 100, 2))}
        Recall (%):           {str(round(avg_recall * 100, 2))}
        ''')
    
    # Calculating per class metrics
    pixel_acc, IoU, dice, precision, recall = utils.metrics.calculate_multi_metrics(mask_true_one_hot, mask_pred_one_hot, 
                                                                                    net.n_classes, average = False)
    
    # Checking if background class is included
    metrics_classes_list = classes_list
    if(args.remove_background):
        metrics_classes_list = classes_list[1:]

    # Create metrics per class
    data_per_class = {
        'Class name':         metrics_classes_list,
        'Pixel accuracy (%)': np.round(pixel_acc * 100, decimals = 2),
        'Dice coeff. (%)':    np.round(dice * 100, decimals = 2),
        'Precision (%)':      np.round(precision * 100, decimals = 2),
        'Recall (%)':         np.round(recall * 100, decimals = 2)
    }
    
    # Calculate average metrics 
    avg_data = {
        'Class name':         "Average",
        'Pixel accuracy (%)': [round(avg_pixel_acc * 100, 2)],
        'Dice coeff. (%)':    [round(avg_dice * 100, 2)],
        'Precision (%)':      [round(avg_precision * 100, 2)],
        'Recall (%)':         [round(avg_recall * 100, 2)]
        }

    # Saving metrics .csv
    csv_file_name   = output_metrics_folder + input_file_id + ".csv"
    logging.info(f'Creating .csv of metrics at : {csv_file_name}')

    # Creating Title row
    df_title        = pd.DataFrame(columns = ["Image ID:", input_file_id, "Type:", input_dataset_str])
    empty_df        = pd.DataFrame([[np.nan] * len(df_title.columns)], columns = df_title.columns)
    df_final        = pd.concat([df_title, empty_df], ignore_index = True)
    df_final.to_csv(csv_file_name, index = False)

    # Filling with Data
    df1        = pd.DataFrame(data_per_class)
    df2        = pd.DataFrame(avg_data)
    empty_df   = pd.DataFrame([[np.nan] * len(df1.columns)], columns = df1.columns)
    df_final   = pd.concat([df1, df2, empty_df], ignore_index = True)
    df_final.to_csv(csv_file_name, index = False, mode = "a")

