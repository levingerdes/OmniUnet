import argparse, logging, os, sys, torch
import torch.nn.functional as F
import utils.data_loading as dataloading
import utils.img_handler as img_handler
import warnings
warnings.filterwarnings("ignore") 
from model.omniunet_model import OmniUnet
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
dir_class   = os.path.join(THIS_FOLDER, 'classes/LAENTIEC')

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model',         '-m', default = 'MODEL.pth', metavar = 'FILE', help = 'Specify the file in which the model is stored')
    parser.add_argument('--rgb_input',     '-i', type = str,   default = '',    help = 'Filenames of input images')
    parser.add_argument('--scale',         '-s', type = float, default = 0.5,   help = 'Scale factor for the input images')
    parser.add_argument('--bilinear',            type = bool,  default = False, help = 'Use bilinear upsampling')
    parser.add_argument('--classes',       '-c', type = int,   default = 2,     help = 'Number of classes')
    parser.add_argument('--dataset',       '-d', type = str,   default = '',    help = 'Dataset name')
    parser.add_argument('--depth_input',   '-p', type = str,   default = '',    help = 'Depth filename')
    parser.add_argument('--thermal_input', '-T', type = str,   default = '',    help = 'Thermal filename')

    return parser.parse_args()


if __name__ == '__main__':
    
    args      = get_args()
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.basicConfig(level = logging.INFO, format = '%(levelname)s: %(message)s')

    # Defining dataset type
    depthmode         = "summed_tokens"
    input_dataset_str = args.dataset.upper()
    
    if input_dataset_str   == 'RGBDT':
        channels      = 5
        input_rgb     = dataloading.load_img(args.rgb_input)
        input_depth   = dataloading.load_img(args.depth_input)
        input_thermal = dataloading.load_img(args.thermal_input)
        input_img     = dataloading.RgbdtDataset.build_rgbdt(input_rgb, input_depth, input_thermal)
    
    elif input_dataset_str == 'RGBT':
        channels      = 4
        input_rgb     = dataloading.load_img(args.rgb_input)
        input_thermal = dataloading.load_img(args.thermal_input)
        input_img     = dataloading.RgbtDataset.build_rgbt(input_rgb, input_thermal)
        
    elif input_dataset_str == 'RGBD':
        channels      = 4
        input_rgb     = dataloading.load_img(args.rgb_input)
        input_depth   = dataloading.load_img(args.depth_input)
        input_img     = dataloading.RgbdDataset.build_rgbd(input_rgb, input_depth)

    elif input_dataset_str == 'RGB':
        channels      = 3
        input_rgb     = dataloading.load_img(args.rgb_input)
        processed_img = dataloading.preprocess_img(input_rgb, args.scale_factor, is_mask = False)
        input_img     = torch.from_numpy(processed_img)
        input_img     = input_img.unsqueeze(0)
    
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

        mask_pred = F.interpolate(mask_pred, size = mask_pred_size, mode = 'bicubic', align_corners = False)
        mask_pred = torch.softmax(mask_pred, dim = 1).argmax(dim = 1)[0].float().cpu()
       
        # Saving fig of pred mask
        classes_list, class_color_list  = img_handler.parse_class_file(dir_class)
        input_file_id                   = img_handler.extract_img_id(args.rgb_input)
        legend_mask                     = img_handler.plot_mask_and_legend(input_file_id, mask_pred, classes_list, class_color_list)
        # Saving standalone and legend image
        save_pred_fig_dir     = "./" + input_file_id + "_out.png"
        save_legend_fig_dir   = "./" + input_file_id + "_legend_out.png"
        cmap_color_list       = ListedColormap(class_color_list)
        plt.imsave(save_pred_fig_dir, mask_pred, cmap = cmap_color_list, origin = 'upper', dpi = 300)
        legend_mask.save(save_legend_fig_dir)

        logging.info(f'Original output image saved at: {save_pred_fig_dir}')
        logging.info(f'Output image with legend saved at: {save_legend_fig_dir}')