import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.patches as mpatches
import PIL
import random

import matplotlib
# matplotlib.use("TkAgg")

def parse_class_file(class_dir):
    """
    parse_class_file receives class.txt path and returns and 
    indexed list with the names of the classes.

    :param class_dir: class.txt path

    :return: indexed list with the names of the classes
    """ 
    class_number_list, class_name_list, class_color_list = [], [], []

    class_file = open(class_dir)
    for line in class_file:
        stripped_line = line.split(' ')
        
        # List of class numbers 
        class_number  = int(stripped_line[0])
        class_number_list.append(class_number)
        
        # List of class names 
        class_name    = stripped_line[1]
        class_name_list.append(class_name)

        # List of normalized colors 
        class_color      =  (int(stripped_line[2]), int(stripped_line[3]), int(stripped_line[4]))
        norm_class_color =  [color / 255 for color in class_color]
        class_color_list.append(norm_class_color)

    class_file.close()

    # Creating array with index class names
    max_classes  = max(class_number_list) + 1
    classes_list = [0] * max_classes
    for i in range(len(class_number_list)):
        class_number = class_number_list[i]
        class_name   = class_name_list[i]       
        classes_list[class_number] = class_name
    
    return classes_list, class_color_list


def parse_class_file_cost(class_dir):
    costmap_list = []

    class_file = open(class_dir)
    for line in class_file:
        stripped_line = line.split(' ')
        
        # List of class numbers 
        class_number  = int(stripped_line[5])
        costmap_list.append(class_number)

    class_file.close()

    return costmap_list


def plot_mask_and_legend(input_file_id, mask, class_list, class_color_list):
    """
    plot_mask_and_legend plots a mask with its legend of classes.

    :param input_file_id: name of the predicted file
    :param mask: predicted mask by the network
    :param class_list: indexed list with the names of the classes

    :return: void. Saves the plot and shows it.
    """
    # Creating legend fig
    fig = plt.figure(figsize = (6, 4.5), dpi = 300)
    ax  = fig.add_subplot()

    # Removing axis and creating frame
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth( 1 )  
        ax.spines[axis].set_color('black')        
        # Turn off tick labels
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    plt.tick_params(left = False, bottom = False) 

    # Other setmaps are terrain_r and gist_earth_r
    # plt.set_cmap('gist_earth_r')
    
    # Creating adapted cmaps from color list
    n_classes           = len(class_color_list) - 1
    color_classes_cmap  = ListedColormap(class_color_list)
    im                  = plt.imshow(mask, cmap = color_classes_cmap, interpolation = 'none', vmin = 0, vmax = n_classes)

    # Get values from image
    values = np.unique(mask.ravel()).astype(int)
    # Creating legend labels
    j = 0
    patches_list = []
    colors = [im.cmap(im.norm(value)) for value in values]
    for i in values:
        patches_list.append(mpatches.Patch(facecolor = colors[j], edgecolor = "k", label = class_list[i]))
        j = j + 1

    # Put a legend below current axis
    leg = plt.legend(handles = patches_list, bbox_to_anchor = (0.5, -0.01), ncol = 4, loc = 'upper center')
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_linewidth(1.0)
    leg.get_frame().set_boxstyle('Square')

    # Create title with file id name
    title_str = "Image ID: " + input_file_id
    plt.title(title_str) 

    # To avoid cutting of the legend
    fig.tight_layout()

    fig.canvas.draw()
    PIL_image = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    ax.clear()
    fig.clear()
    plt.close(fig)
    
    return PIL_image


def plot_mask_comparison(input_file_id, mask_true, mask_pred, class_list, class_color_list):
    """
    plot_mask_comparison plots side by side true mask and prediction mask with its legend of classes.

    :param input_file_id: name of the predicted file
    :param mask: predicted mask by the network
    :param class_list: indexed list with the names of the classes

    :return: void. Saves the plot and shows it.
    """
    # Creating legend fig
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 5), dpi = 300)

    axis_list = []
    axis_list.append(ax1)
    axis_list.append(ax2)
    
    # Creating adapted cmaps from color list
    n_classes           = len(class_color_list) - 1
    color_classes_cmap  = ListedColormap(class_color_list)

    im1  = ax1.imshow(mask_true, cmap = color_classes_cmap, interpolation = 'none', vmin = 0, vmax = n_classes)
    im2  = ax2.imshow(mask_pred, cmap = color_classes_cmap, interpolation = 'none', vmin = 0, vmax = n_classes)
    ax1.set_title("Groundtruth mask")
    ax2.set_title("Prediction mask")

    # Removing axis and creating frame
    for i in range(2):
        for axis in ['top', 'bottom', 'left', 'right']:
            axis_list[i].spines[axis].set_linewidth( 1 )  
            axis_list[i].spines[axis].set_color('black')        
            # Turn off tick labels
            axis_list[i].set_yticklabels([])
            axis_list[i].set_xticklabels([])
            axis_list[i].tick_params(left = False, bottom = False) 

    # Choosing with values to plot in the legend
    true_mask_matrix = np.asarray(mask_true)
    pred_mask_matrix = np.asarray(mask_pred)
    values_true = np.unique(true_mask_matrix.ravel()).astype(int)
    values_pred = np.unique(pred_mask_matrix.ravel()).astype(int)

    if(len(values_true) > len(values_pred)):
        im     = im1
        values = values_true
    else:
        im     = im2
        values = values_pred
    
    # Creating legend labels
    j = 0
    patches_list = []
    colors = [im.cmap(im.norm(value)) for value in values]
    for i in values:
        patches_list.append(mpatches.Patch(facecolor = colors[j], edgecolor = "k", label = class_list[i]))
        j = j + 1

    # Create common legend
    leg = fig.legend(handles = patches_list, bbox_to_anchor = (0.5, 0), ncol = 4, loc = 'lower center')
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_linewidth(1.0)
    leg.get_frame().set_boxstyle('Square')

    # Create title with file id name
    title_str = "Image ID: " + input_file_id
    fig.suptitle(title_str, fontsize = 16) 

    # To avoid cutting of the legend
    fig.tight_layout()

    fig.canvas.draw()
    PIL_image = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    ax1.clear()
    ax2.clear()
    fig.clear()
    plt.close(fig)

    return PIL_image


def plot_mask_comparison_with_input(input_file_id, mask_true, mask_pred, input_rgb, input_thermal, class_list, class_color_list):
    """
    plot_mask_comparison plots side by side true mask and prediction mask with its legend of classes.

    :param input_file_id: name of the predicted file
    :param mask: predicted mask by the network
    :param class_list: indexed list with the names of the classes

    :return: void. Saves the plot and shows it.
    """
    # Creating legend fig
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (12, 10), dpi = 300)

    axis_list = []
    axis_list.append(ax1)
    axis_list.append(ax2)
    axis_list.append(ax3)
    axis_list.append(ax4)

    # Creating adapted cmaps from color list
    n_classes            = len(class_color_list) - 1
    color_classes_cmap   = ListedColormap(class_color_list)

    # Show images with adapted cmaps
    im1  = ax1.imshow(mask_true, cmap = color_classes_cmap, interpolation = 'none', vmin = 0, vmax = n_classes)
    im2  = ax2.imshow(mask_pred, cmap = color_classes_cmap, interpolation = 'none', vmin = 0, vmax = n_classes)
    im3  = ax3.imshow(input_rgb, interpolation = 'none')
    im4  = ax4.imshow(input_thermal, cmap = 'plasma', interpolation = 'none', vmin = 0)
    ax1.set_title("Groundtruth mask")
    ax2.set_title("Prediction mask")
    ax3.set_title("Input RGB")
    ax4.set_title("Input thermal")

    # Removing axis and creating frame
    for i in range(4):
        for axis in ['top', 'bottom', 'left', 'right']:
            axis_list[i].spines[axis].set_linewidth( 1 )  
            axis_list[i].spines[axis].set_color('black')        
            # Turn off tick labels
            axis_list[i].set_yticklabels([])
            axis_list[i].set_xticklabels([])
            axis_list[i].tick_params(left = False, bottom = False) 


    # Choosing with values to plot in the legend
    true_mask_matrix = np.asarray(mask_true)
    pred_mask_matrix = np.asarray(mask_pred)
    values_true = np.unique(true_mask_matrix.ravel()).astype(int)
    values_pred = np.unique(pred_mask_matrix.ravel()).astype(int)

    if(len(values_true) > len(values_pred)):
        im     = im1
        values = values_true
    else:
        im     = im2
        values = values_pred
    
    # Creating legend labels
    j = 0
    patches_list = []
    colors = [im.cmap(im.norm(value)) for value in values]
    for i in values:
        patches_list.append(mpatches.Patch(facecolor = colors[j], edgecolor = "k", label = class_list[i]))
        j = j + 1

    # Create common legend
    leg = fig.legend(handles = patches_list, bbox_to_anchor = (0.5, 0), ncol = 4, loc = 'lower center')
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_linewidth(1.0)
    leg.get_frame().set_boxstyle('Square')

    # Create title with file id name
    title_str = "Image ID: " + input_file_id
    fig.suptitle(title_str, fontsize = 16) 

    # To avoid cutting of the legend
    fig.tight_layout()

    fig.canvas.draw()
    PIL_image = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()
    fig.clear()
    plt.close(fig)

    return PIL_image


def extract_img_id(file_dir):
    """
    extract_img_id receives img path and returns basename without extension.

    :param file_dir: img path

    :return: str basename
    """ 
    basename = os.path.basename(file_dir)
    basename = os.path.splitext(basename)[0]

    return basename


def create_folder(folder):
    folder_exits = os.path.exists(folder)
    if not folder_exits:
       os.makedirs(folder)


def loadCsv(csv_file, height, width):
    file_array  = np.genfromtxt(csv_file, delimiter = ',')
    csv_matrix  = np.reshape(file_array, (height, width))
    
    return csv_matrix

def create_list_txt(filename, image_list):
    with open(filename, 'w') as f:
        for line in image_list:
            f.write(line + '\n')

# Search for the matching substrings IDs between txt an list
# saving the list values
def matching_id(file_name, imgs_list):
    matched_filenames = []
    with open(file_name, 'r') as file:
        for line in file:
            number = line.strip()
            matching_elements = [element for element in imgs_list if number in element]
            matched_filenames.extend(matching_elements)

    return matched_filenames


def missing_modality_selector(images):
    # We consider that the dataset introduced is RGBDT
    choice = random.randint(0, 2)

    # RGDT case
    if choice == 0:
        missing_case = "RGBDT"
        images = images

    elif choice == 1:
        missing_case = "RGB"
        images = images[:,:3,:,:]

    else:
        missing_case = "T"
        images = images[:,4,:,:].unsqueeze(1)
    
    return images, missing_case