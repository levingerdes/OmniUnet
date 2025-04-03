import os,pathlib
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join
import thermal_aligner
import numpy as np
from PIL import Image

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

dir_checkpoint    = pathlib.Path('./checkpoints/')

dir_RGBDT_img     = pathlib.Path('./data/PROCESSED-RGBDT_FEB2023/imgs/')
dir_RGBDT_mask    = pathlib.Path('./data/PROCESSED-RGBDT_FEB2023/mask_one_class_less/')
dir_RGBDT_depth   = pathlib.Path('./data/PROCESSED-RGBDT_FEB2023/depths')
dir_RGBDT_thermal = pathlib.Path('./data/PROCESSED-RGBDT_FEB2023/raw_thermal')
dir_RUGD_img      = pathlib.Path('./data/rugd_images')
dir_RUGD_mask     = pathlib.Path('./data/rugd_masks')

# Saving Dataset
dir_RGBDT_img_save     = pathlib.Path('./data/PROCESSED-RGBDT_2024/imgs/')
dir_RGBDT_mask_save    = pathlib.Path('./data/PROCESSED-RGBDT_2024/masks/')
dir_RGBDT_depth_save   = pathlib.Path('./data/PROCESSED-RGBDT_2024/depths/')
dir_RGBDT_thermal_save = pathlib.Path('./data/PROCESSED-RGBDT_2024/thermals/')

# Alignment cropping limits
left_aligned_limit  = 200
right_aligned_limit = 1100

def loadCsv(csv_file, height, width):
    file_array  = np.genfromtxt(csv_file, delimiter = ',')
    csv_matrix  = np.reshape(file_array, (height, width))
    
    return csv_matrix

def create_folder(folder):
    folder_exits = os.path.exists(folder)
    if not folder_exits:
       os.makedirs(folder)

if __name__ == '__main__':
    rgb_files       = [os.path.join(dir_RGBDT_img, f) for f in sorted(listdir(dir_RGBDT_img)) if isfile(join(dir_RGBDT_img, f))]
    depth_files     = [os.path.join(dir_RGBDT_depth, f) for f in sorted(listdir(dir_RGBDT_depth)) if isfile(join(dir_RGBDT_depth, f))]
    thermal_files   = [os.path.join(dir_RGBDT_thermal, f) for f in sorted(listdir(dir_RGBDT_thermal)) if isfile(join(dir_RGBDT_thermal, f))]
    mask_true_files = [os.path.join(dir_RGBDT_mask, f) for f in sorted(listdir(dir_RGBDT_mask)) if isfile(join(dir_RGBDT_mask, f))]

    # Defining camera transforms to be used in functions
    depth_camera    = thermal_aligner.CameraTransform(thermal_aligner.CameraTransform.TYPE_REALSENSE, width = 1280, height = 720, physical_height = 0.40)
    thermal_camera  = thermal_aligner.CameraTransform(thermal_aligner.CameraTransform.TYPE_THERMAL,   width = 640,  height = 480, physical_height = 0.40)

    N_images = len(mask_true_files)

    for i in range(N_images):
        print("Saving image n: " + str(i))

        merged_image = thermal_aligner.MergedImage()
        
        # Reading original files 
        file_name    = os.path.splitext(os.path.basename(rgb_files[i]))[0]
        rgb_image    = np.array(Image.open(rgb_files[i]))
        mask_image   = np.array(Image.open(mask_true_files[i]))
        depth_csv    = loadCsv(depth_files[i], 720, 1280)
        thermal_csv  = loadCsv(thermal_files[i], 480, 640)

        # Thermal alignment
        aligned_thermal = merged_image.alignThermalToDepth(depth_camera, thermal_camera, depth_csv, thermal_csv)

        # Cropping images to adapt to alignments
        cropped_thermal = aligned_thermal[:, left_aligned_limit:right_aligned_limit]
        cropped_depth   = depth_csv[:, left_aligned_limit:right_aligned_limit]
        cropped_img     = rgb_image[:, left_aligned_limit:right_aligned_limit]
        cropped_mask    = mask_image[:, left_aligned_limit:right_aligned_limit]

        # Creating base savedirs
        create_folder(dir_RGBDT_img_save)
        create_folder(dir_RGBDT_mask_save)
        create_folder(dir_RGBDT_depth_save)
        create_folder(dir_RGBDT_thermal_save)

        # Defining savedirs for new images
        rgb_savename     = str(dir_RGBDT_img_save) + "/" + file_name + ".png"
        mask_savename    = str(dir_RGBDT_mask_save) +  "/" + file_name + "_mask.png"
        depth_savename   = str(dir_RGBDT_depth_save) + "/" +  file_name + "d.csv"
        thermal_savename = str(dir_RGBDT_thermal_save) +  "/" +  file_name + "t.csv"

        # Saving Data
        cropped_img = Image.fromarray(cropped_img)
        cropped_img.save(rgb_savename)
        cropped_mask = Image.fromarray(cropped_mask)
        cropped_mask.save(mask_savename)
        np.savetxt(depth_savename, cropped_depth, delimiter=',', fmt='%0.2f')
        np.savetxt(thermal_savename, cropped_thermal, delimiter=',', fmt='%0.2f')

        
