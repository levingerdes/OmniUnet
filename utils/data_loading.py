import logging, os, sys, pathlib, torch
import numpy as np
import torchvision.transforms as T
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import datetime

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

dir_checkpoint    = pathlib.Path('./checkpoints/')

dir_RGBDT_img     = pathlib.Path('./data/bardenas_seg/imgs/')
dir_RGBDT_mask    = pathlib.Path('./data/bardenas_seg/masks/')
dir_RGBDT_depth   = pathlib.Path('./data/bardenas_seg/depths')
dir_RGBDT_thermal = pathlib.Path('./data/bardenas_seg/thermals')
dir_RUGD_img      = pathlib.Path('./data/rugd_images')
dir_RUGD_mask     = pathlib.Path('./data/rugd_masks')

CLAMP_MAX_THERMAL = 50.0
CLAMP_MIN_THERMAL = 0.0
CLAMP_MIN_DEPTH   = 0.0
CLAMP_MAX_DEPTH   = 10.0

def create_dataset(dataset_str):
    
    dataset_str     = dataset_str.upper()

    if dataset_str == 'RGB':
        dataset = BasicDataset(dir_RGBDT_img, dir_RGBDT_mask)

    elif dataset_str == 'RGBD':
        dataset = RgbdDataset(dir_RGBDT_img, dir_RGBDT_mask, dir_RGBDT_depth)

    elif dataset_str == 'RGBT':
        dataset = RgbtDataset(dir_RGBDT_img, dir_RGBDT_mask, dir_RGBDT_thermal)

    elif dataset_str == 'RGBDT':
        dataset = RgbdtDataset(dir_RGBDT_img, dir_RGBDT_mask, dir_RGBDT_depth, dir_RGBDT_thermal)
    
    elif dataset_str == 'D':
        dataset = DDataset(dir_RGBDT_depth, dir_RGBDT_mask)
    
    elif dataset_str == 'T':
        dataset = TDataset(dir_RGBDT_thermal, dir_RGBDT_mask)
    
    elif dataset_str == 'RUGD':
        dataset = BasicDataset(dir_RUGD_img, dir_RUGD_mask)

    else:
        logging.error("Invalid dataset type specified with argument --dataset")
        sys.exit()

    current_time       = datetime.datetime.now()
    dir_checkpoint_str = "./checkpoints/" + dataset_str + "_" + current_time.strftime('%Y-%m-%d_%H-%M-%S')
    dir_checkpoint     = pathlib.Path(dir_checkpoint_str)
    masks_pred_size    = dataset.pred_size
    
    str_info_log       = "[" + dataset_str + f"] Created dataset with {len(dataset)} examples"
    logging.info(str_info_log)
    str_info_log       = "[" + dataset_str + f"] Mask prediction size will be {masks_pred_size}"
    logging.info(str_info_log)

    return dataset, masks_pred_size, dir_checkpoint


def preprocess_img(pil_img, scale, is_mask):
    w, h       = pil_img.size
    newW, newH = int(scale * w), int(scale * h)
    
    assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
   
    pil_img     = pil_img.resize((newW, newH), resample = Image.NEAREST if is_mask else Image.BICUBIC)
    img_ndarray = np.asarray(pil_img)

    if not is_mask:
        if img_ndarray.ndim == 2:
            img_ndarray = img_ndarray[np.newaxis, ...]
        else:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        img_ndarray = img_ndarray / 255

    return img_ndarray


def load_img(filename): 
    ext = os.path.splitext(filename)[1]
    if ext in ['.npz', '.npy']:
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    elif ext in ['.csv', '.txt']:
        return Image.fromarray(np.genfromtxt(filename, delimiter = ','))
    else:
        return Image.open(filename)
    
def load_multi_matrix(multi_matrix, img_type):
    if(img_type == "rgb"):
        output_image = Image.fromarray(multi_matrix, "RGB")
    elif (img_type == "d"):
        output_image = Image.fromarray(multi_matrix, "F")
    elif (img_type == "t"):
        output_image= Image.fromarray(multi_matrix, "F")
    return output_image

class BasicDataset():
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = '_mask'):
        self.images_dir = pathlib.Path(images_dir)
        self.masks_dir  = pathlib.Path(masks_dir)
        
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        
        self.scale       = scale
        self.mask_suffix = mask_suffix
        self.ids = [os.path.splitext(file)[0] for file in os.listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        # Getting mask_pred_size according to input image size
        example_img_path    = list(self.images_dir.glob('*.*'))[0]      
        example_img         = load_img(example_img_path)
        img_w, img_h        = example_img.size
        self.pred_size      = (img_h, img_w)

    def __len__(self):
        return len(self.ids)

    def get_indexed_filename(self, idx):
        return self.ids[idx]

    def __getitem__(self, idx):
        name      = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file  = list(self.images_dir.glob(name + '.*'))

        assert len(img_file)  == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        
        mask = load_img(mask_file[0])
        img  = load_img(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img  = preprocess_img(img,  self.scale, is_mask = False)
        mask = preprocess_img(mask, self.scale, is_mask = True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask':  torch.as_tensor(mask.copy()).long().contiguous(),
            'image_id': name,
            'input_channels': 3
        }


class RgbdDataset():
    def __init__(self, images_dir: str, masks_dir: str, depth_dir: str, scale: float = 1.0, mask_suffix: str = '_mask'):
        self.images_dir  = pathlib.Path(images_dir)
        self.depth_dir   = pathlib.Path(depth_dir)
        self.masks_dir   = pathlib.Path(masks_dir)
        self.mask_suffix = mask_suffix
        
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        
        self.scale = scale
        self.ids   = [os.path.splitext(file)[0] for file in os.listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        # Getting mask_pred_size according to input image size
        example_img_path    = list(self.images_dir.glob('*.*'))[0]      
        example_img         = load_img(example_img_path)
        img_w, img_h        = example_img.size
        self.pred_size      = (img_h, img_w)

    def __len__(self):
        return len(self.ids)
    
    def get_indexed_filename(self, idx):
        return self.ids[idx]

    @staticmethod
    def build_rgbd(img,dpt):
        rgbd_transform = T.Compose([DepthNorm(max_depth = CLAMP_MAX_DEPTH, min_depth = CLAMP_MIN_DEPTH)])
        img = T.ToTensor()(img)
        dpt = T.ToTensor()(dpt) # In meters

        rgbd = torch.cat([img, dpt], dim=0)
        rgbd = rgbd_transform(rgbd)

        rgbd_input = rgbd[None, :, None, ...]
        
        return rgbd_input

    def __getitem__(self, idx):
        name       = self.ids[idx]
        mask_file  = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file   = list(self.images_dir.glob(name + '.*'))
        depth_file = list(self.depth_dir.glob(name + 'd.*'))

        assert len(img_file)   == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file)  == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(depth_file) == 1, f'Either no depth or multiple depths found for the ID {name}: {depth_file}'

        img   = load_img(img_file[0])
        mask  = load_img(mask_file[0])
        depth = load_img(depth_file[0])

        assert img.size == mask.size and img.size == depth.size, \
            f'Image, mask and depth {name} should be the same size, but are {img.size}, {mask.size} and {depth.size}'

        rgbd = self.build_rgbd(img, depth)
        rgbd = rgbd.squeeze()
        mask = preprocess_img(mask, self.scale, is_mask=True)

        return {
            'image': rgbd.float().contiguous(),
            'mask':  torch.as_tensor(mask.copy()).long().contiguous(),
            'image_id': name,
            'input_channels': 4
        }

class RgbtDataset():
    def __init__(self, images_dir: str, masks_dir: str, thermal_dir: str, scale: float = 1.0, mask_suffix: str = '_mask'):
        self.images_dir  = pathlib.Path(images_dir)
        self.thermal_dir = pathlib.Path(thermal_dir)
        self.masks_dir   = pathlib.Path(masks_dir)
        
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.ids = [os.path.splitext(file)[0] for file in os.listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        # Getting mask_pred_size according to input image size
        example_img_path    = list(self.images_dir.glob('*.*'))[0]      
        example_img         = load_img(example_img_path)
        img_w, img_h        = example_img.size
        self.pred_size      = (img_h, img_w)        
        

    @staticmethod
    def build_rgbt(img,tmp):
        rgbt_transform = T.Compose([ThermalNorm(max_thermal = CLAMP_MAX_THERMAL, min_thermal = CLAMP_MIN_THERMAL)])
        
        img  = T.ToTensor()(img)
        tmp  = T.ToTensor()(tmp) # In Celsius
        rgbt = torch.cat([img, tmp], dim = 0)
        rgbt = rgbt_transform(rgbt)

        # The model expects inputs of shape: B x C x T x H x W
        rgbt_input = rgbt[None, :, None, ...]
        
        return rgbt_input

    def __len__(self):
        return len(self.ids)
    
    def get_indexed_filename(self, idx):
        return self.ids[idx]

    def __getitem__(self, idx):
        name         = self.ids[idx]
        mask_file    = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file     = list(self.images_dir.glob(name + '.*'))
        thermal_file = list(self.thermal_dir.glob(name + 't.*'))

        assert len(img_file)     == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file)    == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(thermal_file) == 1, f'Either no thermal or multiple thermals found for the ID {name}: {thermal_file}'

        img     = load_img(img_file[0])
        mask    = load_img(mask_file[0])
        thermal = load_img(thermal_file[0])

        assert img.size == mask.size and img.size == thermal.size, \
            f'Image, mask and thermal {name} should be the same size, but are {img.size}, {mask.size} and {thermal.size}'

        rgbt = self.build_rgbt(img, thermal)
        rgbt = rgbt.squeeze()
        mask = preprocess_img(mask, self.scale, is_mask = True)

        return {
            'image': rgbt.float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous(),
            'image_id': name,
            'input_channels': 4
        }

class RgbdtDataset():
    def __init__(self, images_dir: str, masks_dir: str, depth_dir: str, thermal_dir: str, scale: float = 1.0, mask_suffix: str = '_mask'):
        self.images_dir  = pathlib.Path(images_dir)
        self.depth_dir   = pathlib.Path(depth_dir)
        self.thermal_dir = pathlib.Path(thermal_dir)
        self.masks_dir   = pathlib.Path(masks_dir)
        self.mask_suffix = mask_suffix
        
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.scale = scale
        self.ids   = [os.path.splitext(file)[0] for file in os.listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        # Getting mask_pred_size according to input image size
        example_img_path    = list(self.images_dir.glob('*.*'))[0]      
        example_img         = load_img(example_img_path)
        img_w, img_h        = example_img.size
        self.pred_size      = (img_h, img_w)   
        
    @staticmethod
    def build_rgbdt(img,dpt,tmp):
        #  8 meters as max. Bigger values are not relevant
        rgbd_transform  = T.Compose([DepthNorm(max_depth = 10.0, min_depth = 0.0)])
                        
        # Temperature max and min values of the dataset
        rgbdt_transform = T.Compose([ThermalNorm(max_thermal = CLAMP_MAX_THERMAL, min_thermal = CLAMP_MIN_THERMAL)])
        
        img   = T.ToTensor()(img)
        dpt   = T.ToTensor()(dpt) # In meters
        tmp   = T.ToTensor()(tmp) # In Celsius
        rgbd  = torch.cat([img, dpt], dim = 0)
        rgbd  = rgbd_transform(rgbd)
        rgbdt = torch.cat([rgbd, tmp], dim = 0)
        rgbdt = rgbdt_transform(rgbdt)

        # The model expects inputs of shape: B x C x T x H x W
        rgbdt_input = rgbdt[None, :, None, ...]
        
        return rgbdt_input

    def __len__(self):
        return len(self.ids)
    
    def get_indexed_filename(self, idx):
        return self.ids[idx]

    def __getitem__(self, idx):
        name         = self.ids[idx]
        mask_file    = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file     = list(self.images_dir.glob(name + '.*'))
        depth_file   = list(self.depth_dir.glob(name + 'd.*'))
        thermal_file = list(self.thermal_dir.glob(name + 't.*'))

        assert len(img_file)     == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file)    == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(depth_file)   == 1, f'Either no depth or multiple depths found for the ID {name}: {depth_file}'
        assert len(thermal_file) == 1, f'Either no thermal or multiple thermals found for the ID {name}: {thermal_file}'

        img     = load_img(img_file[0])
        mask    = load_img(mask_file[0])
        depth   = load_img(depth_file[0])
        thermal = load_img(thermal_file[0])

        assert img.size == mask.size and img.size == depth.size and img.size == thermal.size, \
            f'Image, mask, depth and thermal {name} should be the same size, but are {img.size}, {mask.size}, {depth.size} and {thermal.size}'

        rgbdt = self.build_rgbdt(img, depth, thermal)
        rgbdt = rgbdt.squeeze()
        mask  = preprocess_img(mask, self.scale, is_mask = True)

        return {
            'image':    rgbdt.float().contiguous(),
            'mask':     torch.as_tensor(mask.copy()).long().contiguous(),
            'image_id': name,
            'input_channels': 5
        }
    

class TDataset(BasicDataset):
    def __init__(self, thermal_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = '_mask'):
        self.thermal_dir = pathlib.Path(thermal_dir)
        self.masks_dir   = pathlib.Path(masks_dir)

        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.ids = [os.path.splitext(file)[0] for file in os.listdir(thermal_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {thermal_dir}, make sure you put your images there')

        # Getting mask_pred_size according to input image size
        example_img_path    = list(self.thermal_dir.glob('*.*'))[0]      
        example_img         = load_img(example_img_path)
        img_w, img_h        = example_img.size
        self.pred_size      = (img_h, img_w)   

    @staticmethod
    def build_t(thermal):
        t_transform =  T.Compose([ThermalNorm(max_thermal = CLAMP_MAX_THERMAL, min_thermal = CLAMP_MIN_THERMAL)])
        
        thermal = T.ToTensor()(thermal) # in Celsius

        t = torch.cat([thermal], dim=0)
        t = t_transform(t)

        # The model expects inputs of shape: B x C x T x H x W
        t_input = t[None, :, None, ...]
        
        return t_input
    
    def __len__(self):
        return len(self.ids)
    
    def get_indexed_filename(self, idx):
        return self.ids[idx]

    def __getitem__(self, idx):
        name = self.ids[idx]
        name = name[:-1] # Remove last character (t) to make it similar to RGB datalaoding
        mask_file    = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        thermal_file = list(self.thermal_dir.glob(name + 't.*'))

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(thermal_file) == 1, f'Either no depth or multiple depths found for the ID {name}: {thermal_file}'
        
        mask    = load_img(mask_file[0])
        thermal = load_img(thermal_file[0])

        assert mask.size == thermal.size, \
        f'Mask and thermal {name} should be the same size, but are {thermal.size}, {mask.size}'

        t = self.build_t(thermal)
        t = t.squeeze()
        t = t.unsqueeze(0)
        mask = preprocess_img(mask, self.scale, is_mask = True)

        return {
            'image': t.float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous(),
            'image_id': name,
            'input_channels': 1
        }
    

class DDataset(BasicDataset):
    def __init__(self, depth_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = '_mask'):
        self.depth_dir = pathlib.Path(depth_dir)
        self.masks_dir   = pathlib.Path(masks_dir)

        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.ids = [os.path.splitext(file)[0] for file in os.listdir(depth_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {depth_dir}, make sure you put your images there')

        # Getting mask_pred_size according to input image size
        example_img_path    = list(self.depth_dir.glob('*.*'))[0]      
        example_img         = load_img(example_img_path)
        img_w, img_h        = example_img.size
        self.pred_size      = (img_h, img_w)   

    @staticmethod
    def build_d(depth):
        d_transform =  T.Compose([DepthNorm(max_depth = CLAMP_MAX_DEPTH, min_depth = CLAMP_MIN_DEPTH)])
        
        depth = T.ToTensor()(depth) # in Celsius

        d = torch.cat([depth], dim=0)
        d = d_transform(d)

        # The model expects inputs of shape: B x C x T x H x W
        d_input = d[None, :, None, ...]
        
        return d_input
    
    def __len__(self):
        return len(self.ids)
    
    def get_indexed_filename(self, idx):
        return self.ids[idx]

    def __getitem__(self, idx):
        name = self.ids[idx]
        name = name[:-1] # Remove last character (d) to make it similar to RGB datalaoding
        mask_file    = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        depth_file   = list(self.depth_dir.glob(name + 'd.*'))

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(depth_file) == 1, f'Either no depth or multiple depths found for the ID {name}: {depth_file}'
        
        mask    = load_img(mask_file[0])
        depth   = load_img(depth_file[0])

        assert mask.size == depth.size, \
        f'Mask and thermal {name} should be the same size, but are {depth.size}, {mask.size}'

        d = self.build_d(depth)
        d = d.squeeze()
        d = d.unsqueeze(0)
        mask = preprocess_img(mask, self.scale, is_mask = True)

        return {
            'image': d.float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous(),
            'image_id': name,
            'input_channels': 1
        }


class DepthNorm(nn.Module):
    """
    Normalize the depth channel
    """
    def __init__(self, max_depth: float, min_depth: float = 0.01):
        super().__init__()
        if max_depth < 0.0:
            raise ValueError("max_depth must be > 0; got %.2f" % max_depth)
        
        self.max_depth = max_depth
        self.min_depth = min_depth

    def forward(self, image: torch.Tensor):
        C, H, W = image.shape
        
        if C > 2:
            color_img   = image[:C-1, ...]  # (3, H, W)
            depth_img   = image[C-1:C, ...] # (1, H, W)
        elif C == 2:
            thermal_img = image[:C-1, ...]
            depth_img   = image[C-1:C, ...]
        else:
            depth_img   = image

        depth_img = depth_img.clamp(min = self.min_depth)
        depth_img = depth_img.clamp(max = self.max_depth)
        max_value = torch.max(depth_img)
        min_value = torch.min(depth_img)
        depth_img = (depth_img - min_value) / (max_value - min_value)
        
        if C > 2:
            img = torch.cat([color_img, depth_img],   dim = 0)
        elif C == 2:
            img = torch.cat([depth_img, thermal_img], dim = 0)
        else:
            img = depth_img

        return img


class ThermalNorm(nn.Module):
    """
    Normalize the thermal channel
    """
    def __init__(self, max_thermal: float, min_thermal: float = 0.01):
        super().__init__()
        if max_thermal < 0.0:
            raise ValueError("max_thermal must be > 0; got %.2f" % max_thermal)

        self.max_thermal = max_thermal
        self.min_thermal = min_thermal

    def forward(self, image: torch.Tensor):
        C, H, W = image.shape
        
        if C > 1:
            rgbd_img    = image[:C-1, ...]   # (C-1, H, W)
            thermal_img = image[C-1:C, ...]  # (1, H, W)
        else:
            thermal_img = image

        thermal_img = thermal_img.clamp(min = self.min_thermal)
        thermal_img = thermal_img.clamp(max = self.max_thermal)
        max_value   = torch.max(thermal_img)
        min_value   = torch.min(thermal_img)
        thermal_img = (thermal_img - min_value) / (max_value - min_value)

        if C > 1:
            img = torch.cat([rgbd_img, thermal_img], dim = 0)
        else:
            img = thermal_img

        return img