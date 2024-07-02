""" This script contains useful functions for data processing. """

import re
import os
import random
import math
from typing import Union, Dict, Any, List, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import binary_erosion, binary_dilation, binary_fill_holes
import skimage
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from lungmask import LMInferer
from pydicom import dcmread
import pydicom

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from ctpalette.paths import *

DEVICE = torch.device("cpu")
MIN_INTENSITY = 0
MAX_INTENSITY = 255
RESIZED_IMG_SIZE = 256

inferer = LMInferer(tqdm_disable=True)
downsample_fn = T.Resize(size=256, antialias=True)
upsample_fn = T.Resize(size=512, antialias=True)

def save_numpy(npy_path: str, npy_data: np.ndarray, overwrite: bool = False):
    """ Saves numpy array as a file. """
    if os.path.exists(npy_path) and not overwrite:
        return
    else:
        np.save(npy_path, npy_data)

def mkdir_p(directory_path: str) -> bool:
    """ Recursively creates a directory and its parent directories if they do not exist. """
    if os.path.exists(directory_path):
        return True
    else:
        os.makedirs(directory_path)
        return False

def list_dirs(path: str) -> List[str]:
    """ Finds all directories inside the given path. """
    return [directory for directory in os.listdir(path) if os.path.isdir(os.path.join(path, directory))]

def list_files(path: str) -> List[str]:
    """ Lists all files inside the given path (excludes those which start with '.'). """
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and not f.startswith(".")]

def binary_erosion_torch(obj: torch.Tensor) -> torch.Tensor:
    """
    Binary erosion similar to scipy.ndimage.binary_erosion(obj, boundary_width=2)
    Args:
        obj: A 2-dimensional image (H, W)
    """
    obj = obj[None, None, :, :].to(torch.float32)
    kernel = torch.tensor([[[[0, 0, 1, 0, 0],
                             [0, 1, 1, 1, 0],
                             [1, 1, 1, 1, 1],
                             [0, 1, 1, 1, 0],
                             [0, 0, 1, 0, 0]]]], dtype=torch.float32, device=DEVICE)
    eroded_obj = F.conv2d(obj, kernel, padding=2, bias=None)
    eroded_obj = (eroded_obj == torch.sum(kernel)).float()
    eroded_obj = eroded_obj.squeeze()
    return eroded_obj

def create_img_boundary(img: torch.Tensor):
    """ Creates a new image which is the boundary of the given image. """
    return img - binary_erosion_torch(img).to(torch.int32)

def get_tci_value(body_mask: torch.Tensor, ppr_mask: torch.Tensor) -> float:
    """ Calculates the Tissue Cropping Index (TCI) based on the given body mask and PPR mask. """
    body_boundary = create_img_boundary(body_mask)
    fov_mask = 1 - ppr_mask
    fov_boundary = create_img_boundary(fov_mask)

    fake_body_boundary_value = 2
    body_boundary[(body_boundary == 1) & (fov_boundary == 1)] = fake_body_boundary_value

    num_total_segment = torch.count_nonzero(body_boundary)
    num_fake_segment = torch.count_nonzero(body_boundary == fake_body_boundary_value)
    tci_val = num_fake_segment / num_total_segment

    return tci_val.item()

def classify_tci_to_severity_level(tci_val: float) -> str:
    """ Classifies the given TCI into its corresponding truncation severity level. """
    if tci_val == 0:
        return "no_trunc"
    elif tci_val > 0 and tci_val <= 0.15:
        return "trace"
    elif tci_val > 0.15 and tci_val <= 0.3:
        return "mild"
    elif tci_val > 0.3 and tci_val <= 0.5:
        return "moderate"
    else:
        return "severe"
    
############################## Image processing ##############################

def transform_dcm_to_HU(dcm: pydicom.dataset.FileDataset) -> np.ndarray:
    """
    Extracts the pixel data contained in a DICOM object and converts it into HU values.
    Args:
        dcm: DICOM object as a result of pydicom.dcmread()
    """
    return dcm.pixel_array * float(dcm.RescaleSlope) + float(dcm.RescaleIntercept)

def transform_to_HU(img: Union[np.ndarray, torch.Tensor]):
    """ Transforms an image with range [-1, 1] to [-160, 240] HU. """
    return 200 * img + 40

def transform_to_RGB(img: Union[np.ndarray, torch.Tensor]):
    """ Transforms an image with range [-1, 1] to [0, 255]. """
    return 127.5 * (img + 1)

def normalize_rgb_img(img: Union[np.ndarray, torch.Tensor]):
    """ Normalizes an image with range [0, 255] to [-1, 1]. """
    return 2 * img / 255 - 1

def downsample_img(img: torch.Tensor, downsample_fn=downsample_fn):
    """
    Downsamples image, by default to size 256x256.
    Args:
        img: A 2-dimensional image (H, W)
    """
    img = downsample_fn(img.expand(1, 1, -1, -1)).squeeze()
    return img

def upsample_img(img: torch.Tensor, upsample_fn=upsample_fn):
    """
    Upsamples image, by default to size 512x512.
     Args:
        img: A 2-dimensional image (H, W)
    """
    img = upsample_fn(img.expand(1, 1, -1, -1)).squeeze()
    return img
    
############################## Candidate slice selection ##############################

def find_neighbors_indices(target_slice_idx: int, total_num_slices: int, num_neighbors: int) -> Tuple[List[int], List[int]]:
    """ Finds num_neighbors left and num_neighbors right neighboring slices of the given target_slice_idx. """
    all_slices_indices = [i for i in range(total_num_slices)]
    cand_left_neighbors = [i for i in range(target_slice_idx - num_neighbors, target_slice_idx)]
    cand_right_neighbors = [i for i in range(target_slice_idx + 1, target_slice_idx + num_neighbors + 1)]
    
    left_neighbors, right_neighbors = [], []
    
    for n in cand_left_neighbors:
        if n in all_slices_indices:
            left_neighbors.append(n)
    for n in cand_right_neighbors:
        if n in all_slices_indices:
            right_neighbors.append(n)
    
    # Case 1: There are no left and right neighbors (len(left_neighbors) == 0 and len(right_neighbors) == 0)
    if total_num_slices == 1:
        return [target_slice_idx] * num_neighbors, [target_slice_idx] * num_neighbors
    
    # Case 2: There are enough left neighbors but no right neighbors
    # In this case, use target_slice_idx as right neighbors
    elif len(left_neighbors) == num_neighbors and len(right_neighbors) == 0:
        for i in range(num_neighbors):
            right_neighbors.append(target_slice_idx)
    
    # Case 3: There are enough right neighbors but no left neighbors
    # In this case, use target_slice_idx as left neighbors
    elif len(right_neighbors) == num_neighbors and len(left_neighbors) == 0:
        for i in range(num_neighbors):
            left_neighbors.append(target_slice_idx)
    
    # Case 4: There are not enough left neighbors or/and right neighbors
    elif len(left_neighbors) < num_neighbors or len(right_neighbors) < num_neighbors:
        if len(left_neighbors) < num_neighbors:
            num_missing_left_neighbors = num_neighbors - len(left_neighbors)
            for i in range(num_missing_left_neighbors):
                left_neighbors.append(target_slice_idx)
        if len(right_neighbors) < num_neighbors:
            num_missing_right_neighbors = num_neighbors - len(right_neighbors)
            for i in range(num_missing_right_neighbors):
                right_neighbors.append(target_slice_idx)
            right_neighbors.sort()
    
    # Case 5: There are enough left and right neighbors
    # Do nothing
    
    target_and_neighbor_indices = left_neighbors + [target_slice_idx] + right_neighbors
    
    assert len(target_and_neighbor_indices) == 2 * num_neighbors + 1
    assert all([i >= 0 and i <= max(all_slices_indices) for i in target_and_neighbor_indices]) == True
    
    return left_neighbors, right_neighbors
    
############################## Slice preprocessing ##############################

def normalize_image_intensity(img: np.ndarray, clip_range: Tuple[int, int], scale_range: Tuple[int, int]) -> np.ndarray:
    """ Clips an image to the range [-150, 150] HU and normalizes it to the range [-1, 1]. """
    img = np.clip(img, clip_range[0], clip_range[1])
    normalizer = interp1d(clip_range, scale_range)
    return normalizer(img)

def resize_mask(mask: np.ndarray, resized_dim: int = RESIZED_IMG_SIZE) -> np.ndarray:
    """ Resizes a binary mask with nearest interpolation, by default 256x256. """
    resized_mask = cv2.resize(mask.astype(int), (resized_dim, resized_dim), interpolation=cv2.INTER_NEAREST)
    return resized_mask

def preprocess_slice(ct_slice: Union[np.ndarray, torch.Tensor], body_mask_slice: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Taken from CCDS_BodyComp_NoFilter_NoDocker.
    Preprocesses CT slice from its original HU values by:
        - clipping it to a window of [-160, 240] HU
        - removing extraneous information using the body mask
        - transforming it to the range [0, 255]
    """
    # 1. Apply windowing ([-160, 240] HU to highlight soft tissue -> Then, linear transform to [0, 255])
    win_centre = 40.
    win_width = 400
    range_bottom = win_centre - win_width / 2
    scale = 256 / float(win_width)
    windowed_ct_slice = ct_slice - range_bottom
    windowed_ct_slice = windowed_ct_slice * scale
    windowed_ct_slice[windowed_ct_slice < MIN_INTENSITY] = MIN_INTENSITY
    windowed_ct_slice[windowed_ct_slice > MAX_INTENSITY] = MAX_INTENSITY

    # 2. Remove extraneous information
    windowed_ct_slice[body_mask_slice == 0] = 0
    
    return windowed_ct_slice

def preprocess_slice_vanderbilt(ct_slice: Union[np.ndarray, torch.Tensor], body_mask_slice: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Taken from the code for S-EFOV.
    Preprocesses CT slice from its original HU values by:
        - clipping it to a window of [-150, 150] HU
        - removing extraneous information using the body mask
        - transforming it to the range [0, 255]
    """
    # 1. Apply intensity window [-150, 150] HU
    ct_slice = np.clip(ct_slice, -150, 150)

    # 2. Rescale [-150, 150] HU to [-1, 1] HU
    ct_slice = normalize_image_intensity(ct_slice, (-150, 150), (-1, 1))

    # 3. Remove extraneous information
    ct_slice[body_mask_slice == 0] = -1

    # 4. Linear transform to [0, 255]
    ct_slice = transform_to_RGB(ct_slice)
    
    return ct_slice

############################## Body mask generation ##############################

def create_body_mask_slice(ct_slice: np.ndarray, cutoff: float = -500, inclusive: bool = True) -> np.ndarray:
    """
    Generates the body mask of the given CT slice.
    Args:
        - ct_slice: The CT slice from which body mask should be generated.
        - cutoff: The value used as threshold to identify the body. By default, -500 HU.
        - inclusive: Whether the cutoff value should be included as the body.
    """
    rBody = 4

    dimension_limit = 800 * 800
    image_shape = ct_slice.shape
    if image_shape[0] * image_shape[1] > dimension_limit:
        raise ValueError('The image dimension exceed the preset limit.')
    
    if inclusive:
        BODY = (ct_slice >= cutoff)# & (I<=win_max)
    else:
        BODY = (ct_slice > cutoff)

    if np.sum(BODY) == 0:
        print("BODY could not be extracted!")
        raise ValueError('BODY could not be extracted!')

    # Find largest connected component in 2D
    struct = np.ones((2, 2), dtype=bool)
    BODY = binary_erosion(BODY, structure=struct, iterations=rBody)

    BODY_labels = skimage.measure.label(np.asarray(BODY, dtype=int))

    props = skimage.measure.regionprops(BODY_labels)
    areas = []
    for prop in props:
        areas.append(prop.area)

    # only keep largest, dilate again and fill holes
    BODY = binary_dilation(BODY_labels == (np.argmax(areas) + 1), structure=struct, iterations=rBody)

    # Fill holes slice-wise
    BODY = binary_fill_holes(BODY)
    BODY = BODY.astype(int)

    return BODY

############################## Lung mask generation ##############################

def create_lung_mask_slice(dcm_img: np.ndarray):
    """
    Generates the lung mask of the given CT slice.
    Args:
        dcm_img: A 2-dimensional CT slice with original HU values (before any windowing applied).
    """
    return inferer.apply(dcm_img[None, :, :])[0]

def create_lung_mask(ct_vol: np.ndarray):
    """
    Generates the lung mask of the given CT volume.
    Args:
        ct_vol: A 3-dimensional CT volume (num_slices, H, W) with original HU values (before any windowing applied).
    """
    lung_mask = inferer.apply(ct_vol)
    lung_mask[lung_mask >= 1] = 1

    return lung_mask

############################## PPR/FOV mask generation ##############################

def create_ppr_mask(ct_vol: np.ndarray, cutoff: float):
    """
    Generates the PPR mask (opposite of FOV mask) of the given CT volume.
    Args:
        - ct_vol: A 3-dimensional CT volume (H, W, num_slices) with original HU values (before any windowing applied).
        - cutoff: The HU value used as threshold to identify the field-of-view.
    """
    ppr_mask_slices = []
    for slice_idx in range(ct_vol.shape[2]):
        ct_slice = ct_vol[:, :, slice_idx]
        ppr_mask_slice = ct_slice.copy()
        ppr_mask_slice[ppr_mask_slice > cutoff] = 0
        ppr_mask_slice[ppr_mask_slice <= cutoff] = 1
        ppr_mask_slices.append(ppr_mask_slice)
    return np.prod(ppr_mask_slices, axis=0)

def create_ppr_mask_v2(ct_vol):
    """
    Generates the PPR mask (opposite of FOV mask) of the given CT volume.
    Based on the code for S-EFOV.
    Args:
        - ct_vol: A 3-dimensional CT volume (H, W, num_slices) with original HU values (before any windowing applied).
        - cutoff: The HU value used as threshold to identify the field-of-view.
    """
    z_variance_map = np.var(ct_vol, axis=2)
    ppr_mask = (z_variance_map == 0).astype(int)
    ppr_mask = 1 - binary_fill_holes(1 - ppr_mask)
    return ppr_mask

############################## Synthetic truncation ##############################

class CreateMask(nn.Module):
    """ Creates a FOV mask to synthetically truncate a CT slice. """

    def __init__(self, trunc_severity="train"):
        super(CreateMask, self).__init__()
        assert trunc_severity in ["train", "val_test"]

        self.device = DEVICE
        self.trunc_severity = trunc_severity

    def forward(self, ct_slice): # ct_slice: (256, 256) / (3, 256, 256)
        self.ori_slice_dim = ct_slice.shape[1]

        if self.trunc_severity == "val_test":
            self.r_rfov = self._sample_uniform_dist(0.6, 0.85)
            self.r_dfov = self._sample_uniform_dist(0.7, 0.9)
        elif self.trunc_severity == "train":
            self.r_rfov = self._sample_uniform_dist(0.5, 0.7)
            self.r_dfov = self._sample_uniform_dist(0.65, 0.9)
        
        D = self.ori_slice_dim * (1 - self.r_dfov)
        self.x_dfov = self._sample_uniform_dist(-D/2, D/2)
        self.y_dfov = self._sample_uniform_dist(-D/2, D/2)
        
        self._create_rfov_mask()
        self._create_dfov_mask()
        truncated_ct_slice, zoomed_truncated_ct_slice, zoom_factor = self._truncate_slice(ct_slice)
    
        return {
            "rfov_mask": self.rfov_mask,
            "dfov_mask": self.dfov_mask,
            "truncated_ct_slice": truncated_ct_slice,
            "dfov_cropped_truncated_ct_slice": zoomed_truncated_ct_slice,
            "zoom_factor": zoom_factor,
            "r_rfov": self.r_rfov,
            "r_dfov": self.r_dfov,
            "x_dfov": self.x_dfov,
            "y_dfov": self.y_dfov
        }
    
    def _create_rfov_mask(self):
        rfov_diameter = self.ori_slice_dim * self.r_rfov
        rfov_radius = rfov_diameter / 2
        center = (self.ori_slice_dim / 2, self.ori_slice_dim / 2)
        x, y = torch.meshgrid(torch.arange(0, self.ori_slice_dim, device=self.device),
                              torch.arange(0, self.ori_slice_dim, device=self.device),
                              indexing="ij")
        distance = torch.sqrt((x - center[0])**2 + (y - center[1])**2)
        circle_mask = distance <= rfov_radius
        rfov_mask = torch.zeros(self.ori_slice_dim, self.ori_slice_dim, device=self.device)
        rfov_mask[circle_mask] = 1
        self.rfov_mask = rfov_mask
    
    def _create_dfov_mask(self):
        dfov_square_length = int(self.ori_slice_dim * self.r_dfov)
        dfov_mask = torch.zeros(self.ori_slice_dim, self.ori_slice_dim, device=self.device)
        square = torch.ones(dfov_square_length, dfov_square_length, device=self.device)
        center = (self.ori_slice_dim / 2, self.ori_slice_dim / 2)
        center = (center[0] + self.x_dfov, center[1] + self.y_dfov)
        x_square_start = int(center[0] - square.shape[0] // 2)
        x_square_end = int(x_square_start + dfov_square_length)
        y_square_start = int(center[1] - square.shape[0] // 2)
        y_square_end = int(y_square_start + dfov_square_length)
        dfov_mask[x_square_start:x_square_end, y_square_start:y_square_end] = 1
        self.dfov_mask = dfov_mask
    
    def _truncate_slice(self, ct_slice): # ct_slice: (256, 256) / (3, 256, 256) 
        # Combined mask
        combined_mask = self.rfov_mask * self.dfov_mask

        # Truncate slice
        truncated_ct_slice = ct_slice * combined_mask
        truncated_ct_slice[truncated_ct_slice == 0] = MIN_INTENSITY

        # Crop at DFOV
        nonzero_indices = torch.nonzero(self.dfov_mask)
        top_left = torch.min(nonzero_indices, dim=0).values
        bottom_right = torch.max(nonzero_indices, dim=0).values

        if len(ct_slice.shape) == 2:
            cropped_ct_slice = truncated_ct_slice[top_left[0]:bottom_right[0] + 1, top_left[1]:bottom_right[1] + 1]
            zoom_factor = ct_slice.shape[0] / cropped_ct_slice.shape[0]
        
            # Resize cropped CT slice
            cropped_ct_slice = cropped_ct_slice[None, None, :, :]
            zoomed_truncated_ct_slice = F.interpolate(cropped_ct_slice, size=self.ori_slice_dim, mode='bilinear')
            zoomed_truncated_ct_slice = zoomed_truncated_ct_slice.squeeze()
        
        elif len(ct_slice.shape) == 3:
            cropped_ct_slice = truncated_ct_slice[:, top_left[0]:bottom_right[0] + 1, top_left[1]:bottom_right[1] + 1]
            zoom_factor = ct_slice.shape[1] / cropped_ct_slice.shape[1]
        
            # Resize cropped CT slice
            cropped_ct_slice = cropped_ct_slice[None, :, :, :]
            zoomed_truncated_ct_slice = F.interpolate(cropped_ct_slice, size=self.ori_slice_dim, mode='bilinear')
            zoomed_truncated_ct_slice = zoomed_truncated_ct_slice.squeeze()

        return truncated_ct_slice, zoomed_truncated_ct_slice, zoom_factor

    def _sample_uniform_dist(self, start, end):
        return (end - start) * torch.rand(1).item() + start

def create_rfov_mask(img_dim: int, r_rfov: float) -> torch.Tensor:
    """ Create RFOV mask. """
    rfov_diameter = img_dim * r_rfov
    rfov_radius = rfov_diameter / 2
    center = (img_dim / 2, img_dim / 2)
    x, y = torch.meshgrid(torch.arange(0, img_dim), torch.arange(0, img_dim), indexing="ij")
    distance = torch.sqrt((x - center[0])**2 + (y - center[1])**2)
    circle_mask = distance <= rfov_radius
    rfov_mask = torch.zeros(img_dim, img_dim)
    rfov_mask[circle_mask] = 1
    return rfov_mask
    
def truncate_mask(mask: torch.Tensor, rfov_mask: torch.Tensor, dfov_mask: torch.Tensor) -> torch.Tensor:
    """ Truncates the given mask by the given RFOV and DFOV masks. """
    ori_slice_dim = mask.shape[1]
    
    # Combined mask
    combined_mask = rfov_mask * dfov_mask
    
    # Truncate mask
    truncated_mask = mask * combined_mask
    
    # Crop at DFOV
    nonzero_indices = torch.nonzero(dfov_mask)
    top_left = torch.min(nonzero_indices, dim=0).values
    bottom_right = torch.max(nonzero_indices, dim=0).values

    if len(mask.shape) == 2:
        cropped_mask = truncated_mask[top_left[0]:bottom_right[0] + 1, top_left[1]:bottom_right[1] + 1]
        
        # Resize cropped mask
        cropped_mask = cropped_mask[None, None, :, :]
        zoomed_truncated_mask = F.interpolate(cropped_mask, size=ori_slice_dim, mode='nearest').squeeze()

    elif len(mask.shape) == 3:
        cropped_mask = truncated_mask[:, top_left[0]:bottom_right[0] + 1, top_left[1]:bottom_right[1] + 1]
        
        # Resize cropped mask
        cropped_mask = cropped_mask[None, :, :, :]
        zoomed_truncated_mask = F.interpolate(cropped_mask, size=ori_slice_dim, mode='nearest').squeeze()
    
    return zoomed_truncated_mask

def truncate_slice(ct_slice: torch.Tensor, rfov_mask: torch.Tensor, dfov_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """ Truncates the given CT slice by the given RFOV and DFOV masks. """
    # Combined mask
    combined_mask = rfov_mask * dfov_mask

    # Truncate slice
    truncated_ct_slice = ct_slice * combined_mask
    truncated_ct_slice[truncated_ct_slice == 0] = -1

    # Crop at DFOV
    nonzero_indices = torch.nonzero(dfov_mask)
    top_left = torch.min(nonzero_indices, dim=0).values
    bottom_right = torch.max(nonzero_indices, dim=0).values
    cropped_ct_slice = truncated_ct_slice[top_left[0]:bottom_right[0] + 1, top_left[1]:bottom_right[1] + 1]
    zoom_factor = ct_slice.shape[0] / cropped_ct_slice.shape[0]
    
    # Resize cropped CT slice
    cropped_ct_slice = cropped_ct_slice[None, None, :, :]
    zoomed_truncated_ct_slice = F.interpolate(cropped_ct_slice, size=ct_slice.shape[0], mode='bilinear')
    zoomed_truncated_ct_slice = zoomed_truncated_ct_slice.squeeze()

    return truncated_ct_slice, zoomed_truncated_ct_slice, zoom_factor

def check_if_lung_region_cropped(lung_mask: torch.Tensor, fov_mask: torch.Tensor, tol_num_pixels: int = 0) -> bool:
    """ Checks if the lungs are cropped due to the field-of-view. """  
    fov_boundary = create_img_boundary(fov_mask)
    
    num_pixels_of_lung_at_fov_boundary = torch.sum(fov_boundary * lung_mask)

    if num_pixels_of_lung_at_fov_boundary > tol_num_pixels:
        return True
    else:
        return False

############################## Body bounding box ##############################

def extend_body_bb(body_bb: torch.Tensor, num_pixels: int, img_dim: int = 256) -> Tuple[int, int, int, int]:
    """ Extends the body bounding box by the given number of pixels. """
    x_min = int(body_bb[1])-num_pixels if int(body_bb[1])-num_pixels >= 0 else 0
    x_max = int(body_bb[3])+num_pixels if int(body_bb[3])+num_pixels <= img_dim-1 else img_dim-1
    y_min = int(body_bb[0])-num_pixels if int(body_bb[0])-num_pixels >= 0 else 0
    y_max = int(body_bb[2])+num_pixels if int(body_bb[2])+num_pixels <= img_dim-1 else img_dim-1
    return y_min, x_min, y_max, x_max

def extract_bb_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """ Extracts the bounding box from the given mask. """
    nonzero_indices = torch.nonzero(mask)
    top_left = torch.min(nonzero_indices, dim=0).values
    bottom_right = torch.max(nonzero_indices, dim=0).values
    return torch.stack((top_left[1], top_left[0], bottom_right[1], bottom_right[0]))

def rescale_bb_by_zoom_factor(bounding_box: torch.Tensor, zoom_factor: float):
    """ Rescales the bounding box by the zoom factor. """
    return bounding_box * zoom_factor

def extract_groundtruth_body_bb(body_mask: torch.Tensor, dfov_mask: torch.Tensor, zoom_factor: float, tol_factor: float = 1.001) -> torch.Tensor:
    """
    Extracts the ground-truth body bounding box from a DFOV-cropped truncated slice.
    The body bounding box may exceed the image borders.
    Args:
        - body_mask: The untruncated body mask of the original CT slice.
        - dfov_mask: The DFOV mask which is used to synthetically truncate the original CT slice.
        - zoom_factor: The ratio between the desired image dimension (256) and the dimension after DFOV cropping.
        - tol_factor: A tolerance factor used to increase the size of the bounding box.
    """
    # Rescale body BB and DFOV BB
    body_BB = extract_bb_from_mask(body_mask)
    rescaled_body_BB = rescale_bb_by_zoom_factor(body_BB, zoom_factor * tol_factor)
    
    dfov_BB = extract_bb_from_mask(dfov_mask)
    rescaled_dfov_BB = rescale_bb_by_zoom_factor(dfov_BB, zoom_factor * tol_factor)
    
    # Post-process rescaled body BB
    new_y_min = rescaled_body_BB[0] - rescaled_dfov_BB[0]
    new_x_min = rescaled_body_BB[1] - rescaled_dfov_BB[1]

    new_y_max = new_y_min + rescaled_body_BB[2] - rescaled_body_BB[0]
    new_x_max = new_x_min + rescaled_body_BB[3] - rescaled_body_BB[1]

    new_rescaled_body_BB = torch.stack((new_y_min, new_x_min, new_y_max, new_x_max))
    
    return new_rescaled_body_BB

def transform_body_bb_into_coords(body_bb: torch.Tensor) -> torch.Tensor:
    """
    Transforms the given (body) bounding box into 4 (x, y) coordinates.
    This method assumes that the bounding box is NOT rotated!
    """
    x1 = body_bb[1]
    y1 = body_bb[0]
    x2 = body_bb[3]
    y2 = y1

    x3 = x1
    y3 = body_bb[2]
    x4 = x2
    y4 = y3
    
    return torch.stack((x1, y1, x2, y2, x3, y3, x4, y4))

def axis_align_bounding_box(bb_coords: torch.Tensor) -> torch.Tensor:
    """ Aligns a possibly rotated bounding box by the x and y axis. """
    y = torch.stack([bb_coords[1], bb_coords[3], bb_coords[5], bb_coords[7]])
    x = torch.stack([bb_coords[0], bb_coords[2], bb_coords[4], bb_coords[6]])

    return torch.stack((torch.min(y), torch.min(x), torch.max(y), torch.max(x)))

def plot_body_bb_and_slice(ct_slice: Union[np.ndarray, torch.Tensor], body_BB: Union[np.ndarray, torch.Tensor], axis: bool = True):
    """ Plots the given (body) bounding box on top of the CT slice. """
    fig, ax = plt.subplots()
    ax.imshow(ct_slice, cmap='gray')
    rect = patches.Rectangle((body_BB[0], body_BB[1]), body_BB[2] - body_BB[0], body_BB[3] - body_BB[1],
                             linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    rect.set_clip_on(False)

    if not axis:
        plt.axis('off')

    plt.show()

def plot_body_bb_coords_and_slice(ct_slice: Union[np.ndarray, torch.Tensor], body_bb_coords: Union[np.ndarray, torch.Tensor]):
    """ Plots the given (body) bounding box coordinates on top of the CT slice. """
    fig, ax = plt.subplots()
    ax.imshow(ct_slice, cmap='gray')
    
    transf_x = np.array([body_bb_coords[0], body_bb_coords[2], body_bb_coords[6], body_bb_coords[4], body_bb_coords[0]])
    transf_y = np.array([body_bb_coords[1], body_bb_coords[3], body_bb_coords[7], body_bb_coords[5], body_bb_coords[1]])

    ax.plot(transf_y, transf_x)

    plt.show()

############################## Image outpainting ##############################

def create_CT_Palette_mask(body_mask: torch.Tensor, body_bb: torch.Tensor, img_dim: int = 256, num_tol_pixels: int = 10):
    """ Creates small mask for the image outpainting model using the given body bounding box and body mask. """
    tol_bb_mask = torch.zeros(img_dim, img_dim)
    y_min, x_min, y_max, x_max = extend_body_bb(body_bb, num_pixels=num_tol_pixels)
    tol_bb_mask[x_min:x_max, y_min:y_max] = 1
    mask = tol_bb_mask - body_mask
    mask = mask.expand(1, -1, -1).to(torch.float32)
    return mask

############################## Augmentation ##############################

def augment_slice(slice_tensor: torch.Tensor, degrees: float = 0,
                  translate: Tuple[float, float] = (0.2, 0.1),
                  scale: Tuple[float, float] = (0.7, 1),
                  fill: float = 0, img_size: int = RESIZED_IMG_SIZE):
    """ Augment the CT slice by rotation, translation, and scaling using bilinear interpolation. """
    transformer = T.RandomAffine(degrees=degrees, translate=translate, scale=scale,
                                 interpolation=T.InterpolationMode.BILINEAR, fill=fill)
    transform_params = transformer.get_params(transformer.degrees, transformer.translate, transformer.scale, transformer.shear,
                                              (img_size, img_size))
    transformed_slice = TF.affine(slice_tensor, *transform_params,
                             interpolation=transformer.interpolation,
                             fill=transformer.fill)
    return transformed_slice, transform_params

def augment_mask(mask_tensor: torch.Tensor, transform_params):
    """ Augment the binary mask by rotation, translation, and scaling using nearest interpolation. """
    transformed_mask = TF.affine(mask_tensor, *transform_params,
                             interpolation=T.InterpolationMode.NEAREST,
                             fill=0)
    return transformed_mask

def rotate_torch(x: float, y: float, degree: float) -> Tuple[float, float]:
    """Rotate a point (x, y) around the origin by an angle theta (in radians)"""
    degree = torch.tensor(degree)
    theta = torch.deg2rad(degree)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    x_new = x * cos_theta - y * sin_theta
    y_new = x * sin_theta + y * cos_theta
    return x_new, y_new

def transform_bounding_box(body_bb: torch.Tensor, scale_factor: float, rotation_angle: float, translation_x: float, translation_y: float) -> torch.Tensor:
    x1, y1, x2, y2, x3, y3, x4, y4 = body_bb.to(torch.float32)
    
    center_x = (x1 + x4) / 2
    center_y = (y1 + y4) / 2
    
    # Translate the center of the bounding box to the origin
    x1 -= center_x
    y1 -= center_y
    x2 -= center_x
    y2 -= center_y
    x3 -= center_x
    y3 -= center_y
    x4 -= center_x
    y4 -= center_y

    # Rotate the bounding box around the center
    x1, y1 = rotate_torch(x1, y1, rotation_angle)
    x2, y2 = rotate_torch(x2, y2, rotation_angle)
    x3, y3 = rotate_torch(x3, y3, rotation_angle)
    x4, y4 = rotate_torch(x4, y4, rotation_angle)

    # Scale the bounding box with respect to the center
    x1 *= scale_factor
    y1 *= scale_factor
    x2 *= scale_factor
    y2 *= scale_factor
    x3 *= scale_factor
    y3 *= scale_factor
    x4 *= scale_factor
    y4 *= scale_factor

    # Translate the bounding box back to its original position
    x1 += center_x + translation_x
    y1 += center_y + translation_y
    x2 += center_x + translation_x
    y2 += center_y + translation_y
    x3 += center_x + translation_x
    y3 += center_y + translation_y
    x4 += center_x + translation_x
    y4 += center_y + translation_y

    return torch.round(torch.stack((x1, y1, x2, y2, x3, y3, x4, y4)))

def check_bbb_exceeds_border(body_bb_coords: torch.Tensor, img_size: int, tol: int = 3) -> bool:
    """ Checks if the body bounding box coordinates exceed the image border. """
    x = torch.Tensor([body_bb_coords[0], body_bb_coords[2], body_bb_coords[4], body_bb_coords[6]])
    y = torch.Tensor([body_bb_coords[1], body_bb_coords[3], body_bb_coords[5], body_bb_coords[7]])

    x_min, x_max = torch.min(x), torch.max(x)
    y_min, y_max = torch.min(y), torch.max(y)

    if (x_min < 0 - tol) or (x_max >= img_size + tol) or (y_min < 0 - tol) or (y_max >= img_size + tol):
        return True
    else:
        return False