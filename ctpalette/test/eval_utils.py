""" This script contains useful functions for model evaluation/inference. """

import json
from collections import OrderedDict
from typing import Union, Dict, Any, List, Tuple

import numpy as np
import pydicom
import matplotlib.pyplot as plt
from matplotlib import colors
import statistics
from PIL import Image
import torch
import torchvision.transforms as T
from tensorflow.keras.models import load_model
from scipy.ndimage import binary_erosion, binary_fill_holes

from ctpalette.models.model import BodyBBModel
from ctpalette.models.network import Network
import ctpalette.data.data_utils as data_utils
import ctpalette.core.util as Util

downsample_fn = T.Resize(size=256, antialias=True)
upsample_fn = T.Resize(size=512, antialias=True)

class NoneDict(dict):
    def __missing__(self, key):
        return None

def dict_to_nonedict(opt):
    """ Convert to NoneDict, which return None for missing key. """
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt
    
def parse(config, phase, gpu_ids, batch, debug=False):
    json_str = ''
    with open(config, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)

    ''' replace the config context using args '''
    opt['phase'] = phase
    if gpu_ids is not None:
        opt['gpu_ids'] = [int(id) for id in gpu_ids.split(',')]
    if batch is not None:
        opt['datasets'][opt['phase']]['dataloader']['args']['batch_size'] = batch
 
    ''' set cuda environment '''
    if len(opt['gpu_ids']) > 1:
        opt['distributed'] = True
    else:
        opt['distributed'] = False

    ''' update name '''
    if debug:
        opt['name'] = 'debug_{}'.format(opt['name'])
    elif opt['finetune_norm']:
        opt['name'] = 'finetune_{}'.format(opt['name'])
    else:
        opt['name'] = '{}_{}'.format(opt['phase'], opt['name'])

    return dict_to_nonedict(opt)

def save_img_pillow(img: np.ndarray, img_path: str):
    img = Image.fromarray(img.astype(np.uint8))
    img.save(img_path)

############################## Body composition segmentation ##############################

def load_segmentation_model(model_weights_path: str):
    """ Loads the BC segmentation model stored in the given path. """
    seg_model = load_model(model_weights_path, compile=False)
    return seg_model

def run_segmentation_model(rgb_image: Union[np.ndarray, torch.Tensor], seg_model, pixel_spacing: float) -> Tuple[np.ndarray, float, float]:
    """
    Runs the BC segmentation model and returns the segmentation mask and muscle and SAT areas.
    Args:
        - rgb_image: Image of range [0, 255]
        - seg_model: Segmentation model to segment the image
        - pixel_spacing: The pixel spacing of the image used to calculate the muscle and SAT area segmented by the model in terms of cm2
    """
    if not isinstance(rgb_image, np.ndarray):
        rgb_image = rgb_image.numpy()
    seg_mask = segmentation(rgb_image, model=seg_model)
    hu_image = data_utils.transform_to_HU(data_utils.normalize_rgb_img(rgb_image))
    if not isinstance(hu_image, np.ndarray):
        hu_image = hu_image.numpy()
    seg_results = find_quantitative_seg_results(seg_mask, hu_image, pixel_spacing)
    muscle_area = seg_results["Muscle"]["area_cm2"]
    sat_area = seg_results["SAT"]["area_cm2"]
    return seg_mask, muscle_area, sat_area

def segmentation(rgb_image: np.ndarray, model) -> np.ndarray:
    """
    Runs the BC segmentation model and returns the segmentation mask.
    Args:
        - rgb_image: Image of range [0, 255]
        - model: Segmentation model to segment the image
    """
    # Resize and reshape
    rgb_image = np.transpose(rgb_image[:, :, np.newaxis, np.newaxis], (2, 0, 1, 3)) # (1, H, W, 1)

    segmentation_predictions = model.predict(rgb_image) # chest_segmentation.hdf5
    segmentation_mask = np.argmax(segmentation_predictions, axis=3)
    segmentation_mask = np.squeeze(segmentation_mask)

    return segmentation_mask

def segmentation_batch(rgb_images: np.ndarray, model) -> np.ndarray:
    """
    Runs the BC segmentation model for a batch of images and returns the segmentation masks.
    Args:
        - rgb_images: Images of range [0, 255] with shape (B, C, H, W)
        - model: Segmentation model to segment the image
    """
    # Resize and reshape
    rgb_images = np.transpose(rgb_images, (0, 2, 3, 1)) # (B, H, W, C)

    segmentation_predictions = model.predict(rgb_images) # chest_segmentation.hdf5
    segmentation_masks = np.argmax(segmentation_predictions, axis=3)
    segmentation_masks = np.squeeze(segmentation_masks)

    return segmentation_masks

def find_pixel_statistics(pixels: np.ndarray) -> Dict:
    """ Finds the mean, median, standard deviation, and interquartile range of HU values (pixels). """
    results = {}
    if len(pixels) < 1:
        results['mean_hu'] = None
        results['median_hu'] = None
        results['std_hu'] = None
        results['iqr_hu'] = None
    else:
        results['mean_hu'] = float(pixels.mean())
        results['median_hu'] = float(np.median(pixels))
        results['std_hu'] = float(np.std(pixels))
        q75, q25 = np.percentile(pixels, [75, 25])
        results['iqr_hu'] = float(q75 - q25)
    return results

def find_quantitative_seg_results(seg_mask: np.ndarray, hu_image: np.ndarray, pixel_spacing: float) -> Dict:
    """ Finds the quantitative results from segmentation, such as muscle and SAT areas and HU statistics. """
    # Loop over tissues and calculate metrics
    seg_results = {
        "Muscle": {},
        "SAT": {}
    }
    for c in [1, 2]:
        if c == 1:
            organ = "Muscle"
        elif c == 2:
            organ = "SAT"
        pixel_area = float(pixel_spacing) * float(pixel_spacing) / 100.0
        pixel_count = (seg_mask == c).sum()
        area_cm2 = float(pixel_count * pixel_area)
        seg_values = hu_image[seg_mask == c]
        if len(seg_values) == 0:
            for key in ['mean_hu', 'median_hu', 'std_hu', 'iqr_hu']:
                seg_results[organ][key] = np.nan
        else:
            # Add pixel statistics to the result
            seg_results[organ] = find_pixel_statistics(seg_values)
        seg_results[organ]["area_cm2"] = area_cm2
    return seg_results

def dice_similarity_coefficient(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """ Finds the Dice Similarity Coefficient (DSC) between the given masks. """
    intersection = np.logical_and(mask1, mask2).sum()
    total_pixels_mask1 = mask1.sum()
    total_pixels_mask2 = mask2.sum()
    
    dice_coefficient = (2.0 * intersection) / (total_pixels_mask1 + total_pixels_mask2)
    
    return dice_coefficient

############################## Body bounding box detector ##############################

def load_body_bb_model(model_weights_path: str):
    """ Loads the body bounding box detector stored in the given path. """
    body_bb_model = BodyBBModel()
    body_bb_model = body_bb_model.to(body_bb_model.device)
    body_bb_model.load_state_dict(torch.load(model_weights_path, map_location="cpu"))
    body_bb_model.eval()
    return body_bb_model

def run_body_bb_model(body_bb_model, image: torch.Tensor) -> torch.Tensor:
    """
    Runs the body bounding box detector and returns the predicted body bounding box.
    Args:
        - body_bb_model: The body bounding box detector
        - image: Image of shape (3, 256, 256)
    """
    with torch.no_grad():
        body_bb_pred = body_bb_model(image[None, :, :, :].cuda()) # (batch_size, 4)
        body_bb_pred = body_bb_pred.squeeze().cpu()
    return body_bb_pred

############################## Image outpainting model ##############################

def load_outpainting_model(config_json_path: str, model_weights_path: str):
    """
    Loads the image outpainting model (Palette).
    Args:
        - config_json_path: The path to the configuration for the image outpainting model
        - model_weights_path: The path to the model weights.
    """
    opt = parse(config=config_json_path, phase="test", gpu_ids="0", batch=32)
    opt["model"]["which_networks"][0]["args"]["beta_schedule"]["test"]["n_timestep"] = 1000
    torch.backends.cudnn.enabled = True
    Util.set_seed(opt['seed'])
    network = Network(
        **opt["model"]["which_networks"][0]["args"]
    )
    network.load_state_dict(torch.load(model_weights_path, map_location="cpu"), strict=False)
    network.set_new_noise_schedule(phase="test")
    network = network.cuda()
    network.eval()
    return network

def run_outpainting_model(outpainting_model, cond_image: torch.Tensor,
                          mask_image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Runs the image outpainting model once and returns the outpainted image.
    Args:
        - outpainting_model: The Palette model used for outpainting
        - cond_image: Image of size (1, 256, 256) representing the truncated CT slice but with noise in the unknown region
        - mask_image: Image of size (1, 256, 256) representing the truncated CT slice
        - mask: Binary mask of size (1, 256, 256) indicating the region the model should outpaint in
    Returns:
        outpainted_slice: Outpainted image of size (1, 256, 256) and range [-1, 1]
    """
    with torch.no_grad():
        outpainted_slice, _ = outpainting_model.restoration(y_cond=cond_image[None, :, :, :].cuda(),
                                                            y_t=cond_image[None, :, :, :].cuda(),
                                                            y_0=mask_image[None, :, :, :].cuda(),
                                                            mask=mask[None, :, :, :].cuda(),
                                                            sample_num=8)
    return outpainted_slice

def outpainting_single_inference(outpainting_model, cond_image: torch.Tensor,
                                 mask_image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Executes single inference for the image outpainting model and returns the outpainted image.
    Args:
        - outpainting_model: The Palette model used for outpainting
        - cond_image: Image of size (1, 256, 256) representing the truncated CT slice but with noise in the unknown region
        - mask_image: Image of size (1, 256, 256) representing the truncated CT slice
        - mask: Binary mask of size (1, 256, 256) indicating the region the model should outpaint in
    Returns:
        rgb_outpainted_slice: Outpainted image of size (512, 512) and range [0, 255]
    """
    outpainted_slice = run_outpainting_model(outpainting_model=outpainting_model, cond_image=cond_image, mask_image=mask_image, mask=mask)
    outpainted_slice = outpainted_slice.squeeze().cpu() # take first channel
    outpainted_slice = torch.clamp(outpainted_slice, -1, 1)
    outpainted_slice = mask_image * (1 - mask) + outpainted_slice * mask
    rgb_outpainted_slice = data_utils.transform_to_RGB(outpainted_slice)
    rgb_outpainted_slice = data_utils.upsample_img(img=rgb_outpainted_slice, upsample_fn=upsample_fn)
    
    return rgb_outpainted_slice

def outpainting_multiple_inference(outpainting_model, seg_model, cond_image: torch.Tensor, mask_image: torch.Tensor, mask: torch.Tensor,
                                   pixel_spacing: float, num_generated_samples: int = 5,
                                   mode: str = "median", outlier_detection: bool = True) -> torch.Tensor:
    """
    Executes multiple inference for the image outpainting model and returns the outpainted image.
    Args:
        - outpainting_model: The Palette model used for outpainting
        - seg_model: The BC segmentation model used for detecting outliers and extracting muscle and SAT areas
        - cond_image: Image of size (1, 256, 256) representing the truncated CT slice but with noise in the unknown region
        - mask_image: Image of size (1, 256, 256) representing the truncated CT slice
        - mask: Binary mask of size (1, 256, 256) indicating the region the model should outpaint in
        - pixel_spacing: Pixel spacing of the truncated CT image
        - num_generated_samples: The number of outpainted images to be generated
        - mode: median/mean
        - outlier_detection: Whether outlier detection should be used to remove outliers
    Returns:
        rgb_sample_with_smallest_dist: Outpainted image with the smallest distance to median/mean, of size (512, 512) and range [0, 255]
    """
    assert mode in ["median", "mean"]

    preds = []
    pred_seg_masks = []
    pred_muscle_areas = []
    pred_sat_areas = []

    # 1. Generate n outpainted slices
    for i in range(num_generated_samples):
        outpainted_slice = run_outpainting_model(outpainting_model=outpainting_model, cond_image=cond_image, mask_image=mask_image, mask=mask)

        preds.append(outpainted_slice)

        # Pass the outpainted sample through the segmentation model
        pred_slice = outpainted_slice.squeeze().cpu() # take first channel
        pred_slice = torch.clamp(pred_slice, -1, 1)
        hu_pred_slice = data_utils.transform_to_HU(pred_slice)
        hu_pred_slice = data_utils.upsample_img(img=hu_pred_slice, upsample_fn=upsample_fn)

        rgb_pred_slice = data_utils.transform_to_RGB(pred_slice) # change the intensity back to [0, 255]
        rgb_pred_slice = data_utils.upsample_img(img=rgb_pred_slice, upsample_fn=upsample_fn)
        pred_seg_mask = segmentation(rgb_pred_slice.numpy(), seg_model)
        pred_seg_masks.append(pred_seg_mask)
        
        # Find the quantitative measures of the segmentation results
        pred_seg_results = find_quantitative_seg_results(pred_seg_mask, hu_pred_slice.numpy(), pixel_spacing)
        pred_muscle_areas.append(pred_seg_results["Muscle"]["area_cm2"])
        pred_sat_areas.append(pred_seg_results["SAT"]["area_cm2"])

    # 2. Outlier detection
    if outlier_detection:
        # Find outliers from the first batch of runs
        outlier_indices = find_outliers(pred_muscle_areas, pred_sat_areas)
        num_reruns = len(outlier_indices)

        # Rerun model if there are outliers
        while num_reruns > 0:
            # Save non-outlier indices in a new list
            non_outlier_preds = []
            non_outlier_pred_seg_masks = []
            non_outlier_pred_muscle_areas = []
            non_outlier_pred_sat_areas = []
            for i in range(len(pred_muscle_areas)):
                if i not in outlier_indices:
                    non_outlier_preds.append(preds[i])
                    non_outlier_pred_seg_masks.append(pred_seg_masks[i])
                    non_outlier_pred_muscle_areas.append(pred_muscle_areas[i])
                    non_outlier_pred_sat_areas.append(pred_sat_areas[i])
            
            preds = non_outlier_preds
            pred_seg_masks = non_outlier_pred_seg_masks
            pred_muscle_areas = non_outlier_pred_muscle_areas
            pred_sat_areas = non_outlier_pred_sat_areas

            # Rerun Palette
            for i in range(num_reruns):
                outpainted_slice = run_outpainting_model(outpainting_model=outpainting_model, cond_image=cond_image, mask_image=mask_image, mask=mask)
                preds.append(outpainted_slice)
                
                # Pass the outpainted sample through the segmentation model
                pred_slice = outpainted_slice.squeeze().cpu() # take first channel
                pred_slice = torch.clamp(pred_slice, -1, 1)
                hu_pred_slice = data_utils.transform_to_HU(pred_slice)
                hu_pred_slice = data_utils.upsample_img(img=hu_pred_slice, upsample_fn=upsample_fn)
                rgb_pred_slice = data_utils.transform_to_RGB(pred_slice)
                rgb_pred_slice = data_utils.upsample_img(img=rgb_pred_slice, upsample_fn=upsample_fn)
                pred_seg_mask = segmentation(rgb_pred_slice.numpy(), seg_model)
                pred_seg_masks.append(pred_seg_mask)
                
                # Find the quantitative measures of the segmentation results
                pred_seg_results = find_quantitative_seg_results(pred_seg_mask, hu_pred_slice.numpy(), pixel_spacing)
                pred_muscle_areas.append(pred_seg_results["Muscle"]["area_cm2"])
                pred_sat_areas.append(pred_seg_results["SAT"]["area_cm2"])

            # Are there still outliers? If yes, run the loop again. Else, break.
            outlier_indices = find_outliers(pred_muscle_areas, pred_sat_areas)
            num_reruns = len(outlier_indices)

    # 3. Find predicted image closest to mean/median BC metrics
    mean_bc_muscle_area = statistics.mean(pred_muscle_areas)
    mean_bc_sat_area = statistics.mean(pred_sat_areas)
    median_bc_muscle_area = statistics.median(pred_muscle_areas)
    median_bc_sat_area = statistics.median(pred_sat_areas)

    dists = []
    for i in range(num_generated_samples):
        if mode == "mean": # Calculate L2 distance
            dist = np.sqrt((mean_bc_muscle_area - pred_muscle_areas[i]) ** 2 + (mean_bc_sat_area - pred_sat_areas[i]) ** 2)
        elif mode == "median": # Calculate L1 distance
            dist = abs(median_bc_muscle_area - pred_muscle_areas[i]) + abs(median_bc_sat_area - pred_sat_areas[i])
        dists.append(dist)

    sample_idx_smallest_dist = dists.index(min(dists))
    sample_with_smallest_dist = preds[sample_idx_smallest_dist].squeeze().cpu()
    sample_with_smallest_dist = torch.clamp(sample_with_smallest_dist, -1, 1)
    rgb_sample_with_smallest_dist = data_utils.transform_to_RGB(sample_with_smallest_dist)
    rgb_sample_with_smallest_dist = data_utils.upsample_img(img=rgb_sample_with_smallest_dist, upsample_fn=upsample_fn)

    return rgb_sample_with_smallest_dist

def find_outliers(pred_muscle_areas: List[float], pred_sat_areas: List[float]) -> List[int]:
    """ Finds outpainted images which are outliers based on their muscle and SAT areas. """
    num_samples = len(pred_muscle_areas)

    median_bc_muscle_area = statistics.median(pred_muscle_areas)
    median_bc_sat_area = statistics.median(pred_sat_areas)
    
    MAD_bc_muscle_area = statistics.median([abs(muscle_area - median_bc_muscle_area) for muscle_area in pred_muscle_areas])
    MAD_bc_sat_area = statistics.median([abs(sat_area - median_bc_sat_area) for sat_area in pred_sat_areas])
    
    muscle_z_scores = [(0.6745 * (muscle_area - median_bc_muscle_area)) / (MAD_bc_muscle_area + 1e-8) for muscle_area in pred_muscle_areas]
    sat_z_scores = [(0.6745 * (sat_area - median_bc_sat_area)) /(MAD_bc_sat_area + 1e-8) for sat_area in pred_sat_areas]
    
    outlier_indices = []
    for i in range(num_samples):
        if (muscle_z_scores[i] < -3) or (muscle_z_scores[i] > 3) or (sat_z_scores[i] < -3) or (sat_z_scores[i] > 3):
            outlier_indices.append(i)

    return outlier_indices

############################## CT-Palette ##############################

def extend_fov(ct_slice: torch.Tensor, body_mask: np.ndarray, pixel_spacing: float,
               body_bb: torch.Tensor = None, img_dim: int = 256, center: bool = True) -> Tuple[torch.Tensor, np.ndarray, float, float]:
    """
    Extends the field-of-view of the DFOV-cropped truncated slice.
    Args:
        - ct_slice (torch.Tensor): DFOV-cropped truncated CT slice whose FOV is to be extended
        - body_mask (np.ndarray): The body mask corresponding to the truncated CT slice
        - pixel_spacing (float): The pixel spacing of the truncated slice
        - body_bb (torch.Tensor): The body BB of the complete body. If None, we will zoom out to a fixed pixel spacing of 0.977.
        - img_dim (int): The dimension of the truncated CT slice
        - center (bool): Whether the body should be centered when doing FOV extension

    Returns:
        - extend_img (torch.Tensor): The FOV-extended truncated CT slice
        - extend_body_mask (np.ndarray): The corresponding FOV-extended body mask belonging to the truncated CT slice
        - ext_ratio (float): The ratio between the dimension of the FOV-extended image and the original image
        - new_pixel_spacing (float): The pixel spacing after FOV extension
    """
    if body_bb != None: # If body BB is available, zoom out s.t. the body occupies ~42% of the image
        h = body_bb[2] - body_bb[0]
        w = body_bb[3] - body_bb[1]
        extend_fov_dim = int(np.round(torch.sqrt(h * w / 0.42).item()))
    else: # If body BB is unavailable, zoom out to a pixel spacing of 0.977
        extend_fov_dim = int(0.977 * img_dim / pixel_spacing)

    if extend_fov_dim < img_dim:
        # This is usually when the slices are truncated because the patients are not centered, not because the patients have high BMI
        # Hence, they may already occupy less than 42% of the image
        # In this case, we just center the patient and not do FOV extension
        center = True

    if center:
        trunc_body_bb = data_utils.extract_bb_from_mask(torch.from_numpy(body_mask).to(torch.float32))
        trunc_body_bb = data_utils.extend_body_bb(body_bb=trunc_body_bb, num_pixels=2)
        h = trunc_body_bb[2] - trunc_body_bb[0]
        w = trunc_body_bb[3] - trunc_body_bb[1]
        center_extend_img = int(extend_fov_dim // 2) # 187
        x_start = int(center_extend_img - w/2)
        y_start = int(center_extend_img - h/2)

        # Extend the ct slice
        extend_img = torch.ones(extend_fov_dim, extend_fov_dim).to(torch.float32)
        extend_img = extend_img * -1
        extend_img[x_start:x_start + w, y_start:y_start + h] = ct_slice[0][trunc_body_bb[1]:trunc_body_bb[3], trunc_body_bb[0]:trunc_body_bb[2]]
    
        # Extend the body mask
        extend_body_mask = np.zeros((extend_fov_dim, extend_fov_dim), dtype=float)
        extend_body_mask[x_start:x_start + w, y_start:y_start + h] = body_mask[trunc_body_bb[1]:trunc_body_bb[3], trunc_body_bb[0]:trunc_body_bb[2]]
        
    else:
        x_start = int(round((extend_fov_dim - img_dim) / 2))
        y_start = int(round((extend_fov_dim - img_dim) / 2))
    
        # Extend the ct slice
        extend_img = torch.ones(extend_fov_dim, extend_fov_dim).to(torch.float32)
        extend_img = extend_img * -1
        extend_img[x_start:x_start + img_dim, y_start:y_start + img_dim] = ct_slice[0][:, :]
    
        # Extend the body mask
        extend_body_mask = np.zeros((extend_fov_dim, extend_fov_dim), dtype=float)
        extend_body_mask[x_start:x_start + img_dim, y_start:y_start + img_dim] = body_mask[:, :] # The mask is flipped (0 -> 1, 1 -> 0)

    ext_ratio = extend_fov_dim / img_dim
    new_pixel_spacing = pixel_spacing * ext_ratio

    return extend_img, extend_body_mask, ext_ratio, new_pixel_spacing

def run_CT_Palette(dcm: pydicom.dataset.FileDataset, body_bb_model, outpainting_model,
                   center: bool = True, multiple_inference: bool = False, **multiple_inference_kwargs) -> Tuple[torch.Tensor, float]:
    """
    Runs CT-Palette (body bounding box detector + image outpainting model) with single/multiple inference.
    Args:
        - dcm: The dcm whose FOV is to be extended
        - body_bb_model: The body bounding box model
        - outpainting_model: The image outpainting model
        - center (bool): Whether the body should be centered when doing FOV extension
        - multiple_inference (bool): If True, use multiple inference for outpainting
        - multiple_inference_kwargs: Arguments for multiple inference (seg_model, num_generated_samples, mode, outlier_detection)
    Returns:
        - rgb_outpainted_slice (torch.Tensor): The outpainted image of size (512, 512) and range [0, 255]
        - new_pixel_spacing (float): The pixel spacing after FOV extension
    """

    dcm_img = data_utils.transform_dcm_to_HU(dcm)

    # Create body mask for truncated CT slice
    try:
        body_mask = data_utils.create_body_mask_slice(dcm_img)
    except:
        print("Body mask cannot be extracted. Skip slice.")

    lung_mask = data_utils.create_lung_mask_slice(dcm_img)
    body_mask = body_mask + lung_mask
    body_mask[body_mask >= 1] = 1
    body_mask = binary_fill_holes(body_mask)

    # Preprocess truncated CT slice
    preprocessed_ct_slice = data_utils.preprocess_slice(dcm_img, body_mask)
    preprocessed_ct_slice = torch.from_numpy(preprocessed_ct_slice).to(torch.float32)
    preprocessed_ct_slice = data_utils.normalize_rgb_img(preprocessed_ct_slice)

    if np.min(dcm_img) <= -1024:
        ppr_mask = data_utils.create_ppr_mask(dcm_img[:, :, None], cutoff=np.min(dcm_img))
        fov_mask = binary_erosion(binary_fill_holes(1 - ppr_mask), iterations=4)
        body_mask = fov_mask * body_mask # erode the body mask because the generated body mask is a bit bigger than the body

    # Resize truncated CT slice and body mask from 512 to 256
    input_slice = data_utils.downsample_img(img=preprocessed_ct_slice, downsample_fn=downsample_fn)
    input_slice = input_slice.expand(3, -1, -1)
    body_mask = data_utils.resize_mask(body_mask, resized_dim=256)
    
    # Predict body bounding box for truncated CT slice
    body_bb_pred = run_body_bb_model(body_bb_model=body_bb_model, image=input_slice)

    # Extend truncated CT slice and body mask before outpainting
    extend_img, extend_body_mask, ext_ratio, new_pixel_spacing = extend_fov(ct_slice=input_slice,
                                                                            body_mask=body_mask,
                                                                            pixel_spacing=dcm.PixelSpacing[0],
                                                                            body_bb=body_bb_pred,
                                                                            center=center)

    # Resize the truncated CT slice and body mask back to 256
    extend_img = data_utils.downsample_img(img=extend_img, downsample_fn=downsample_fn)
    extend_body_mask = data_utils.resize_mask(extend_body_mask, resized_dim=256)
    trunc_slice = extend_img.expand(3, -1, -1)

    # Predict body bounding box for extended truncated CT slice
    body_bb_pred = run_body_bb_model(body_bb_model=body_bb_model, image=trunc_slice)

    # Create mask for the image outpainting model
    mask = data_utils.create_CT_Palette_mask(body_mask=extend_body_mask, body_bb=body_bb_pred)
    
    # Prepare inputs for the image outpainting model
    trunc_slice = trunc_slice[0][None, :, :]
    cond_slice = trunc_slice * (1 - mask) + torch.randn_like(trunc_slice) * mask

    # Outpaint truncated CT slice
    if multiple_inference: # Multiple inference
        rgb_outpainted_slice = outpainting_multiple_inference(outpainting_model=outpainting_model,
                                                              cond_image=cond_slice,
                                                              mask_image=trunc_slice,
                                                              mask=mask,
                                                              pixel_spacing=new_pixel_spacing,
                                                              **multiple_inference_kwargs)
        

    else: # Single inference
        rgb_outpainted_slice = outpainting_single_inference(outpainting_model=outpainting_model,
                                                            cond_image=cond_slice,
                                                            mask_image=trunc_slice,
                                                            mask=mask)
    
    return rgb_outpainted_slice, new_pixel_spacing

def run_SEFOV(dcm: pydicom.dataset.FileDataset, body_bb_model, outpainting_model, center: bool = False):
    """
    Runs S-EFOV (body bounding box detector + image outpainting model).

    Args:
        - dcm (pydicom.dataset.FileDataset): The dcm whose FOV is to be extended
        - body_bb_model: The body bounding box model
        - outpainting_model: The image outpainting model
        - center (bool): Whether the body should be centered when doing FOV extension

    Returns:
        - rgb_outpainted_slice (torch.Tensor): The outpainted image of size (512, 512) and range [0, 255]
        - new_pixel_spacing (float): The pixel spacing after FOV extension
    """

    dcm_img = data_utils.transform_dcm_to_HU(dcm)

    # Create body mask for truncated CT slice
    try:
        body_mask = data_utils.create_body_mask_slice(dcm_img)
    except:
        print("Body mask cannot be extracted. Skip slice.")

    lung_mask = data_utils.create_lung_mask_slice(dcm_img)
    body_mask = body_mask + lung_mask
    body_mask[body_mask >= 1] = 1
    body_mask = binary_fill_holes(body_mask)

    # Preprocess truncated CT slice
    preprocessed_ct_slice = data_utils.preprocess_slice(dcm_img, body_mask)
    preprocessed_ct_slice = torch.from_numpy(preprocessed_ct_slice).to(torch.float32)
    preprocessed_ct_slice = data_utils.normalize_rgb_img(preprocessed_ct_slice)

    if np.min(dcm_img) <= -1024:
        ppr_mask = data_utils.create_ppr_mask(dcm_img[:, :, None], cutoff=np.min(dcm_img))
        fov_mask = binary_erosion(binary_fill_holes(1 - ppr_mask), iterations=4)
    else:
        fov_mask, _ = data_utils.create_rfov_mask(img_dim=512, r_rfov=1)

    # Resize truncated CT slice and body mask from 512 to 256
    input_slice = data_utils.downsample_img(img=preprocessed_ct_slice, downsample_fn=downsample_fn)
    input_slice = input_slice.expand(3, -1, -1)
    fov_mask = data_utils.resize_mask(fov_mask, resized_dim=256)
    
    # Predict body bounding box for truncated CT slice
    body_bb_pred = run_body_bb_model(body_bb_model=body_bb_model, image=input_slice)

    # Extend truncated CT slice and body mask before outpainting
    extend_img, extend_fov_mask, ext_ratio, new_pixel_spacing = extend_fov(ct_slice=input_slice,
                                                                           body_mask=fov_mask,
                                                                           pixel_spacing=dcm.PixelSpacing[0],
                                                                           body_bb=body_bb_pred,
                                                                           center=center)

    # Resize the truncated CT slice, body mask, FOV mask back to 256
    extend_img = data_utils.downsample_img(img=extend_img, downsample_fn=downsample_fn)
    extend_fov_mask = data_utils.resize_mask(extend_fov_mask, resized_dim=256)
    trunc_slice = extend_img.expand(3, -1, -1)

    # Outpaint truncated CT slice
    with torch.no_grad():
        mask_image = trunc_slice[None, :, :, :].cuda()
        fov_mask = torch.from_numpy(extend_fov_mask).to(torch.float32).expand(3, -1, -1)
        mask = fov_mask[None, :, :, :].cuda()
        fake_image, _ = outpainting_model(mask_image, mask)
        outpainted_image = fake_image * (1 - mask) + mask_image * mask
        outpainted_image = outpainted_image.squeeze()[0].cpu()
        outpainted_image = torch.clamp(outpainted_image, -1, 1)
        rgb_outpainted_image = data_utils.transform_to_RGB(outpainted_image)
        rgb_outpainted_image = data_utils.upsample_img(img=rgb_outpainted_image, upsample_fn=upsample_fn)

    return rgb_outpainted_image, new_pixel_spacing

############################## Plotting functions ##############################

def plot_seg_results(untruncated_slice, truncated_slice, pred_slice,
                     untrunc_seg_mask, trunc_seg_mask, pred_seg_mask,
                     untrunc_seg_results, trunc_seg_results, pred_seg_results,
                     save=False, save_img_path=None, titles=None, colors=['#ff7f00', '#377eb8']):
    """
    Plots segmentation results.
    The first row displays the untruncated, truncated, and outpainted slices with segmentation masks overlayed on top of them.
    The second row displays their corresponding muscle and SAT areas.
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 15))
    boundaries = [0.5, 1.5, 2.5]
    
    # Plot the untruncated slice
    axes[0, 0].imshow(
        untruncated_slice,
        interpolation='bilinear',
        cmap='gray'
    )
    
    # Overlay the untruncated segmentation mask on the untruncated slice
    untrunc_seg_mask = untrunc_seg_mask.astype(float)
    untrunc_seg_mask[(untrunc_seg_mask != 2) & (untrunc_seg_mask != 1)] = np.nan
    cmap = colors.ListedColormap(colors)
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)

    axes[0, 0].imshow(
        untrunc_seg_mask,
        interpolation='none',
        cmap=cmap,
        norm=norm,
        alpha=0.5
    )
    
    if titles is None:
        axes[0, 0].set_title("Untruncated slice", fontsize=20)
    else:
        axes[0, 0].set_title(titles[0], fontsize=20)
    axes[0, 0].text(0, 600, f"Muscle (cm2): {untrunc_seg_results['Muscle']['area_cm2']:.3f}\nSAT (cm2): {untrunc_seg_results['SAT']['area_cm2']:.3f}", fontsize=15)
    
    # Plot the truncated slice
    axes[0, 1].imshow(
        truncated_slice,
        interpolation='bilinear',
        cmap='gray'
    )
    
    # Overlay the truncated segmentation mask on the truncated slice
    trunc_seg_mask = trunc_seg_mask.astype(float)
    trunc_seg_mask[(trunc_seg_mask != 2) & (trunc_seg_mask != 1)] = np.nan
    cmap = colors.ListedColormap(colors)
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)

    axes[0, 1].imshow(
        trunc_seg_mask,
        interpolation='none',
        cmap=cmap,
        norm=norm,
        alpha=0.5
    )
    
    if titles is None:
        axes[0, 1].set_title("Truncated slice", fontsize=20)
    else:
        axes[0, 1].set_title(titles[1], fontsize=20)
    axes[0, 1].text(0, 600, f"Muscle (cm2): {trunc_seg_results['Muscle']['area_cm2']:.3f}\nSAT (cm2): {trunc_seg_results['SAT']['area_cm2']:.3f}", fontsize=15)
    
    # Plot the predicted/outpainted slice
    axes[0, 2].imshow(
        pred_slice,
        interpolation='bilinear',
        cmap='gray'
    )
    
    # Overlay the predicted segmentation mask on the outpainted slice
    pred_seg_mask = pred_seg_mask.astype(float)
    pred_seg_mask[(pred_seg_mask != 2) & (pred_seg_mask != 1)] = np.nan
    cmap = colors.ListedColormap(colors)
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)

    axes[0, 2].imshow(
        pred_seg_mask,
        interpolation='none',
        cmap=cmap,
        norm=norm,
        alpha=0.5
    )
    
    if titles is None:
        axes[0, 2].set_title("Outpainted slice", fontsize=20)
    else:
        axes[0, 2].set_title(titles[2], fontsize=20)
    axes[0, 2].text(0, 600, f"Muscle (cm2): {pred_seg_results['Muscle']['area_cm2']:.3f}\nSAT (cm2): {pred_seg_results['SAT']['area_cm2']:.3f}", fontsize=15)
    
    # Bar plot
    categories = ['Muscle', 'SAT']
    untrunc_res = [untrunc_seg_results["Muscle"]["area_cm2"], untrunc_seg_results["SAT"]["area_cm2"]]
    trunc_res = [trunc_seg_results["Muscle"]["area_cm2"], trunc_seg_results["SAT"]["area_cm2"]]
    pred_res = [pred_seg_results["Muscle"]["area_cm2"], pred_seg_results["SAT"]["area_cm2"]]
    
    axes[1, 0].bar(categories, untrunc_res, facecolor="none", linestyle="dashed", edgecolor=colors, linewidth=3)

    axes[1, 1].bar(categories, trunc_res, color=colors, alpha=0.7)
    axes[1, 1].bar(categories, untrunc_res, facecolor="none", linestyle="dashed", edgecolor=colors, linewidth=3)
    axes[1, 1].set_title('Before')

    axes[1, 2].bar(categories, pred_res, color=colors, alpha=0.7)
    axes[1, 2].bar(categories, untrunc_res, facecolor='none', linestyle="dashed", edgecolor=colors, linewidth=3)
    axes[1, 2].set_title('After')

    max_value = max(*untrunc_res, *trunc_res, *pred_res)

    axes[1, 0].set_ylim(0, max_value + 10)
    axes[1, 1].set_ylim(0, max_value + 10)
    axes[1, 2].set_ylim(0, max_value + 10)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save_img_path)
    else:
        plt.show()
    
    plt.close()

def plot_truncated_and_outpainted_slices(rgb_trunc_slice, rgb_outpainted_slice, title1, title2, save=False, save_img_path=None):
    """ Plots truncated and outpainted slices side-by-side. """
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    
    axes[0].imshow(rgb_trunc_slice, cmap="gray")
    axes[0].set_title(title1, fontsize=20)
    axes[0].axis("off")
    
    axes[1].imshow(rgb_outpainted_slice, cmap="gray")
    axes[1].set_title(title2, fontsize=20)
    axes[1].axis("off")
    
    plt.tight_layout()

    if save:
        plt.savefig(save_img_path)
    else:
        plt.show()

    plt.close()

def plot_truncated_and_outpainted_slices_and_segmasks(rgb_trunc_slice, rgb_outpainted_slice,
                                                      trunc_seg_mask, outpainted_seg_mask,
                                                      save=False, save_img_path=None):
    """ Plots truncated and outpainted slices side-by-side on the first row and their corresponding segmentation masks on the second row. """
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    color_list = ['#ff7f00', '#377eb8']
    boundaries = [0.5, 1.5, 2.5]

    axes[0, 0].imshow(rgb_trunc_slice, cmap="gray")
    axes[0, 0].set_title("Truncated", fontsize=20)
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(rgb_outpainted_slice.cpu(), cmap="gray")
    axes[0, 1].set_title("CT-Palette", fontsize=20)
    axes[0, 1].axis("off")
    
    # ----- Plot the truncated slice -----
    axes[1, 0].imshow(
        rgb_trunc_slice,
        interpolation='bilinear',
        cmap='gray'
    )
    
    trunc_seg_mask = trunc_seg_mask.astype(float)
    trunc_seg_mask[(trunc_seg_mask != 2) & (trunc_seg_mask != 1)] = np.nan
    cmap = colors.ListedColormap(color_list)
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    
    axes[1, 0].imshow(
        trunc_seg_mask,
        interpolation='none',
        cmap=cmap,
        norm=norm,
        alpha=0.5
    )
    
    axes[1, 0].axis("off")

    # ----- Plot the outpainted slice by Palette -----
    axes[1, 1].imshow(
        rgb_outpainted_slice,
        interpolation='bilinear',
        cmap='gray'
    )
    
    # Overlay the predicted segmentation mask on the outpainted slice
    outpainted_seg_mask = outpainted_seg_mask.astype(float)
    outpainted_seg_mask[(outpainted_seg_mask != 2) & (outpainted_seg_mask != 1)] = np.nan
    cmap = colors.ListedColormap(color_list)
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    
    axes[1, 1].imshow(
        outpainted_seg_mask,
        interpolation='none',
        cmap=cmap,
        norm=norm,
        alpha=0.5
    )
    
    axes[1, 1].axis("off")
    
    plt.tight_layout()

    if save:
        plt.savefig(save_img_path)
    else:
        plt.show()

    plt.close()
   
def plot_truncated_and_outpainted_slices_and_segmasks_and_areas(rgb_trunc_slice, rgb_outpainted_slice,
                                                                trunc_seg_mask, outpainted_seg_mask,
                                                                trunc_seg_results, outpainted_seg_results,
                                                                save=False, save_img_path=None):
    """ Plots truncated and outpainted slices side-by-side on the first row and their corresponding segmentation masks on the second row. """
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    color_list = ['#ff7f00', '#377eb8']
    boundaries = [0.5, 1.5, 2.5]

    axes[0, 0].imshow(rgb_trunc_slice, cmap="gray")
    axes[0, 0].set_title("Truncated", fontsize=20)
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(rgb_outpainted_slice.cpu(), cmap="gray")
    axes[0, 1].set_title("CT-Palette", fontsize=20)
    axes[0, 1].axis("off")
    
    # ----- Plot the truncated slice -----
    axes[1, 0].imshow(
        rgb_trunc_slice,
        interpolation='bilinear',
        cmap='gray'
    )
    
    trunc_seg_mask = trunc_seg_mask.astype(float)
    trunc_seg_mask[(trunc_seg_mask != 2) & (trunc_seg_mask != 1)] = np.nan
    cmap = colors.ListedColormap(color_list)
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    
    axes[1, 0].imshow(
        trunc_seg_mask,
        interpolation='none',
        cmap=cmap,
        norm=norm,
        alpha=0.5
    )
    
    axes[1, 0].axis("off")
    axes[1, 0].text(0, 550, f"Muscle (cm2): {trunc_seg_results['Muscle']:.3f}\nSAT (cm2): {trunc_seg_results['SAT']:.3f}", fontsize=15)

    # ----- Plot the outpainted slice by Palette -----
    axes[1, 1].imshow(
        rgb_outpainted_slice,
        interpolation='bilinear',
        cmap='gray'
    )
    
    # Overlay the predicted segmentation mask on the outpainted slice
    outpainted_seg_mask = outpainted_seg_mask.astype(float)
    outpainted_seg_mask[(outpainted_seg_mask != 2) & (outpainted_seg_mask != 1)] = np.nan
    cmap = colors.ListedColormap(color_list)
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    
    axes[1, 1].imshow(
        outpainted_seg_mask,
        interpolation='none',
        cmap=cmap,
        norm=norm,
        alpha=0.5
    )
    
    axes[1, 1].axis("off")
    axes[1, 1].text(0, 550, f"Muscle (cm2): {outpainted_seg_results['Muscle']:.3f}\nSAT (cm2): {outpainted_seg_results['SAT']:.3f}", fontsize=15)
    
    plt.tight_layout()

    if save:
        plt.savefig(save_img_path)
    else:
        plt.show()

    plt.close()