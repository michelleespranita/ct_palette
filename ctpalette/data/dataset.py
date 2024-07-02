""" This script contains torch datasets for training, validation, and testing
the body bounding box detector and image outpainting models. """

import torch.utils.data as data
import os
import re
import torch
import numpy as np
import pandas as pd

from ctpalette.paths import *
import ctpalette.data.data_utils as data_utils

RESIZED_IMG_SIZE = 256

############################## Body bounding box detector datasets ##############################

class BodyBBDataset_Train(data.Dataset):
    def __init__(self, aug=True, overfit=False):
        self.untrunc_slices_files = list(pd.read_csv(os.path.join(data_split_dir, "train.csv"))["path"])
        self.untrunc_slices_files = [os.path.basename(f).replace('.dcm', '.npy') for f in self.untrunc_slices_files]

        if overfit:
            self.untrunc_slices_files = [self.untrunc_slices_files[0]]

        self.aug = aug
        self.num_channels = 3

        self.precision = torch.float32
        self.device = data_utils.DEVICE

        self.create_mask_engine = data_utils.CreateMask(trunc_severity="train")
        
        print(f"Length of dataset: {len(self.untrunc_slices_files)}")
    
    def __len__(self):
        return len(self.untrunc_slices_files)

    def __getitem__(self, idx):
        untrunc_slice_file = self.untrunc_slices_files[idx]
        untrunc_slice = data_utils.downsample_img(torch.from_numpy(np.load(os.path.join(untruncated_slices_dir, untrunc_slice_file))).to(self.precision).to(self.device))
        body_mask_slice = torch.from_numpy(data_utils.resize_mask(np.load(os.path.join(body_mask_slices_dir, untrunc_slice_file)))).to(self.precision).to(self.device)
        
        filename = untrunc_slice_file.replace(".npy", "")
        
        ct_slice_dict = self.create_mask_engine(untrunc_slice)
        rfov_mask = ct_slice_dict["rfov_mask"]
        dfov_mask = ct_slice_dict["dfov_mask"]
        fov_mask = rfov_mask * dfov_mask
        dfov_cropped_truncated_ct_slice = ct_slice_dict["dfov_cropped_truncated_ct_slice"]
        zoom_factor = ct_slice_dict["zoom_factor"]

        # Get body mask of the truncated slice
        truncated_body_mask = body_mask_slice * fov_mask

        # Get PPR mask of the truncated slice
        ppr_mask = 1 - fov_mask
        tci_val = data_utils.get_tci_value(truncated_body_mask, ppr_mask)
        sev_level = data_utils.classify_tci_to_severity_level(tci_val)

        # 3. GT body BB extraction
        gt_body_bb = data_utils.extract_groundtruth_body_bb(body_mask_slice, dfov_mask, zoom_factor)

        # 4. Data augmentation
        if self.aug:
            # Augment the truncated slice
            aug_dfov_cropped_trunc_slice, transform_params = data_utils.augment_slice(dfov_cropped_truncated_ct_slice.expand(1, -1, -1))
            aug_dfov_cropped_trunc_slice = aug_dfov_cropped_trunc_slice.squeeze()
            aug_dfov_cropped_trunc_slice = data_utils.normalize_rgb_img(aug_dfov_cropped_trunc_slice)
            aug_dfov_cropped_trunc_slice = aug_dfov_cropped_trunc_slice.expand(self.num_channels, -1, -1)
            aug_fov_mask = data_utils.augment_mask(fov_mask.expand(1, -1, -1), transform_params).squeeze()
            aug_trunc_body_mask = data_utils.augment_mask(truncated_body_mask.expand(1, -1, -1), transform_params).squeeze()
            tci_val = data_utils.get_tci_value(aug_trunc_body_mask, 1 - aug_fov_mask)
            sev_level = data_utils.classify_tci_to_severity_level(tci_val)

            # Find GT body BB of truncated slice after augmentation
            gt_body_bb_coords = data_utils.transform_body_bb_into_coords(gt_body_bb)
            aug_gt_body_bb_coords = data_utils.transform_bounding_box(
                gt_body_bb_coords,
                scale_factor=transform_params[2],
                rotation_angle=-transform_params[0],
                translation_x=transform_params[1][1],
                translation_y=transform_params[1][0]
            )
            
            # Axis-align the GT body BB of truncated slice after augmentation
            aug_gt_body_bb = data_utils.axis_align_bounding_box(aug_gt_body_bb_coords)

            return {
                "dfov_cropped_truncated_slice": aug_dfov_cropped_trunc_slice.cpu(),
                "body_bb_coords": aug_gt_body_bb_coords.cpu(), # 8-number body BB coords
                "body_bb": aug_gt_body_bb.cpu(), # axis-aligned (4-number) body coords
                "tci_val": tci_val,
                "sev_level": sev_level, # Augmentation can cause further truncation, but we assume the severity level stays the same
                "filename": filename
            }
        
        dfov_cropped_truncated_ct_slice = data_utils.normalize_rgb_img(dfov_cropped_truncated_ct_slice)
        dfov_cropped_truncated_ct_slice = dfov_cropped_truncated_ct_slice.expand(self.num_channels, -1, -1)
        
        return {
            "dfov_cropped_truncated_slice": dfov_cropped_truncated_ct_slice.cpu(),
            "body_bb": torch.Tensor(gt_body_bb).cpu(), # axis-aligned (4-number) body coords
            "tci_val": tci_val,
            "sev_level": sev_level,
            "filename": filename
        }
    
class BodyBBDataset_Val_Test(data.Dataset):
    def __init__(self, data_split):
        assert data_split in ["val", "test"]

        dfov_cropped_trunc_slices_files_tmp = list(pd.read_csv(os.path.join(data_split_dir, f"{data_split}.csv"))["path"])
        dfov_cropped_trunc_slices_files_tmp = [os.path.basename(f).replace('.dcm', '') for f in dfov_cropped_trunc_slices_files_tmp]
        self.dfov_cropped_trunc_slices_files = []
        for f in dfov_cropped_trunc_slices_files_tmp:
            files = [k for k in os.listdir(dfov_cropped_truncated_slices_dir) if k.startswith(f)]
            self.dfov_cropped_trunc_slices_files += files
        
        self.num_channels = 3
        self.precision = torch.float32

        print(f"Length of dataset: {len(self.dfov_cropped_trunc_slices_files)}")
        
    def __len__(self):
        return len(self.dfov_cropped_trunc_slices_files)

    def __getitem__(self, idx):
        dfov_cropped_trunc_slice = data_utils.downsample_img(torch.from_numpy(np.load(os.path.join(dfov_cropped_truncated_slices_dir, self.dfov_cropped_trunc_slices_files[idx]))).to(self.precision))
        dfov_cropped_trunc_slice = data_utils.normalize_rgb_img(dfov_cropped_trunc_slice)
        dfov_cropped_trunc_slice = dfov_cropped_trunc_slice.expand(self.num_channels, -1, -1)

        with open(os.path.join(gt_body_bb_dir, self.dfov_cropped_trunc_slices_files[idx].replace(".npy", ".txt")), "r") as f:
            body_bb = f.read()
            body_bb = torch.Tensor([float(c) for c in body_bb.split(',')])

        factor = 512 / data_utils.RESIZED_IMG_SIZE
        body_bb = body_bb / factor

        filename = self.dfov_cropped_trunc_slices_files[idx].replace(".npy", "")

        # Extract sev_level
        sev_level = re.search("_(trace|mild|moderate|severe).npy", self.dfov_cropped_trunc_slices_files[idx]).group(1)

        return {
            "dfov_cropped_truncated_slice": dfov_cropped_trunc_slice.cpu(),
            "body_bb": body_bb.cpu(),
            "sev_level": sev_level,
            "filename": filename
        }

############################## Image outpainting model datasets ##############################

class ImageCompletionDataset_Train(data.Dataset):
    def __init__(self, aug=False, overfit=False):        
        self.untrunc_slices_files = list(pd.read_csv(os.path.join(data_split_dir, "train.csv"))["path"])
        self.untrunc_slices_files = [os.path.basename(f).replace('.dcm', '.npy') for f in self.untrunc_slices_files]
        
        if overfit:
            self.untrunc_slices_files = [self.untrunc_slices_files[0]]
        
        self.aug = aug
        self.num_channels = 1

        self.precision = torch.float32
        self.device = data_utils.DEVICE

        self.create_mask_engine = data_utils.CreateMask(trunc_severity="train")

        print(f"Length of dataset: {len(self.untrunc_slices_files)}")
    
    def __len__(self):
        return len(self.untrunc_slices_files)

    def __getitem__(self, idx):
        untrunc_slice_file = self.untrunc_slices_files[idx]
        untrunc_slice = data_utils.downsample_img(torch.from_numpy(np.load(os.path.join(untruncated_slices_dir, untrunc_slice_file))).to(self.precision).to(self.device))
        body_mask_slice = torch.from_numpy(data_utils.resize_mask(np.load(os.path.join(body_mask_slices_dir, untrunc_slice_file)))).to(self.precision).to(self.device)

        filename = untrunc_slice_file.replace(".npy", "")

        ct_slice_dict = self.create_mask_engine(untrunc_slice)
        rfov_mask = ct_slice_dict["rfov_mask"].to(self.precision)
        dfov_mask = ct_slice_dict["dfov_mask"].to(self.precision)
        fov_mask = rfov_mask * dfov_mask

        # Calculate TCI value of the supposedly-truncated slice -> Check whether there is any truncation at all (If not, discard because we want a truncated slice)
        truncated_body_mask = body_mask_slice * fov_mask
        ppr_mask = 1 - fov_mask
        tci_val = data_utils.get_tci_value(truncated_body_mask, ppr_mask)
        sev_level = data_utils.classify_tci_to_severity_level(tci_val)

        # 3. Data augmentation
        aug = self.aug
        while aug:
            # Augment the untruncated slice, FoV mask, truncated body mask
            aug_untrunc_slice, transform_params = data_utils.augment_slice(untrunc_slice.expand(1, -1, -1))
            aug_untrunc_slice = aug_untrunc_slice.squeeze()
            aug_untrunc_slice = data_utils.normalize_rgb_img(aug_untrunc_slice)
            aug_fov_mask = data_utils.augment_mask(fov_mask.expand(1, -1, -1), transform_params).squeeze()
            aug_trunc_body_mask = data_utils.augment_mask(truncated_body_mask.expand(1, -1, -1), transform_params).squeeze()

            # Check if body BB of untruncated slice exceeds image border after augmentation
            body_bb = data_utils.extract_bb_from_mask(body_mask_slice)
            body_bb_coords = data_utils.transform_body_bb_into_coords(body_bb)
            aug_body_bb_coords = data_utils.transform_bounding_box(
                body_bb_coords,
                scale_factor=transform_params[2],
                rotation_angle=-transform_params[0],
                translation_x=transform_params[1][1],
                translation_y=transform_params[1][0]
            )
            aug_body_bb = data_utils.axis_align_bounding_box(aug_body_bb_coords)

            mask = data_utils.create_CT_Palette_mask(body_mask=aug_trunc_body_mask, body_bb=aug_body_bb)
            
            if not data_utils.check_bbb_exceeds_border(aug_body_bb_coords, img_size=256):
                aug = False

                # Calculate TCI value and severity level after augmentation
                tci_val = data_utils.get_tci_value(aug_trunc_body_mask, 1 - aug_fov_mask)
                sev_level = data_utils.classify_tci_to_severity_level(tci_val)

                aug_fov_mask = aug_fov_mask.expand(self.num_channels, -1, -1)
                aug_ppr_mask = 1 - aug_fov_mask
                aug_trunc_slice = aug_untrunc_slice * aug_fov_mask
                aug_cond_slice = aug_untrunc_slice * (1 - mask) + torch.randn_like(aug_untrunc_slice) * mask

                aug_untrunc_slice = aug_untrunc_slice.expand(self.num_channels, -1, -1)
                aug_untrunc_slice[aug_untrunc_slice == 0] = -1
                aug_trunc_slice[aug_trunc_slice == 0] = -1

                return {
                    "gt_image": aug_untrunc_slice.cpu(),
                    "mask": mask.cpu(),
                    "fov_mask": aug_ppr_mask.cpu(),
                    "mask_image": aug_trunc_slice.cpu(),
                    "cond_image": aug_cond_slice.cpu(),
                    "tci_val": tci_val,
                    "sev_level": sev_level,
                    "path": filename + ".png",
                    "body_bb_coords": aug_body_bb_coords,
                    "body_bb": aug_body_bb
                }

            else:
                continue
        
        body_bb = data_utils.extract_bb_from_mask(body_mask_slice)
        mask = data_utils.create_CT_Palette_mask(body_mask=truncated_body_mask, body_bb=body_bb)
        
        fov_mask = fov_mask.expand(self.num_channels, -1, -1)
        ppr_mask = 1 - fov_mask
        untrunc_slice = data_utils.normalize_rgb_img(untrunc_slice)
        untrunc_slice = untrunc_slice.expand(self.num_channels, -1, -1)
        trunc_slice = untrunc_slice * fov_mask
        cond_slice = untrunc_slice * (1 - mask) + torch.randn_like(untrunc_slice) * mask
        untrunc_slice[untrunc_slice == 0] = -1
        trunc_slice[trunc_slice == 0] = -1

        return {
            "gt_image": untrunc_slice.cpu(),
            "mask": mask.cpu(),
            "body_mask": body_mask_slice.cpu(),
            "fov_mask": ppr_mask.cpu(),
            "mask_image": trunc_slice.cpu(),
            "cond_image": cond_slice.cpu(),
            "tci_val": tci_val,
            "sev_level": sev_level,
            "path": filename + ".png",
            "body_bb": body_bb
        }
    
class ImageCompletionDataset_Val_Test(data.Dataset):
    def __init__(self, data_split, eval_type="small"):
        assert data_split in ["val", "test"]
        assert eval_type in ["overfit", "small", "full"]

        fov_mask_slices_files_tmp = list(pd.read_csv(os.path.join(data_split_dir, f"{data_split}.csv"))["path"])
        fov_mask_slices_files_tmp = [os.path.basename(f).replace('.dcm', '') for f in fov_mask_slices_files_tmp]
        self.fov_mask_slices_files = []
        for f in fov_mask_slices_files_tmp:
            files = [k for k in os.listdir(fov_mask_slices_dir) if k.startswith(f)]
            self.fov_mask_slices_files += files

        if eval_type == "overfit":
            self.fov_mask_slices_files = [self.fov_mask_slices_files[0]]
        elif eval_type == "small":
            self.fov_mask_slices_files = self.fov_mask_slices_files[0:8]
        
        self.num_channels = 1
        self.precision = torch.float32

        print(f"Length of dataset: {len(self.fov_mask_slices_files)}")

    def __len__(self):
        return len(self.fov_mask_slices_files)

    def __getitem__(self, idx):
        untrunc_slice_filename = re.search("(.*?)_trunc", self.fov_mask_slices_files[idx]).group(1) + ".npy"
        untrunc_slice = torch.from_numpy(np.load(os.path.join(untruncated_slices_dir, untrunc_slice_filename))).to(self.precision)
        untrunc_slice = data_utils.downsample_img(data_utils.normalize_rgb_img(untrunc_slice))
        body_mask_slice = torch.from_numpy(data_utils.resize_mask(np.load(os.path.join(body_mask_slices_dir, untrunc_slice_filename)))).to(self.precision)
        fov_mask_slice = torch.from_numpy(data_utils.resize_mask(np.load(os.path.join(fov_mask_slices_dir, self.fov_mask_slices_files[idx])))).to(self.precision)
        
        truncated_body_mask = body_mask_slice * fov_mask_slice
        body_bb = data_utils.extract_bb_from_mask(body_mask_slice)
        mask = data_utils.create_CT_Palette_mask(body_mask=truncated_body_mask, body_bb=body_bb)

        cond_slice = untrunc_slice * (1 - mask) + torch.randn_like(untrunc_slice) * mask
        
        mask = mask.expand(self.num_channels, -1, -1)
        untrunc_slice = untrunc_slice.expand(self.num_channels, -1, -1)
        fov_mask_slice = fov_mask_slice.expand(self.num_channels, -1, -1)
        ppr_mask_slice = 1 - fov_mask_slice
        cond_slice = cond_slice.expand(self.num_channels, -1, -1)
        trunc_slice = untrunc_slice * fov_mask_slice
        trunc_slice[trunc_slice == 0] = -1
        trunc_slice = trunc_slice.expand(self.num_channels, -1, -1)
        
        filename = self.fov_mask_slices_files[idx].replace(".npy", "")

        # Extract sev_level
        sev_level = re.search("_(trace|mild|moderate|severe).npy", self.fov_mask_slices_files[idx]).group(1)

        return {
            "gt_image": untrunc_slice.cpu(),
            "mask": mask.cpu(),
            "fov_mask": ppr_mask_slice.cpu(),
            "body_mask": body_mask_slice.cpu(),
            "mask_image": trunc_slice.cpu(),
            "cond_image": cond_slice.cpu(),
            "sev_level": sev_level,
            "path": filename + ".png",
            "body_bb": body_bb
        }