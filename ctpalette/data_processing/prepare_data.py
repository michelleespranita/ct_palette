""" This script extracts body masks and lung masks from untruncated CT slices
and saves them as numpy files. """

import numpy as np
import pandas as pd
from pydicom import dcmread

import os
from tqdm import tqdm

from ctpalette.paths import *
import ctpalette.data.data_utils as data_utils

OVERWRITE = True

if __name__ == "__main__": 
    data_utils.mkdir_p(untruncated_slices_dir)
    data_utils.mkdir_p(body_mask_slices_dir)
    data_utils.mkdir_p(lung_mask_slices_dir)

    train_imgs = list(pd.read_csv(os.path.join(data_split_dir, "train.csv"))["path"])
    val_imgs = list(pd.read_csv(os.path.join(data_split_dir, "val.csv"))["path"])
    test_imgs = list(pd.read_csv(os.path.join(data_split_dir, "test.csv"))["path"])
    all_imgs = train_imgs + val_imgs + test_imgs

    # Save the untruncated slices, body masks, and lung masks as numpy files
    for dcm_img in tqdm(all_imgs):
        try:
            # Preprocess the slice and save
            dcm_img_name = os.path.basename(dcm_img)
            dcm = dcmread(dcm_img)
            dcm_img = data_utils.transform_dcm_to_HU(dcm) 
            body_mask_slice = data_utils.create_body_mask_slice(dcm_img)
            body_mask_slice = body_mask_slice.astype(np.float32)
            preprocessed_ct_slice = data_utils.preprocess_slice(dcm_img, body_mask_slice) 
            preprocessed_ct_slice = preprocessed_ct_slice.astype(np.float32) # image is of range [0, 255]

            data_utils.save_numpy(os.path.join(untruncated_slices_dir, f"{dcm_img_name.replace('.dcm', '')}.npy"), preprocessed_ct_slice, overwrite=OVERWRITE)

            # Save the slice's body mask
            data_utils.save_numpy(os.path.join(body_mask_slices_dir, f"{dcm_img_name.replace('.dcm', '')}.npy"), body_mask_slice, overwrite=OVERWRITE)

            # Generate and save the slice's lung mask
            lung_mask_slice = data_utils.create_lung_mask(dcm_img[np.newaxis, :, :])
            lung_mask_slice = lung_mask_slice.astype(np.float32).squeeze()
            data_utils.save_numpy(os.path.join(lung_mask_slices_dir, f"{dcm_img_name.replace('.dcm', '')}.npy"), lung_mask_slice, overwrite=OVERWRITE)
        except:
            print(f"Error processing {dcm_img}")
