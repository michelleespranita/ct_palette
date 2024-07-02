""" This script produces FOV masks used for synthetic truncation of the val and test sets,
and the corresponding DFOV-cropped truncated slices, ground-truth body bounding boxes, and zoom factors. """

import numpy as np
import pandas as pd
import torch

import os
import math
import random
from tqdm import tqdm

from ctpalette.paths import *
import ctpalette.data.data_utils as data_utils

NUM_SAMPLES_PER_SEV_LEVEL = 100
NUM_TIMES_SYNTHETIC_TRUNCATION = 10
NUM_SAMPLES_PER_BATCH = 250

if __name__ == "__main__":
    data_utils.mkdir_p(dfov_cropped_truncated_slices_dir)
    data_utils.mkdir_p(fov_mask_slices_dir)
    data_utils.mkdir_p(gt_body_bb_dir)

    create_mask_engine = data_utils.CreateMask(trunc_severity="val_test")

    # 1. Read val and test imgs from val.csv and test.csv
    val_df = pd.read_csv(os.path.join(data_split_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(data_split_dir, "test.csv"))
    val_imgs = list(val_df["path"])
    test_imgs = list(test_df["path"])
    val_test_untruncated_slices_files = val_imgs + test_imgs

    print(f"There are {len(val_imgs)} images in the val set.")
    print(f"There are {len(test_imgs)} images in the test set.")
    print(f"There are {len(val_test_untruncated_slices_files)} untruncated slices to do synthetic truncation on.")

    # 2. Divide the untruncated slices into batches to avoid out-of-memory error
    num_batches = math.ceil(len(val_test_untruncated_slices_files) / NUM_SAMPLES_PER_BATCH)
    print(f"{num_batches} batches to process!")
    batches_indices = [[i*NUM_SAMPLES_PER_BATCH, (i+1)*NUM_SAMPLES_PER_BATCH] if i < num_batches - 1 else [i*NUM_SAMPLES_PER_BATCH, len(val_test_untruncated_slices_files)] for i in range(num_batches)]

    # 3. For each untruncated slice, generate NUM_TIMES_SYNTHETIC_TRUNCATION synthetically truncated slices.
    # At the end, randomly sample NUM_SAMPLES_PER_SEV_LEVEL truncated slices for each severity level.
    for batch_idx in range(num_batches):
        batch_indices = batches_indices[batch_idx]
        num_samples_in_batch = batch_indices[1] - batch_indices[0]
        print(f"Processing {num_samples_in_batch} untruncated slices, from index {batch_indices[0]} to {batch_indices[1]}")

        trunc_severity_level_dict = {"trace": [], "mild": [], "moderate": [], "severe": []}

        for untrunc_slice_file in tqdm(val_test_untruncated_slices_files[batch_indices[0]:batch_indices[1]]):
            file_name = os.path.basename(untrunc_slice_file).replace('.dcm', '.npy')
            untrunc_slice = torch.from_numpy(np.load(os.path.join(untruncated_slices_dir, file_name))).to(data_utils.DEVICE)
            body_mask_slice = torch.from_numpy(np.load(os.path.join(body_mask_slices_dir, file_name))).to(data_utils.DEVICE)
            lung_mask_slice = torch.from_numpy(np.load(os.path.join(lung_mask_slices_dir, file_name))).to(data_utils.DEVICE)

            # Generate NUM_TIMES_SYNTHETIC_TRUNCATION synthetically truncated slices. If lung exceeds image border, discard the truncated slice.
            for i in range(NUM_TIMES_SYNTHETIC_TRUNCATION):
                ct_slice_dict = create_mask_engine(untrunc_slice)
                rfov_mask = ct_slice_dict["rfov_mask"]
                dfov_mask = ct_slice_dict["dfov_mask"]
                truncated_ct_slice = ct_slice_dict["truncated_ct_slice"]
                dfov_cropped_truncated_ct_slice = ct_slice_dict["dfov_cropped_truncated_ct_slice"]
                zoom_factor = ct_slice_dict["zoom_factor"]
                fov_mask = rfov_mask * dfov_mask

                # Get PPR mask of the truncated slice
                truncated_fov_mask = data_utils.truncate_mask(fov_mask, rfov_mask, dfov_mask)
                truncated_ppr_mask = 1 - truncated_fov_mask

                # Get body mask of the truncated slice
                truncated_body_mask = data_utils.truncate_mask(body_mask_slice, rfov_mask, dfov_mask)

                # Get TCI value for truncated slice
                tci_val = data_utils.get_tci_value(truncated_body_mask, truncated_ppr_mask)
                sev_level = data_utils.classify_tci_to_severity_level(tci_val)
                if sev_level == "no_trunc":
                    print("No truncation. Discard.")
                    continue

                # Get lung mask of the truncated slice -> Check whether the lung region exceeds the image border (If yes, discard synthetically generated truncated slice)
                truncated_lung_mask = data_utils.truncate_mask(lung_mask_slice, rfov_mask, dfov_mask)
                if data_utils.check_if_lung_region_cropped(truncated_lung_mask, truncated_fov_mask):
                    print("Lung exceeds image border. Discard.")
                    continue

                # Get groundtruth body bounding box
                gt_body_bb = data_utils.extract_groundtruth_body_bb(body_mask_slice, dfov_mask, zoom_factor)
                gt_body_bb = gt_body_bb.tolist()
                
                # Save sample to its severity level
                trunc_severity_level_dict[sev_level].append({"dfov_cropped_truncated_slice": dfov_cropped_truncated_ct_slice,
                                                             "zoom_factor": zoom_factor,
                                                             "fov_mask": fov_mask,
                                                             "gt_body_bb": gt_body_bb,
                                                             "lung_mask": truncated_lung_mask,
                                                             "untrunc_slice_file": file_name})

        # 4. For each severity level, randomly sample NUM_SAMPLES_PER_SEV_LEVEL samples and save them
        for sev_level, samples in trunc_severity_level_dict.items():
            if len(samples) >= NUM_SAMPLES_PER_SEV_LEVEL:
                rand_samples = random.sample(samples, NUM_SAMPLES_PER_SEV_LEVEL)
            elif len(samples) > 0 and len(samples) < NUM_SAMPLES_PER_SEV_LEVEL:
                rand_samples = samples
            else:
                rand_samples = []

            print(f"There are {len(rand_samples)} samples for severity level {sev_level}")

            for i, sample in enumerate(rand_samples):
                dfov_cropped_truncated_ct_slice = sample["dfov_cropped_truncated_slice"]
                zoom_factor = sample["zoom_factor"]
                fov_mask = sample["fov_mask"]
                gt_body_bb = sample["gt_body_bb"]
                lung_mask = sample["lung_mask"]
                untrunc_slice_file = sample["untrunc_slice_file"]

                # Save DFoV cropped truncated CT slice, FoV mask, zoom factor, and groundtruth body BB (of truncated CT slice)
                np.save(os.path.join(dfov_cropped_truncated_slices_dir, f"{untrunc_slice_file.replace('.npy', '')}_trunc{i}_{sev_level}.npy"), dfov_cropped_truncated_ct_slice.cpu().numpy())
                
                np.save(os.path.join(fov_mask_slices_dir, f"{untrunc_slice_file.replace('.npy', '')}_trunc{i}_{sev_level}.npy"), fov_mask.cpu().numpy())

                # with open(os.path.join(zoom_factor_dir, f"{untrunc_slice_file.replace('.npy', '')}_trunc{i}_{sev_level}.txt"), "w") as f:
                #     f.write(str(zoom_factor))

                with open(os.path.join(gt_body_bb_dir, f"{untrunc_slice_file.replace('.npy', '')}_trunc{i}_{sev_level}.txt"), "w") as f:
                    gt_body_bb = [str(c) for c in gt_body_bb]
                    gt_body_bb = ','.join(gt_body_bb)
                    f.write(gt_body_bb)