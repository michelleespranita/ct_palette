import os

MAIN_DATA_PATH = "/Users/michelleespranita/Documents/Master Thesis/MICCAI/ct_palette_test_data" # FILL THIS WITH YOURS!!!

############################### Data paths ##############################

# Input
data_split_dir = os.path.join(MAIN_DATA_PATH, "data_split") # Path to train.csv, val.csv, test.csv that contain paths to the original .dcm files


# Output of ctpalette/data_processing/prepare_data.py
untruncated_slices_dir = os.path.join(MAIN_DATA_PATH, "untruncated_slices") # Path to untruncated CT slices in .npy format
body_mask_slices_dir = os.path.join(MAIN_DATA_PATH, "body_mask") # Path to body masks in .npy format
lung_mask_slices_dir = os.path.join(MAIN_DATA_PATH, "lung_mask") # Path to lung masks in .npy format

# Output of ctpalette/data_processing/synthetic_truncation.py
dfov_cropped_truncated_slices_dir = os.path.join(MAIN_DATA_PATH, "dfov_cropped_truncated_slices") # Path to DFOV-cropped truncated slices in .npy format
gt_body_bb_dir = os.path.join(MAIN_DATA_PATH, "gt_body_bb") # Path to ground-truth body bounding boxes in .txt format
fov_mask_slices_dir = os.path.join(MAIN_DATA_PATH, "fov_mask") # Path to FOV masks in .npy format

############################## Training log paths ##############################

MLFLOW_DIR = os.path.join(MAIN_DATA_PATH, "mlruns") # where results from training the body bounding box detector are saved

# Note: To configure the path to the results from training the image outpainting model, go to config/outpainting_model.json.

############################## Model config paths ##############################

config_dir = "./config"

body_bb_train_config_path = os.path.join(config_dir, "body_bb.yaml") # Path to the .yaml config file for training the body bounding box detector
image_outpainting_config_path = os.path.join(config_dir, "outpainting_model.json") # Path to the .json config file for training and inference of the image outpainting model

############################## Model paths ##############################

# IMPORTANT FOR INFERENCE
model_dir = "./models"

body_bb_model_path = os.path.join(model_dir, "body_bb_detector/body_bb.pth") # Path to the trained body bounding box detector for inference
image_outpainting_model_path = os.path.join(model_dir, "image_outpainting_model/image_outpainting_model.pth") # Path to the trained image outpainting model for inference
seg_model_path = "" # Path to the trained body composition segmentation model used for multiple inference