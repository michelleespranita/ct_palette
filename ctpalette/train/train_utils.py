""" This script contains useful functions for model training. """

import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from datetime import datetime
import os
import shutil
import tempfile
import random
import logging
from tqdm import tqdm
import mlflow

from ctpalette.models.model import BodyBBModel
from ctpalette.paths import *
import ctpalette.data.data_utils as data_utils

def pad_date(date):
    if len(str(date)) == 1:
        return '0' + str(date)
    return str(date)

def get_current_time():
    current_time = datetime.now()
    year = current_time.year
    month = pad_date(current_time.month)
    day = pad_date(current_time.day)
    hour = pad_date(current_time.hour)
    minute = pad_date(current_time.minute)
    return f"{year}_{month}_{day}_{hour}_{minute}"

def get_current_run_id():
    return mlflow.active_run().info.run_id

def get_current_experiment_id():
    return mlflow.active_run().info.experiment_id

def delete_model_ckpt_in_current_run():
    current_experiment_id = get_current_experiment_id()
    current_run_id = get_current_run_id()
    if os.path.exists(os.path.join(MLFLOW_DIR, current_experiment_id, current_run_id, "checkpoints")):
        shutil.rmtree(os.path.join(MLFLOW_DIR, current_experiment_id, current_run_id, "checkpoints"))

def load_model_ckpt(model, model_weights_path):
    model.load_state_dict(torch.load(model_weights_path, map_location="cpu"))

def save_model_ckpt(model, epoch):
    current_experiment_id = get_current_experiment_id()
    current_run_id = get_current_run_id()
    current_date = get_current_time()
    data_utils.mkdir_p(os.path.join(MLFLOW_DIR, current_experiment_id, current_run_id, "checkpoints"))
    torch.save(model.state_dict(), os.path.join(MLFLOW_DIR, current_experiment_id, current_run_id, "checkpoints", f"{current_date}_epoch{epoch}.pth"))

def summarize_model(model):
    layers = [(name if len(name) > 0 else 'TOTAL', str(module.__class__.__name__), sum(np.prod(p.shape) for p in module.parameters())) for name, module in model.named_modules()]
    layers.append(layers[0])
    del layers[0]

    columns = [
        [" ", list(map(str, range(len(layers))))],
        ["Name", [layer[0] for layer in layers]],
        ["Type", [layer[1] for layer in layers]],
        ["Params", [layer[2] for layer in layers]]
    ]

    n_rows = len(columns[0][1])
    n_cols = 1 + len(columns)

    col_widths = []
    for c in columns:
        col_width = max(len(str(a)) for a in c[1]) if n_rows else 0
        col_width = max(col_width, len(c[0]))
        col_widths.append(col_width)
    
    s = "{:<{}}"
    total_width = sum(col_widths) + 3 * n_cols
    header = [s.format(c[0], l) for c, l in zip(columns, col_widths)]

    summary = " | ".join(header) + "\n" + "-" * total_width
    for i in range(n_rows):
        line = []
        for c, l in zip(columns, col_widths):
            line.append(s.format(str(c[1][i]), l))
        summary += "\n" + " | ".join(line)
    
    return summary

def total_params(model):
    for name, module in model.named_modules():
        if len(name) == 0:
            return sum(np.prod(p.shape) for p in module.parameters())

def model_param_types(model):
    for name, param in model.named_parameters():
        print(f"Parameter: {name}, Data type: {param.dtype}")

############################## GIoU ##############################

def get_bb_area(bb):
    area = (bb[:, 2] - bb[:, 0]) * (bb[:, 3] - bb[:, 1])
    return area

def intersection(bb_gt, bb_pred, device):
    batch_size = bb_gt.shape[0]
    
    # Left top
    left_top_bb_gt, left_top_bb_pred = bb_gt[:, 0:2], bb_pred[:, 0:2]
    left_top_bb_max = torch.max(left_top_bb_gt, left_top_bb_pred)
    
    # Right bottom
    right_bottom_bb_gt, right_bottom_bb_pred = bb_gt[:, 2:], bb_pred[:, 2:]
    right_bottom_bb_min = torch.min(right_bottom_bb_gt, right_bottom_bb_pred)
    
    # Intersection BB
    intersection_bb = torch.cat([left_top_bb_max, right_bottom_bb_min], dim=1)
    
    heights = torch.maximum(intersection_bb[:, 3] - intersection_bb[:, 1], torch.tensor([0] * batch_size, device=device))
    widths = torch.maximum(intersection_bb[:, 2] - intersection_bb[:, 0], torch.tensor([0] * batch_size, device=device))
    
    return heights * widths

def union(bb_gt, bb_pred, device):
    area_gt = get_bb_area(bb_gt)
    area_pred = get_bb_area(bb_pred)
    return area_gt + area_pred - intersection(bb_gt, bb_pred, device)

def c(bb_gt, bb_pred, device):
    batch_size = bb_gt.shape[0]
    
    # Left top
    left_top_bb_gt, left_top_bb_pred = bb_gt[:, 0:2], bb_pred[:, 0:2]
    left_top_bb_min = torch.min(left_top_bb_gt, left_top_bb_pred)
    
    # Right bottom
    right_bottom_bb_gt, right_bottom_bb_pred = bb_gt[:, 2:], bb_pred[:, 2:]
    right_bottom_bb_max = torch.max(right_bottom_bb_gt, right_bottom_bb_pred)
    
    # Union BB
    union_bb = torch.cat([left_top_bb_min, right_bottom_bb_max], dim=1)
    
    heights = torch.maximum(union_bb[:, 3] - union_bb[:, 1], torch.tensor([0] * batch_size, device=device))
    widths = torch.maximum(union_bb[:, 2] - union_bb[:, 0], torch.tensor([0] * batch_size, device=device))
    
    return heights * widths

def iou(bb_gt, bb_pred, device):
    U = union(bb_gt, bb_pred, device)
    if U == 0:
        return 0
    return intersection(bb_gt, bb_pred) / U

def giou(bb_gt, bb_pred, device):
    I = intersection(bb_gt, bb_pred, device)
    U = union(bb_gt, bb_pred, device)
    C = c(bb_gt, bb_pred, device)
    # iou_term = (I / U) if U > 0 else 0
    iou_term = torch.where(U > 0, I / U, 0)
    # giou_term = ((C - U) / C) if C > 0 else 0
    giou_term = torch.where(C > 0, (C - U) / C, 0)
    return iou_term - giou_term

############################## Train functions for body BB detector ##############################

def train_body_bb_model(model, train_dataloader, val_dataloader, config):
    data_utils.mkdir_p(MLFLOW_DIR)
    mlflow.set_tracking_uri(MLFLOW_DIR)
    if mlflow.get_experiment_by_name("body_bb_model") is not None:
        experiment = mlflow.set_experiment("body_bb_model")
    else:
        experiment_id = mlflow.create_experiment("body_bb_model")
        experiment = mlflow.set_experiment("body_bb_model")
    
    with mlflow.start_run():

        # Log model architecture
        with tempfile.NamedTemporaryFile(mode='w', delete=True) as temp_file:
            model_summary = summarize_model(model)
            temp_file.write(model_summary)
            temp_file.flush()

            temp_file_path = temp_file.name
            mlflow.log_artifact(temp_file_path)

        # Log hyperparams
        mlflow.log_params(config)

        # Set device
        device = model.device

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        mse_loss_fn = nn.MSELoss(reduction="none")

        best_avg_val_loss = np.inf

        model.train()

        global iteration
        iteration = 0
        num_epochs_no_improve = 0
        early_stopping = False

        for epoch in tqdm(range(config["max_epochs"])):
            #################### Training ####################
            total_train_loss = 0. # Accumulates the average loss of each train batch

            for train_batch_idx, train_batch in enumerate(train_dataloader):
                iteration = epoch * len(train_dataloader) + train_batch_idx

                # Forward pass
                truncated_slices, body_bb_gt = train_batch["dfov_cropped_truncated_slice"].to(device), train_batch["body_bb"].to(device)
                truncated_slices = truncated_slices.to(torch.float32)
                body_bb_pred = model(truncated_slices)

                # Calculate loss
                train_mse_loss = mse_loss_fn(body_bb_gt, body_bb_pred).mean(dim=1)
                train_giou = giou(body_bb_gt, body_bb_pred, device=device)
                train_loss = train_mse_loss + config["lambda"] * train_giou
                train_giou = torch.mean(train_giou)
                train_loss = torch.mean(train_loss)
                total_train_loss += train_loss
                print(f'[{epoch+1:03d}/{train_batch_idx+1:03d}] train_loss: {train_loss:.6f} (batch)') # prints the average train loss of a batch

                # Backprop
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                # Logging (GIoU, train_loss)
                mlflow.log_metric(key="train_giou", value=train_giou, step=iteration)
                mlflow.log_metric(key="train_loss", value=train_loss, step=iteration)
            
            # Calculate the average train loss for all samples
            avg_train_loss = total_train_loss / len(train_dataloader)
            print(f'[{epoch+1:03d}] train_loss: {avg_train_loss:.6f} (all samples)') # prints the average train loss of all samples
            
            #################### Validation ####################
            model.eval()
            
            total_val_loss = 0. # Accumulates the average loss of each val batch
            total_val_giou = 0.
            for val_batch_idx, val_batch in enumerate(val_dataloader):
                with torch.no_grad():
                    # Forward pass
                    truncated_slices, body_bb_gt = val_batch["dfov_cropped_truncated_slice"].to(device), val_batch["body_bb"].to(device)
                    truncated_slices = truncated_slices.to(torch.float32)
                    body_bb_pred = model(truncated_slices)

                    # Calculate loss
                    val_mse_loss = mse_loss_fn(body_bb_gt, body_bb_pred).mean(dim=1)
                    val_giou = giou(body_bb_gt, body_bb_pred, device=device)
                    val_loss = val_mse_loss + config["lambda"] * val_giou
                    val_giou = torch.mean(val_giou)
                    val_loss = torch.mean(val_loss)
                    total_val_giou += val_giou
                    total_val_loss += val_loss

            # Calculate the average val loss for all samples
            avg_val_giou = total_val_giou / len(val_dataloader) # The average GIoU loss of all samples in the val set
            avg_val_loss = total_val_loss / len(val_dataloader) # The average loss of all samples in the val set
            
            # Logging (GIoU, val_loss)
            mlflow.log_metric(key="val_giou", value=avg_val_giou, step=iteration)
            mlflow.log_metric(key="val_loss", value=avg_val_loss, step=iteration)

            # Save the best model
            if avg_val_loss <= best_avg_val_loss:
                delete_model_ckpt_in_current_run()
                save_model_ckpt(model=model, epoch=epoch)
                best_avg_val_loss = avg_val_loss
                num_epochs_no_improve = 0
            elif avg_val_loss > best_avg_val_loss and config["early_stopping_enabled"]:
                num_epochs_no_improve += 1
                print(f'Not improving after training for {num_epochs_no_improve} epochs')
                if epoch >= (config['enable_early_stop_after_num_epochs'] - 1) and num_epochs_no_improve >= config['max_epochs_no_improve']:
                    early_stopping = True
                    print('Early stopping!')
                    break
            
            print(f'[{epoch+1:03d}] val_loss: {avg_val_loss:.6f} | best_val_loss: {best_avg_val_loss:.6f} (all samples)')

            if early_stopping:
                break

            # Set model back to training mode
            model.train()
        
        #################### Inference on val set with the best model ####################
        print("Evaluation...")
        inference_body_bb_model(val_dataloader, data_split="val", config=config)
     
        return get_current_experiment_id(), get_current_run_id()
    
def inference_body_bb_model(dataloader, data_split, config, experiment_id=None, run_id=None, save_preds=True):
    mse_loss_fn = nn.MSELoss(reduction="none")

    active_run_exists = False
    if mlflow.active_run() is not None:
        active_run_exists = True

    if active_run_exists:
        experiment_id = get_current_experiment_id()
        run_id = get_current_run_id()
    else:
        experiment_id = experiment_id
        run_id = run_id

    best_model = BodyBBModel()
    best_model = best_model.to(best_model.device)
    best_model.eval()
    checkpoint_path = os.path.join(MLFLOW_DIR, experiment_id, run_id, "checkpoints")
    model_weights_path = os.path.join(checkpoint_path, data_utils.list_files(checkpoint_path)[0])
    load_model_ckpt(best_model, model_weights_path)

    total_mse_loss_dict = {"trace": 0., "mild": 0., "moderate": 0., "severe": 0.}
    total_giou_dict = {"trace": 0., "mild": 0., "moderate": 0., "severe": 0.}
    total_loss_dict = {"trace": 0., "mild": 0., "moderate": 0., "severe": 0.}
    trunc_sev_freqs = {"trace": 0, "mild": 0, "moderate": 0, "severe": 0}
    total_mse_loss = 0.
    total_giou = 0.
    total_loss = 0.
    rand_batch_idx = random.sample([i for i in range(len(dataloader))], 1)[0]

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloader)):
            truncated_slices, body_bb_gts = batch["dfov_cropped_truncated_slice"].to(best_model.device), batch["body_bb"].to(best_model.device)
            sev_levels = batch["sev_level"]
            truncated_slices = truncated_slices.to(torch.float32) # (batch_size, 3, 256, 256)
            body_bb_preds = best_model(truncated_slices) # (batch_size, 4)
            batch_mse_loss = mse_loss_fn(body_bb_gts, body_bb_preds).mean(dim=1)
            batch_giou = giou(body_bb_gts, body_bb_preds, device=best_model.device)
            batch_loss = batch_mse_loss + config["lambda"] * batch_giou

            for sev_level, mse_loss in zip(sev_levels, batch_mse_loss):
                total_mse_loss_dict[sev_level] += (mse_loss.item())
                trunc_sev_freqs[sev_level] += 1
            for sev_level, giou_val in zip(sev_levels, batch_giou):
                total_giou_dict[sev_level] += (giou_val.item())
            for sev_level, loss_val in zip(sev_levels, batch_loss):
                total_loss_dict[sev_level] += (loss_val.item())

            batch_mse_loss, batch_giou, batch_loss = torch.mean(batch_mse_loss), torch.mean(batch_giou), torch.mean(batch_loss)
            total_mse_loss += batch_mse_loss
            total_giou += batch_giou
            total_loss += batch_loss

            # Save predictions for 1 random batch
            if batch_idx == rand_batch_idx and save_preds:
                save_predictions_path = os.path.join(MLFLOW_DIR, experiment_id, run_id, "predictions")
                data_utils.mkdir_p(save_predictions_path)
                slice_filenames = batch["filename"]
                sev_levels = batch["sev_level"]
                for i in range(truncated_slices.shape[0]):
                    truncated_slice, body_bb_gt, body_bb_pred = truncated_slices[i].unsqueeze(0), body_bb_gts[i].unsqueeze(0), body_bb_preds[i].unsqueeze(0)
                    slice_filename = slice_filenames[i]
                    sev_level = sev_levels[i]
                    plot_compare_body_bb_gt_and_pred(truncated_slice, body_bb_gt, body_bb_pred, additional_text=slice_filename, save=True, save_img_path=os.path.join(save_predictions_path, f"{sev_level}_{data_split}_{i}.png"))
            
    # Logging metrics
    avg_mse_loss = total_mse_loss / len(dataloader)
    avg_giou = total_giou / len(dataloader)
    avg_loss = total_loss / len(dataloader)
    
    if active_run_exists:
        mlflow.log_metric(key=f"{data_split}_mse_loss_final", value=avg_mse_loss)
        mlflow.log_metric(key=f"{data_split}_giou_final", value=avg_giou)
    else:
        print(f"{data_split}_mse_loss_final: {avg_mse_loss}")
        print(f"{data_split}_giou_final: {avg_giou}")
        print(f"{data_split}_loss_final: {avg_loss}")

    for sev_level, mse_loss in total_mse_loss_dict.items():
        avg_mse_loss_sev_level = mse_loss / trunc_sev_freqs[sev_level]
        if active_run_exists:
            mlflow.log_metric(key=f"{data_split}_{sev_level}_mse_loss_final", value=avg_mse_loss_sev_level)
        else:
            print(f"{data_split}_{sev_level}_mse_loss_final: {avg_mse_loss_sev_level}")
    for sev_level, giou_val in total_giou_dict.items():
        avg_giou_val_sev_level = giou_val / trunc_sev_freqs[sev_level]
        if active_run_exists:
            mlflow.log_metric(key=f"{data_split}_{sev_level}_giou_final", value=avg_giou_val_sev_level)
        else:
            print(f"{data_split}_{sev_level}_giou_final: {avg_giou_val_sev_level}")
    for sev_level, loss_val in total_loss_dict.items():
        avg_loss_sev_level = loss_val / trunc_sev_freqs[sev_level]
        if active_run_exists:
            pass
        else:
            print(f"{data_split}_{sev_level}_loss_final: {avg_loss_sev_level}")

def plot_compare_body_bb_gt_and_pred(ct_slice, body_bb_gt, body_bb_pred, additional_text, save=True, save_img_path=None):
    """ Plots the CT slice with ground-truth and predicted body bounding boxes overlayed on top of it. """
    fig, ax = plt.subplots()

    ct_slice = ct_slice.squeeze().cpu().numpy()[0] # (1, 3, 256, 256) -> (1, 256, 256) (take the first channel, it's the same as the other channels anyway)
    ax.imshow(ct_slice, cmap='gray')
    
    mse_loss_fn = nn.MSELoss()
    
    giou_val = giou(body_bb_gt, body_bb_pred, body_bb_gt.device).item()
    mse_loss = mse_loss_fn(body_bb_gt, body_bb_pred).item()

    body_bb_gt = body_bb_gt.squeeze().detach().cpu().numpy()
    body_bb_pred = body_bb_pred.squeeze().detach().cpu().numpy()

    rect_gt = patches.Rectangle((body_bb_gt[0], body_bb_gt[1]), body_bb_gt[2] - body_bb_gt[0], body_bb_gt[3] - body_bb_gt[1],
                            linewidth=1, edgecolor='red', facecolor='none')
    ax.add_patch(rect_gt)
    rect_gt.set_clip_on(False)

    rect_pred = patches.Rectangle((body_bb_pred[0], body_bb_pred[1]), body_bb_pred[2] - body_bb_pred[0], body_bb_pred[3] - body_bb_pred[1],
                            linewidth=1, edgecolor='blue', facecolor='none')
    ax.add_patch(rect_pred)
    rect_pred.set_clip_on(False)
    
    plt.text(128, -10, additional_text, fontsize=8, ha='center')
    plt.text(128, 280, f"GIoU: {giou_val:.3f}", fontsize=10, ha='center')
    plt.text(128, 290, f"MSE loss: {mse_loss:.3f}", fontsize=10, ha='center')
    
    if save:
        plt.savefig(save_img_path)
    else:
        plt.show()
    
    plt.close()

############################## RFR-Net ##############################

class VGG16FeatureExtractor(nn.Module):
    def __init__(self, precision=32):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False
            
        self.precision = precision

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        if self.precision == 16:
            results = [res.to(torch.float16) for res in results]
        return results[1:]

def plot_compare_gt_and_pred_untrunc_images(gt_untruncated_slice, truncated_slice, pred_untruncated_slice, fake_untruncated_slice,
                                            target_slice_channel, ssim, ssim_comp, additional_text,
                                            save=True, save_img_path=None):
    """ Plots the ground-truth untruncated, truncated, final, and predicted image side-by-side. """
    fig, ax = plt.subplots(1, 4)

    gt_untruncated_slice = gt_untruncated_slice.squeeze().cpu().numpy()[target_slice_channel] # (1, 3, 256, 256) -> (1, 256, 256) (take the first channel, it's the same as the other channels anyway)
    truncated_slice = truncated_slice.squeeze().cpu().numpy()[target_slice_channel]
    pred_untruncated_slice = pred_untruncated_slice.squeeze().cpu().numpy()[target_slice_channel]
    fake_untruncated_slice = fake_untruncated_slice.squeeze().cpu().numpy()[target_slice_channel]

    ax[0].imshow(gt_untruncated_slice, cmap='gray')
    ax[0].set_title("Groundtruth image", fontsize=8)
    ax[1].imshow(truncated_slice, cmap='gray')
    ax[1].set_title("Truncated image", fontsize=8)
    ax[2].imshow(pred_untruncated_slice, cmap='gray')
    ax[2].set_title(f"Final image\nSSIM: {ssim_comp:.3f}", fontsize=8)
    ax[3].imshow(fake_untruncated_slice, cmap='gray')
    ax[3].set_title(f"Predicted image\nSSIM: {ssim:.3f}", fontsize=8)

    fig.text(0.5, 0.7, additional_text, fontsize=8, ha='center')

    if save:
        plt.savefig(save_img_path)
    else:
        plt.show()
    
    plt.close()

