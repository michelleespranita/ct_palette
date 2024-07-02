#################################################################
# extended and adapted from:
# https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models
#################################################################

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import transforms as T
import torch.utils.data
from tensorflow.keras.models import load_model

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy

import lpips

from ctpalette.paths import *
import ctpalette.test.eval_utils as eval_utils

loss_fn_alex = lpips.LPIPS(net='alex')
loss_fn_alex.cuda()

upsample_fn = T.Resize(size=512)
seg_model = load_model(seg_model_path, compile=False)

def mae(input, target): # input/target shape (B,C,H,W)
    with torch.no_grad():
        loss = nn.L1Loss()
        output = loss(input, target)
    return output

def lpips_fn(input, target): # input/target shape (B,C,H,W)
    with torch.no_grad():
        return torch.mean(loss_fn_alex(input, target).squeeze()).cpu()
    
def rmse_muscle_sat_area_diff_untrunc_pred(untrunc_slices, pred_slices):
    '''
    untrunc_slices, pred_slices -> (B, C, H, W): (B, 1, 256, 256)
    '''
    # Segment untrunc slice
    untrunc_slices = untrunc_slices.cpu()
    zero_twofivefive_untrunc_slices = 127.5 * (untrunc_slices + 1) # change the intensity back to [0, 255]
    zero_twofivefive_untrunc_slices = upsample_fn(zero_twofivefive_untrunc_slices)
    untrunc_seg_masks = eval_utils.segmentation_batch(zero_twofivefive_untrunc_slices.numpy(), seg_model)

    # Segment pred slice
    pred_slices = pred_slices.cpu()
    pred_slices = torch.clamp(pred_slices, -1, 1)
    zero_twofivefive_pred_slices = 127.5 * (pred_slices + 1) # change the intensity back to [0, 255]
    zero_twofivefive_pred_slices = upsample_fn(zero_twofivefive_pred_slices)
    pred_seg_masks = eval_utils.segmentation_batch(zero_twofivefive_pred_slices.numpy(), seg_model)

    # Calculate num pixels associated with seg mask
    untrunc_muscle_num_pixels = (untrunc_seg_masks == 1).sum()
    untrunc_sat_num_pixels = (untrunc_seg_masks == 2).sum()
    pred_muscle_num_pixels = (pred_seg_masks == 1).sum()
    pred_sat_num_pixels = (pred_seg_masks == 2).sum()

    return np.sqrt(((untrunc_muscle_num_pixels - pred_muscle_num_pixels)**2 + (untrunc_sat_num_pixels - pred_sat_num_pixels)**2)/2)

def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)