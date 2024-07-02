""" This script is used for inference of CT-Palette. """

from argparse import ArgumentParser
from pydicom import dcmread
from tqdm import tqdm
from PIL import Image

from ctpalette.paths import *
import ctpalette.test.eval_utils as eval_utils
import ctpalette.data.data_utils as data_utils

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--path_to_dcm", type=str, help="Path to the .dcm file that you want to outpaint")
    argparser.add_argument("--path_to_result_dir", type=str, help="Path to directory where you want to save the resulting images")
    argparser.add_argument("--multiple_inf", action="store_true", help="If True, use multiple inference. Else, single inference.")
    argparser.add_argument("--mode", type=str, default="median", choices=["median", "mean"], help="L1/median or L2/mean. Only used in multiple inference.")

    args = argparser.parse_args()
    path_to_dcm = args.path_to_dcm
    path_to_result_dir = args.path_to_result_dir
    multiple_inf = args.multiple_inf
    mode = args.mode

    if multiple_inf and seg_model_path == "":
        raise ValueError("Please provide the path to the segmentation model!")

    data_utils.mkdir_p(path_to_result_dir)

    # Load body bounding box detector
    body_bb_model = eval_utils.load_body_bb_model(body_bb_model_path)

    # Load image outpainting model
    outpainting_model = eval_utils.load_outpainting_model(image_outpainting_config_path, image_outpainting_model_path)

    # Load segmentation model
    if multiple_inf:
        seg_model = eval_utils.load_segmentation_model(seg_model_path)

    # Load dcm
    dcm = dcmread(path_to_dcm)
    img_name = os.path.basename(path_to_dcm).replace('.dcm', '')

    # CT-Palette
    if multiple_inf:
        rgb_outpainted_slice, new_pixel_spacing = eval_utils.run_CT_Palette(dcm=dcm,
                                                                            body_bb_model=body_bb_model,
                                                                            outpainting_model=outpainting_model,
                                                                            center=True,
                                                                            multiple_inference=True,
                                                                            seg_model=seg_model,
                                                                            num_generated_samples=5,
                                                                            mode=mode,
                                                                            outlier_detection=True)
    else:
        rgb_outpainted_slice, new_pixel_spacing = eval_utils.run_CT_Palette(dcm=dcm,
                                                                            body_bb_model=body_bb_model,
                                                                            outpainting_model=outpainting_model,
                                                                            center=True,
                                                                            multiple_inference=False)

    # Save outpainted image
    eval_utils.save_img_pillow(img=rgb_outpainted_slice.numpy(), img_path=os.path.join(path_to_result_dir, f"{img_name}_outpainted.png"))
