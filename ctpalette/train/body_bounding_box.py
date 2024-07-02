""" This script is used to train a body bounding box detector.
Adjust the train configs in config/body_bb.yaml. """

from torch.utils.data import DataLoader
import yaml

from ctpalette.paths import *
from ctpalette.data.dataset import *
from ctpalette.models.model import BodyBBModel
import ctpalette.train.train_utils as train_utils

if __name__ == "__main__":
    with open(body_bb_train_config_path, "r") as f:
        train_config = yaml.safe_load(f)

    train_set = BodyBBDataset_Train(aug=train_config["aug"])
    val_set = BodyBBDataset_Val_Test(data_split="val")

    print(f"Train set: {len(train_set)} samples")
    print(f"Val set: {len(val_set)} samples")

    train_dataloader = DataLoader(train_set, batch_size=train_config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=train_config["batch_size"], shuffle=False)

    print("Training...")
    bb_model = BodyBBModel()
    experiment_id, run_id = train_utils.train_body_bb_model(bb_model, train_dataloader, val_dataloader, train_config)