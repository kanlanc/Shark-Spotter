import argparse
import logging
import pathlib
from typing import Any, Dict
from pathlib import Path

import torch
from torch import nn
import determined as det
from determined import pytorch
from determined.pytorch import DataLoader

from models.yolo import Model
from utils.datasets import create_dataloader
from utils.loss import ComputeLoss
from utils.general import check_dataset, labels_to_class_weights


class SharkspotterTrial(pytorch.PyTorchTrial):
    def __init__(self, context: pytorch.PyTorchTrialContext, hparams: Dict) -> None:
        self.context = context
        self.hparams = hparams

        # Set up data paths using the new structure
        self.data_dir = Path(hparams.get("data_dir", "/data/sharkspotter"))
        self.train_dir = Path(hparams.get("train_dir", "/data/sharkspotter/train"))
        self.valid_dir = Path(hparams.get("valid_dir", "/data/sharkspotter/valid"))
        self.test_dir = Path(hparams.get("test_dir", "/data/sharkspotter/test"))
        
        # Get batch size taking into account distributed training
        self.batch_size = hparams.get("batch_size", 16)
        self.per_slot_batch_size = self.batch_size // self.context.distributed.get_size()
        
        # Image size from hyperparameters
        self.img_size = hparams.get("img_size", 640)

        # Initialize YOLOv7 model
        self.model = self.context.wrap_model(
            Model(
                cfg='models/yolov7.yaml',
                ch=3,
                nc=hparams.get("nc", 1)
            )
        )

        # Initialize loss function
        self.compute_loss = ComputeLoss(self.model)

        # Configure optimizer
        self.optimizer = self.context.wrap_optimizer(
            torch.optim.Adam(
                self.model.parameters(),
                lr=hparams.get("learning_rate", 0.001),
                weight_decay=hparams.get("weight_decay", 0.0005)
            )
        )

    def build_training_data_loader(self) -> DataLoader:
        """Create training data loader with augmentations"""
        train_path = str(self.train_dir)  # Using the direct train directory
        dataloader, dataset = create_dataloader(
            train_path,
            imgsz=self.img_size,
            batch_size=self.per_slot_batch_size,
            stride=32,
            hyp=self.hparams,
            augment=True,
            cache=False,
            rect=False,
            rank=self.context.distributed.get_rank(),
            workers=8,
            image_weights=False,
            prefix="train: "
        )
        return DataLoader(dataset, batch_size=None)

    def build_validation_data_loader(self) -> DataLoader:
        """Create validation data loader"""
        val_path = str(self.valid_dir)  # Using the direct valid directory
        dataloader, dataset = create_dataloader(
            val_path,
            imgsz=self.img_size,
            batch_size=self.per_slot_batch_size,
            stride=32,
            hyp=self.hparams,
            augment=False,
            cache=False,
            rect=True,
            rank=self.context.distributed.get_rank(),
            workers=8,
            image_weights=False,
            prefix="val: "
        )
        return DataLoader(dataset, batch_size=None)

    def train_batch(
        self, batch: pytorch.TorchData, epoch_idx: int, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Training step"""
        imgs, targets, _, _ = batch
        imgs = imgs.to(self.context.device).float() / 255.0  # Scale to [0, 1]
        
        # Forward pass
        pred = self.model(imgs)
        
        # Calculate loss
        loss, loss_items = self.compute_loss(pred, targets.to(self.context.device))
        
        # Backward pass
        self.context.backward(loss)
        self.context.step_optimizer(self.optimizer)

        return {
            "loss": loss,
            "box_loss": loss_items[0],
            "obj_loss": loss_items[1],
            "cls_loss": loss_items[2]
        }

    def evaluate_batch(self, batch: pytorch.TorchData, batch_idx: int) -> Dict[str, Any]:
        """Validation step"""
        imgs, targets, _, _ = batch
        imgs = imgs.to(self.context.device).float() / 255.0
        
        # Forward pass with no gradient
        with torch.no_grad():
            pred = self.model(imgs)
            loss, loss_items = self.compute_loss(pred, targets.to(self.context.device))
        
        return {
            "validation_loss": loss.item(),
            "val_box_loss": loss_items[0].item(),
            "val_obj_loss": loss_items[1].item(),
            "val_cls_loss": loss_items[2].item(),
        }


def run(max_length, local: bool = False):
    """Initialize and run the training process"""
    info = det.get_cluster_info()

    if local:
        # Load hyperparameters from const.yaml for local training
        from ruamel import yaml
        yml = yaml.YAML(typ="safe", pure=True)
        conf = yml.load(pathlib.Path("./const.yaml").read_text())
        hparams = conf["hyperparameters"]
        latest_checkpoint = None
    else:
        # Get hyperparameters from Determined cluster
        hparams = info.trial.hparams
        latest_checkpoint = info.latest_checkpoint

    # Initialize training context and start training
    with pytorch.init() as train_context:
        trial = SharkspotterTrial(train_context, hparams=hparams)
        trainer = pytorch.Trainer(trial, train_context)
        
        trainer.fit(
            max_length=max_length,
            latest_checkpoint=latest_checkpoint,
            validation_period=pytorch.Batch(100),  # Validate every 100 batches
        )


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format=det.LOG_FORMAT)

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--batches", type=int)
    group.add_argument("--epochs", type=int)
    args = parser.parse_args()

    # Set training length
    if args.batches:
        max_length = pytorch.Batch(args.batches)
    elif args.epochs:
        max_length = pytorch.Epoch(args.epochs)
    else:
        max_length = pytorch.Epoch(100)  # Default to 100 epochs

    # Detect if running locally or on cluster
    local_training = det.get_cluster_info() is None
    run(max_length, local=local_training)