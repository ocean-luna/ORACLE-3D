import os
import argparse
import torch.nn as nn
import MinkowskiEngine as ME
import pytorch_lightning as pl
from utils.read_config import generate_config
from pretrain.model_builder import make_model
from pytorch_lightning.plugins import DDPPlugin
from pretrain.lightning_trainer import LightningPretrain
from pretrain.lightning_datamodule import PretrainDataModule
from pretrain.lightning_trainer_spconv import LightningPretrainSpconv
import warnings
import sys
from model.glee.backbone import *
sys.path.append("model/maskdino")


def main():
    """
    Code for launching the pretraining
    """
    # 忽略所有“is_namedtuple is deprecated”的警告
    warnings.filterwarnings("ignore", category=UserWarning, message=".*is_namedtuple is deprecated.*")
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file", type=str, default="config/slidr_minkunet.yaml", help="specify the config for training"
    )
    parser.add_argument(
        "--resume_path", type=str, default=None, help="provide a path to resume an incomplete training"
    )
    parser.add_argument(
        "--pretraining_path", type=str, default=None, help="provide a path to pre-trained weights"
    )
    args = parser.parse_args()
    config = generate_config(args.cfg_file)
    if args.resume_path:
        config['resume_path'] = args.resume_path
    if args.pretraining_path:
        config['pretraining_path'] = args.pretraining_path

    if os.environ.get("LOCAL_RANK", 0) == 0:
        print(
            "\n" + "\n".join(list(map(lambda x: f"{x[0]:20}: {x[1]}", config.items())))
        )

    dm = PretrainDataModule(config)
    model_points, model_images, model_fusion, feature_mapping, model_SAM = make_model(config)

    if config["num_gpus"] > 1:
        if config["model_points"] == "minkunet":
            model_points = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model_points)
        model_images = nn.SyncBatchNorm.convert_sync_batchnorm(model_images)
        # model_points = model_points #nn.SyncBatchNorm.convert_sync_batchnorm(model_points)
        # model_points = nn.SyncBatchNorm.convert_sync_batchnorm(model_points)
        model_fusion = nn.SyncBatchNorm.convert_sync_batchnorm(model_fusion)
        feature_mapping = nn.SyncBatchNorm.convert_sync_batchnorm(feature_mapping)
    if config["model_points"] == "minkunet":
        module = LightningPretrain(model_points, model_images, model_fusion, feature_mapping, model_SAM, config)
    elif config["model_points"] == "voxelnet":
        module = LightningPretrainSpconv(model_points, model_images, config)
    path = os.path.join(config["working_dir"], config["datetime"])
    trainer = pl.Trainer(
        gpus=config["num_gpus"],
        accelerator="ddp",
        default_root_dir=path,
        checkpoint_callback=True,
        max_epochs=config["num_epochs"],
        plugins=DDPPlugin(find_unused_parameters=True),
        num_sanity_val_steps=0,
        resume_from_checkpoint=config["resume_path"],
        check_val_every_n_epoch=4,
    )
    print("Starting the training")
    trainer.fit(module, dm)


if __name__ == "__main__":
    main()



