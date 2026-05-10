import os
import gc
import argparse
import MinkowskiEngine as ME
import pytorch_lightning as pl
from downstream.evaluate import evaluate_points, evaluate_images, evaluate_visual_points
from utils.read_config import generate_config
from downstream.model_builder import make_model
from pytorch_lightning.plugins import DDPPlugin
from downstream.lightning_trainer import LightningDownstream
from downstream.lightning_datamodule import DownstreamDataModule
from downstream.dataloader_kitti import make_data_loader as make_data_loader_kitti
from downstream.dataloader_nuscenes import make_data_loader as make_data_loader_nuscenes
from downstream.dataloader_scannet import make_data_loader as make_data_loader_scannet
import torch.nn as nn

def main():
    """
    Code for launching the downstream training
    """
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file", type=str, default="config/semseg_nuscenes.yaml", help="specify the config for training"
    )
    parser.add_argument(
        "--resume_path", type=str, default=None, help="provide a path to resume an incomplete training"
    )
    parser.add_argument(
        "--pretraining_path", type=str, default=None, help="provide a path to pre-trained weights"
    )
    #/volumes/jointmodel/wyr/code/sam_clip3d/output/foundationTrans/nuscenes/240524-0613/model.pt
    #"/volumes/jointmodel/wyr/code/sam_clip3d/output/foundationTrans/nuscenes/220524-0258/model.pt"
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
    dm = DownstreamDataModule(config)
    # model, model_images, SAM = make_model(config, config["pretraining_path"])
    model_points, model_images, model_clip, SAM = make_model(config, config["pretraining_path"])
    if config["num_gpus"] > 1:
        model_points = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model_points)
        # model_images = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model_images)
        model_images = nn.SyncBatchNorm.convert_sync_batchnorm(model_images)
    module = LightningDownstream(model_points, model_clip, config)
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
        check_val_every_n_epoch=10,
    )
    print("Starting the training")
    # trainer.fit(module, dm)

    print("Training finished, now evaluating the results")
    del trainer
    del dm
    del module
    gc.collect()
    if config["dataset"].lower() == "nuscenes":
        phase = "verifying" if config['training'] in ("parametrize", "parametrizing") else "val"
        val_dataloader = make_data_loader_nuscenes(
            config, phase, num_threads=config["num_threads"]
        )
    elif config["dataset"].lower() == "kitti":
        val_dataloader = make_data_loader_kitti(
            config, "val", num_threads=config["num_threads"]
        )
    elif config["dataset"].lower() == "scannet":
        val_dataloader = make_data_loader_scannet(
            config, "val", num_threads=config["num_threads"]
        )
    # evaluate_visual_points(model_points.to(0), val_dataloader, config)
    evaluate_images(model_images.to(0), model_clip.to(0), SAM, val_dataloader, config)

if __name__ == "__main__":
    main()
