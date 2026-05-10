import torch
from model import MinkUNet, SPVCNN, maskClipFeatureExtractor, Preprocessing
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from model.deeplabv3 import deeplabv3_resnet50
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.engine import (
    DefaultTrainer)
# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from model.deeplabv3 import deeplabv3_resnet50
import torch
from detectron2.config import get_cfg
from model.maskdino import (
#     COCOInstanceNewBaselineDatasetMapper,
#     COCOPanopticNewBaselineDatasetMapper,
#     InstanceSegEvaluator,
#     MaskFormerSemanticDatasetMapper,
#     SemanticSegmentorWithTTA,
    add_maskdino_config,
    add_eva_config,
#     DetrDatasetMapper,
)
from detectron2.checkpoint import DetectionCheckpointer
def load_state_with_same_shape(model, weights):
    """
    Load common weights in two similar models
    (for instance between a pretraining and a downstream training)
    """

    model_state = model.state_dict()
    if list(weights.keys())[0].startswith("model."):
        weights = {k.partition("model.")[2]: weights[k] for k in weights.keys()}

    if list(weights.keys())[0].startswith("model_points."):
        weights = {k.partition("model_points.")[2]: weights[k] for k in weights.keys()}

    if list(weights.keys())[0].startswith("module."):
        print("Loading multigpu weights with module. prefix...")
        weights = {k.partition("module.")[2]: weights[k] for k in weights.keys()}

    if list(weights.keys())[0].startswith("encoder."):
        print("Loading multigpu weights with encoder. prefix...")
        weights = {k.partition("encoder.")[2]: weights[k] for k in weights.keys()}

    filtered_weights = {
        k: v
        for k, v in weights.items()
        if (k in model_state and v.size() == model_state[k].size())
    }
    removed_weights = {
        k: v
        for k, v in weights.items()
        if not (k in model_state and v.size() == model_state[k].size())
    }
    print("Loading weights:" + ", ".join(filtered_weights.keys()))
    print("")
    print("Not loading weights:" + ", ".join(removed_weights.keys()))
    return filtered_weights


def make_model(config, load_path=None):
    """
    Build the points model according to what is in the config
    """

    assert not config[
        "normalize_features"
    ], "You shouldn't normalize features for the downstream task"
    # model = MinkUNet(1, config["model_n_out"], config)
    # model = SPVCNN(1, config["model_n_out"], config)


    model_points = MinkUNet(1, config["model_n_out"], config)
    model_clip = maskClipFeatureExtractor(config, preprocessing=Preprocessing())

    if config["model_image"] == "deeplab":
        model_images = deeplabv3_resnet50(config=config)
    else:

        config_file = "config/glee/eva02.yaml"
        cfg = get_cfg()
        cfg.set_new_allowed(True)
        add_deeplab_config(cfg)
        add_maskdino_config(cfg)
        cfg.merge_from_file(config_file)
        add_eva_config(cfg)
        model_images = DefaultTrainer.build_model(cfg)

        DetectionCheckpointer(model_images, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS
        )

    sam_checkpoint = "./model_path/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    #
    # sam_checkpoint = "sam_vit_h_4b8939.pth"
    # model_type = "vit_h"
    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    # sam.to(device=device)
    # mask_generator = SamAutomaticMaskGenerator(sam.to(device=device))
    mask_generator = SamPredictor(sam.to(device=device))
    # mask_generator = SamAutomaticMaskGenerator(
    #     model=sam.to(device=device),
    #     points_per_side=32,
    #     pred_iou_thresh=0.86,
    #     stability_score_thresh=0.92,
    #     crop_n_layers=1,
    #     crop_n_points_downscale_factor=2,
    #     min_mask_region_area=100,  # Requires open-cv to run post-processing
    # )

    if load_path:
        print("Training with pretrained model")
        checkpoint = torch.load(load_path, map_location="cpu")
        if "config" in checkpoint:
            for cfg in ("voxel_size", "cylindrical_coordinates"):
                assert checkpoint["config"][cfg] == config[cfg], (
                    f"{cfg} is not consistant. "
                    f"Checkpoint: {checkpoint['config'][cfg]}, "
                    f"Config: {config[cfg]}."
                )
        if set(checkpoint.keys()) == set(["epoch", "model", "optimizer", "train_criterion"]):
            print("Pre-trained weights are coming from DepthContrast.")
            pretraining_epochs = checkpoint["epoch"]
            print(f"==> Number of pre-training epochs {pretraining_epochs}")
            checkpoint = checkpoint["model"]
            if list(checkpoint.keys())[0].startswith("module."):
                print("Loading multigpu weights with module. prefix...")
                checkpoint = {k.partition("module.")[2]: checkpoint[k] for k in checkpoint.keys()}
            voxel_net_suffix = "trunk.2."
            checkpoint = {
                key.partition(voxel_net_suffix)[2]: checkpoint[key]
                for key in checkpoint.keys() if key.startswith(voxel_net_suffix)
            }
            print(f"==> Number of loaded weight blobs {len(checkpoint)}")
            checkpoint = {"model_points": checkpoint}
        key = "model_points" if "model_points" in checkpoint else "state_dict"
        filtered_weights = load_state_with_same_shape(model_points, checkpoint[key])
        model_dict = model_points.state_dict()
        model_dict.update(filtered_weights)
        model_points.load_state_dict(model_dict)

        if set(checkpoint.keys()) == set(["epoch", "model", "optimizer", "train_criterion"]):
            print("Pre-trained weights are coming from DepthContrast.")
            pretraining_epochs = checkpoint["epoch"]
            print(f"==> Number of pre-training epochs {pretraining_epochs}")
            checkpoint = checkpoint["model"]
            if list(checkpoint.keys())[0].startswith("module."):
                print("Loading multigpu weights with module. prefix...")
                checkpoint = {k.partition("module.")[2]: checkpoint[k] for k in checkpoint.keys()}
            voxel_net_suffix = "trunk.2."
            checkpoint = {
                key.partition(voxel_net_suffix)[2]: checkpoint[key]
                for key in checkpoint.keys() if key.startswith(voxel_net_suffix)
            }
            print(f"==> Number of loaded weight blobs {len(checkpoint)}")
            checkpoint = {"model_images": checkpoint}
        key = "model_images" if "model_images" in checkpoint else "state_dict"
        filtered_weights = load_state_with_same_shape(model_images, checkpoint[key])
        model_dict = model_images.state_dict()
        model_dict.update(filtered_weights)
        model_images.load_state_dict(model_dict)



    if config["freeze_layers"]:
        for param in list(model_points.parameters())[:-2]:
            param.requires_grad = False
    return model_points, model_images, model_clip, mask_generator
