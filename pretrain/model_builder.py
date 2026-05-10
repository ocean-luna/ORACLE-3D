from model import (
    SPVCNN,
    MinkUNet,
    VoxelNet,
    DilationFeatureExtractor,
    PPKTFeatureExtractor,
    Preprocessing,
    DinoVitFeatureExtractor,
    fusionNet,
    maskClipFeatureExtractor,
    feature_mappingNet,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.engine import (
    DefaultTrainer)
# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from model.deeplabv3 import deeplabv3_resnet50
import torch
from detectron2.config import get_cfg
from model.maskdino import (
    # COCOInstanceNewBaselineDatasetMapper,
    # COCOPanopticNewBaselineDatasetMapper,
    # InstanceSegEvaluator,
    # MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskdino_config,
    add_eva_config,
    # DetrDatasetMapper,
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


def forgiving_state_restore(net, loaded_dict):
    """
    Handle partial loading when some tensors don't match up in size.
    Because we want to use models that were trained off a different
    number of classes.
    """
    loaded_dict = {
        k.replace("module.", ""): v for k, v in loaded_dict.items()
    }
    net_state_dict = net.state_dict()
    new_loaded_dict = {}
    for k in net_state_dict:
        new_k = k
        if (
            new_k in loaded_dict and net_state_dict[k].size() == loaded_dict[new_k].size()
        ):
            new_loaded_dict[k] = loaded_dict[new_k]
        else:
            print("Skipped loading parameter {}".format(k))
    net_state_dict.update(new_loaded_dict)
    net.load_state_dict(net_state_dict)
    return net

def make_model(config):
    """
    Build points and image models according to what is in the config
    """

    #
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"

    # model_SAM = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    # sam.to(device=device)
    # model_SAM = SamAutomaticMaskGenerator(model_SAM.to(device=device))
    model_SAM = 0

    if config["model_image"]=="deeplab":
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


    feature_mapping = feature_mappingNet(config=config)
    # model_fusion = fusionNet(config)
    if config['dataset'] == "nuscenes":
        in_feature_dim = 1
    elif config['dataset'] == "scannet":
        in_feature_dim = 3
    if config["model_points"] == "voxelnet":
        model_points = VoxelNet(4, config["model_n_out"], config)
    else:
        # model_points = SPVCNN(1, config["model_n_out"], config)
        model_points = MinkUNet(in_feature_dim, config["model_n_out"], config)
    if config["images_encoder"].find("vit_") != -1:
        model_clip = DinoVitFeatureExtractor(config, preprocessing=Preprocessing())
    elif config["images_encoder"] == "maskclip":
        # model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
        model_clip = maskClipFeatureExtractor(config, preprocessing=Preprocessing())

    elif config["decoder"] == "dilation":
        model_clip = DilationFeatureExtractor(config, preprocessing=Preprocessing())
    elif config["decoder"] == "ppkt":
        model_clip = PPKTFeatureExtractor(config, preprocessing=Preprocessing())
    else:
        # model with a decoder
        raise Exception(f"Model not found: {config['decoder']}")

    print(config["pretraining_path"])
    print("=========================================================")

    if config["pretraining_path"]:
        print("Training with pretrained model")
        checkpoint = torch.load(config["pretraining_path"], map_location="cpu")
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


    return model_points, model_clip, model_images, feature_mapping, model_SAM
