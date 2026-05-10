import os
import torch
import requests
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import model_urls
from model.modules.resnet_encoder import resnet_encoders
import model.modules.dino.vision_transformer as dino_vit
class feature_mappingNet(nn.Module):
    """
    Dilated ResNet Feature Extractor
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.text_embeddings_path = self.config['text_embeddings_path']
        text_categories = self.config['text_categories']
        if self.text_embeddings_path is None:
            self.text_embeddings = nn.Parameter(torch.zeros(text_categories, 512))
            nn.init.normal_(self.text_embeddings, mean=0.0, std=0.01)
        else:
            self.register_buffer('text_embeddings', torch.randn(text_categories, 512))
            loaded = torch.load(self.text_embeddings_path, map_location='cuda')
            self.text_embeddings[:, :] = loaded[:, :]

        self.deeplab_to_clip = nn.Conv2d(256, 64, 1, 1)
        self.deeplab_to_sam = nn.Conv2d(256, 64, 1, 1)
        self.point_to_clip = nn.Linear(96, 64)
        self.point_to_sam = nn.Linear(96, 64)
        self.clip_mapping = nn.Conv2d(768, 64, 1, 1)
        self.sam_mapping = nn.Conv2d(256, 64, 1, 1)
        self.image_size = (224, 416)


    def forward(self, image_feats_clip, image_feats_deeplab, point_feats, embeddings_sam, pairing_points, m):
        # torch.Size([10, 768, 15, 20]), torch.Size([8985, 96]), torch.Size([10, 2048, 15, 20]), torch.Size([10, 256, 64, 64])
        # print(image_feats_clip.shape, point_feats.shape, image_feats_deeplab.shape, embeddings_sam.shape)

        # point mapping to clip and sam
        point_feats_to_clip = self.point_to_clip(point_feats)[pairing_points]
        point_feats_to_sam = self.point_to_sam(point_feats)[pairing_points]

        # deeplab image mapping to clip and sam
        image_feats_deeplab_to_clip = self.deeplab_to_clip(image_feats_deeplab)
        image_feats_deeplab_to_sam = self.deeplab_to_sam(image_feats_deeplab)
        image_feats_deeplab_to_sam = F.interpolate(image_feats_deeplab_to_sam, size=(48, 64), mode='bilinear',
                                                   align_corners=False)

        # clip image mapping to deeplab image
        image_feats_clip = self.clip_mapping(image_feats_clip)
        image_feats_clip_to_deeplab = image_feats_clip
        image_feats_clip_to_deeplab = F.normalize(image_feats_clip_to_deeplab, p=2, dim=1)

        # clip image mapping to point
        image_feats_clip_to_point = F.interpolate(image_feats_clip, size=self.image_size, mode='bilinear',
                                                   align_corners=False)
        image_feats_clip_to_point = image_feats_clip_to_point.permute(0, 2, 3, 1).contiguous()[m]
        image_feats_clip_to_point = F.normalize(image_feats_clip_to_point, p=2, dim=1)


        # sam image mapping to deeplab image
        image_feats_sam = self.sam_mapping(embeddings_sam)
        image_feats_sam = image_feats_sam[:, :, :48, :]
        image_feats_sam_to_deeplab = F.normalize(image_feats_sam, p=2, dim=1)

        # sam image mapping to point
        image_feats_sam_to_point = F.interpolate(image_feats_sam, size=self.image_size, mode='bilinear',
                                                   align_corners=False)
        image_feats_sam_to_point = image_feats_sam_to_point.permute(0, 2, 3, 1).contiguous()[m]
        image_feats_sam_to_point = F.normalize(image_feats_sam_to_point, p=2, dim=1)

        return {"point_to_clip": point_feats_to_clip, "point_to_sam": point_feats_to_sam,
                "deeplab_to_clip": image_feats_deeplab_to_clip, "deeplab_to_sam": image_feats_deeplab_to_sam,
                "clip_to_deeplab": image_feats_clip_to_deeplab, "clip_to_point": image_feats_clip_to_point,
                "sam_to_deeplab": image_feats_sam_to_deeplab, "sam_to_point": image_feats_sam_to_point}