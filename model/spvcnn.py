import torchsparse
import torchsparse.nn as spnn
import torch
import torch.nn.functional as F
import numpy as np
import pickle
from torch import nn
from torchsparse import PointTensor
# from pcdet.models.segmentors.base_segmentors import BaseSegmentor
from torchsparse import SparseTensor
from torchsparse.nn.utils import fapply
# from ..utils import initial_voxelize, point_to_voxel, voxel_to_point
# from ......utils.loss_utils import LabelSmoothSoftmaxCE
# from ......utils.lovasz_losses import lovasz_softmax
# from pcdet.losses import Losses
# from range_utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
# from .range_utils import resample_grid_stacked
import torch
from torch.nn import functional as F1
# import range_utils.nn.functional as rnf
import torch
import torchsparse.nn.functional as F
from torchsparse import PointTensor, SparseTensor
from torchsparse.nn.utils import get_kernel_offsets
import os
# __all__ = ['initial_voxelize', 'point_to_voxel', 'voxel_to_point']


# z: PointTensor
# return: SparseTensor
def initial_voxelize(z, init_res, after_res):
    new_float_coord = torch.cat(
        [(z.C[:, :3] * init_res) / after_res, z.C[:, -1].view(-1, 1)], 1)

    pc_hash = F.sphash(torch.floor(new_float_coord).int())
    sparse_hash = torch.unique(pc_hash)
    idx_query = F.sphashquery(pc_hash, sparse_hash)
    counts = F.spcount(idx_query.int(), len(sparse_hash))

    inserted_coords = F.spvoxelize(torch.floor(new_float_coord), idx_query,
                                   counts)
    inserted_coords = torch.round(inserted_coords).int()
    inserted_feat = F.spvoxelize(z.F, idx_query, counts)

    new_tensor = SparseTensor(inserted_feat, inserted_coords, 1)
    new_tensor.cmaps.setdefault(new_tensor.stride, new_tensor.coords)
    z.additional_features['idx_query'][1] = idx_query
    z.additional_features['counts'][1] = counts
    z.C = new_float_coord

    return new_tensor


# x: SparseTensor, z: PointTensor
# return: SparseTensor
def point_to_voxel(x, z):
    if z.additional_features is None or z.additional_features.get(
            'idx_query') is None or z.additional_features['idx_query'].get(
                x.s) is None:
        pc_hash = F.sphash(
            torch.cat([
                torch.floor(z.C[:, :3] / x.s[0]).int() * x.s[0],
                z.C[:, -1].int().view(-1, 1)
            ], 1))
        sparse_hash = F.sphash(x.C)
        idx_query = F.sphashquery(pc_hash, sparse_hash)
        counts = F.spcount(idx_query.int(), x.C.shape[0])
        z.additional_features['idx_query'][x.s] = idx_query
        z.additional_features['counts'][x.s] = counts
    else:
        idx_query = z.additional_features['idx_query'][x.s]
        counts = z.additional_features['counts'][x.s]

    inserted_feat = F.spvoxelize(z.F, idx_query, counts)
    new_tensor = SparseTensor(inserted_feat, x.C, x.s)
    new_tensor.cmaps = x.cmaps
    new_tensor.kmaps = x.kmaps

    return new_tensor


# x: SparseTensor, z: PointTensor
# return: PointTensor
def voxel_to_point(x, z, nearest=False):
    if z.idx_query is None or z.weights is None or z.idx_query.get(
            x.s) is None or z.weights.get(x.s) is None:
        off = get_kernel_offsets(2, x.s, 1, device=z.F.device)
        old_hash = F.sphash(
            torch.cat([
                torch.floor(z.C[:, :3] / x.s[0]).int() * x.s[0],
                z.C[:, -1].int().view(-1, 1)
            ], 1), off)
        pc_hash = F.sphash(x.C.to(z.F.device))
        idx_query = F.sphashquery(old_hash, pc_hash)
        weights = F.calc_ti_weights(z.C, idx_query,
                                    scale=x.s[0]).transpose(0, 1).contiguous()
        idx_query = idx_query.transpose(0, 1).contiguous()
        if nearest:
            weights[:, 1:] = 0.
            idx_query[:, 1:] = -1
        new_feat = F.spdevoxelize(x.F, idx_query, weights)
        new_tensor = PointTensor(new_feat,
                                 z.C,
                                 idx_query=z.idx_query,
                                 weights=z.weights)
        new_tensor.additional_features = z.additional_features
        new_tensor.idx_query[x.s] = idx_query
        new_tensor.weights[x.s] = weights
        z.idx_query[x.s] = idx_query
        z.weights[x.s] = weights

    else:
        new_feat = F.spdevoxelize(x.F, z.idx_query.get(x.s), z.weights.get(x.s))
        new_tensor = PointTensor(new_feat,
                                 z.C,
                                 idx_query=z.idx_query,
                                 weights=z.weights)
        new_tensor.additional_features = z.additional_features

    return new_tensor

save_ceph = False
if save_ceph:
    from petrel_client.client import Client
    ceph_client = Client()

__all__ = ['SPVCNN']


class SyncBatchNorm(nn.SyncBatchNorm):

    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)


class BasicConvolutionBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride),
            SyncBatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class BasicDeconvolutionBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        stride=stride,
                        transposed=True),
            SyncBatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride),
            SyncBatchNorm(outc),
            spnn.ReLU(True),
            spnn.Conv3d(outc, outc, kernel_size=ks, dilation=dilation,
                        stride=1),
            SyncBatchNorm(outc),
        )

        if inc == outc * self.expansion and stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                spnn.Conv3d(inc, outc * self.expansion, kernel_size=1, dilation=1,
                            stride=stride),
                SyncBatchNorm(outc * self.expansion),
            )

        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc, outc, 1, bias=False),
            SyncBatchNorm(outc),
            spnn.Conv3d(outc, outc, ks, stride, bias=False, dilation=dilation),
            SyncBatchNorm(outc),
            spnn.Conv3d(outc, outc * self.expansion, 1, bias=False),
            SyncBatchNorm(outc * self.expansion)
        )

        if inc == outc * self.expansion and stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                spnn.Conv3d(inc, outc * self.expansion, kernel_size=1, dilation=1,
                            stride=stride),
                SyncBatchNorm(outc * self.expansion),
            )

        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out




class BaseSegmentor(nn.Module):
    def __init__(self, model_cfg, num_class):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_class = num_class
        # self.dataset = dataset
        # self.class_names = dataset.class_names

    def load_params(self, model_state_disk, strict=False):
        my_model_dict = self.state_dict()
        part_load = {}
        for k in model_state_disk.keys():
            value = model_state_disk[k]
            if k.startswith("module."):
                k = k[len("module."):]
            if k in my_model_dict and my_model_dict[k].shape == value.shape:
                part_load[k] = value

        return self.load_state_dict(part_load, strict=strict)

    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError
        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        model_state_disk = torch.load(filename, map_location=loc_type)
        if 'model_state' in model_state_disk:
            model_state_disk = model_state_disk['model_state']
        msg = self.load_params(model_state_disk)
        logger.info(f"==> Done {msg}")

    def forward(self, batch_dict):
        raise NotImplementedError

class SPVCNN(nn.Module):

    def _make_layer(self, block, out_channels, num_block, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride=stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_block):
            layers.append(block(self.in_channels, out_channels))
        return layers

    # (self, in_channels, out_channels, config, D=3):
    # def __init__(self, model_cfg, num_class, dataset=None):
    def __init__(self, in_channels, num_class, config):
        super().__init__()
        self.name = "spvcnn"
        self.in_feature_dim = in_channels
        self.num_class = num_class
        self.config = config

        # Default is MinkUNet50
        # self.num_layer = model_cfg.get('NUM_LAYER', [2, 3, 4, 6, 2, 2, 2, 2])
        # [2, 3, 4, 6, 2, 2, 2, 2]
        self.num_layer = [2, 2, 2, 2, 2, 2, 2, 2]
        # self.num_layer = [2, 3, 4, 6, 2, 2, 2, 2]
        self.block = ResidualBlock
        # self.block = {
        #     'ResBlock': ResidualBlock,
        #     'Bottleneck': Bottleneck,
        # }[model_cfg.get('BLOCK', 'Bottleneck')]
        cr = 1
        # cs = model_cfg.get('PLANES', [32, 32, 64, 128, 256, 256, 128, 96, 96])
        cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        cs = [int(cr * x) for x in cs]

        self.pres = 0.05
        self.vres = 0.05

        self.stem = nn.Sequential(
            spnn.Conv3d(self.in_feature_dim, cs[0], kernel_size=3, stride=1),
            SyncBatchNorm(cs[0]), spnn.ReLU(True),
            spnn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1),
            SyncBatchNorm(cs[0]), spnn.ReLU(True))

        self.in_channels = cs[0]
        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(self.in_channels, self.in_channels, ks=2, stride=2, dilation=1),
            *self._make_layer(self.block, cs[1], self.num_layer[0]),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(self.in_channels, self.in_channels, ks=2, stride=2, dilation=1),
            *self._make_layer(self.block, cs[2], self.num_layer[1]),
        )

        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(self.in_channels, self.in_channels, ks=2, stride=2, dilation=1),
            *self._make_layer(self.block, cs[3], self.num_layer[2]),
        )
        
        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(self.in_channels, self.in_channels, ks=2, stride=2, dilation=1),
            *self._make_layer(self.block, cs[4], self.num_layer[3]),
        )

        self.up1 = [BasicDeconvolutionBlock(self.in_channels, cs[5], ks=2, stride=2)]
        self.in_channels = cs[5] + cs[3] * self.block.expansion
        self.up1.append(nn.Sequential(*self._make_layer(self.block, cs[5], self.num_layer[4])))
        self.up1 = nn.ModuleList(self.up1)

        self.up2 = [BasicDeconvolutionBlock(self.in_channels, cs[6], ks=2, stride=2)]
        self.in_channels = cs[6] + cs[2] * self.block.expansion
        self.up2.append(nn.Sequential(*self._make_layer(self.block, cs[6], self.num_layer[5])))
        self.up2 = nn.ModuleList(self.up2)

        self.up3 = [BasicDeconvolutionBlock(self.in_channels, cs[7], ks=2, stride=2)]
        self.in_channels = cs[7] + cs[1] * self.block.expansion
        self.up3.append(nn.Sequential(*self._make_layer(self.block, cs[7], self.num_layer[6])))
        self.up3 = nn.ModuleList(self.up3)

        self.up4 = [BasicDeconvolutionBlock(self.in_channels, cs[8], ks=2, stride=2)]
        self.in_channels = cs[8] + cs[0]
        self.up4.append(nn.Sequential(*self._make_layer(self.block, cs[8], self.num_layer[7])))
        self.up4 = nn.ModuleList(self.up4)

        # self.multi_scale = self.model_cfg.get('MULTI_SCALE', 'concat')
        self.multi_scale = 'concat'
        if self.multi_scale == 'concat':
            self.classifier = nn.Sequential(nn.Linear((cs[4] + cs[6] + cs[8]) * self.block.expansion, self.num_class))
        elif self.multi_scale == 'sum':
            raise Exception('obsolete')
            self.l1 = nn.Linear(cs[4] * self.block.expansion, cs[8] * self.block.expansion)
            self.l2 = nn.Linear(cs[6] * self.block.expansion, cs[8] * self.block.expansion)
            self.classifier = nn.Sequential(nn.Linear(cs[8] * self.block.expansion + (23 if self.concatattheend else 0), self.num_class))
        elif self.multi_scale == 'se':
            raise Exception('obsolete')
            self.pool = nn.AdaptiveMaxPool1d(1)
            self.attn = nn.Sequential(
                nn.Linear((cs[4] + cs[6] + cs[8])  * self.block.expansion + (23 if self.concatattheend else 0), cs[8] * self.block.expansion, bias=False),
                nn.ReLU(True),
                nn.Linear(cs[8] * self.block.expansion, (cs[4] + cs[6] + cs[8])  * self.block.expansion + (23 if self.concatattheend else 0), bias=False),
                nn.Sigmoid(),
            )
            self.classifier = nn.Sequential(nn.Linear((cs[4] + cs[6] + cs[8]) * self.block.expansion + (23 if self.concatattheend else 0), self.num_class))
        else:
            self.classifier = nn.Sequential(nn.Linear(cs[8] * self.block.expansion + (23 if self.concatattheend else 0), self.num_class))

        self.point_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cs[0], cs[4] * self.block.expansion),
                nn.SyncBatchNorm(cs[4] * self.block.expansion),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[4] * self.block.expansion, cs[6] * self.block.expansion),
                nn.SyncBatchNorm(cs[6] * self.block.expansion),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[6] * self.block.expansion, cs[8] * self.block.expansion),
                nn.SyncBatchNorm(cs[8] * self.block.expansion),
                nn.ReLU(True),
            )
        ])

        self.weight_initialization()

        dropout_p = 0.0  #model_cfg.get('DROPOUT_P', 0.3)
        self.dropout = nn.Dropout(dropout_p, True)
        
        # label_smoothing = model_cfg.get('LABEL_SMOOTHING', 0.0)

        # # loss
        # default_loss_config = {
        #     # 'LOSS_TYPES': ['GeoLoss', 'LovLoss'],
        #     'LOSS_TYPES': ['CELoss', 'LovLoss'],
        #     # 'LOSS_WEIGHTS': [1.5, 1.0],
        #     'LOSS_WEIGHTS': [1.0, 1.0],
        #     'KNN': 10,
        #     'CLASS_NUM_POINTS': [0, 287609107, 34238008, 24868094, 14020293, 79450, 990565,
        #                         25893304, 19767528, 2510103, 36863257, 1858221, 386979,
        #                         575902, 1177084619, 731660133, 77525690, 43779989,
        #                         743066542, 20518725, 18302669, 271272135, 192753579]}
        # loss_config = self.model_cfg.get('LOSS_CONFIG', default_loss_config)
        #
        # loss_types = loss_config.get('LOSS_TYPES', default_loss_config['LOSS_TYPES'])
        # loss_weights = loss_config.get('LOSS_WEIGHTS', default_loss_config['LOSS_WEIGHTS'])
        # assert len(loss_types) == len(loss_weights)
        # class_num_points = loss_config.get('CLASS_NUM_POINTS', default_loss_config['CLASS_NUM_POINTS'])
        # k_nearest_neighbors = loss_config.get('KNN', default_loss_config['KNN'])
        # self.criterion_losses = Losses(loss_types=loss_types,
        #                                 loss_weights=loss_weights,
        #                                 cls_num_pts=class_num_points,
        #                                 ignore_index=model_cfg.IGNORE_LABEL,
        #                                 knn=k_nearest_neighbors,
        #                                 label_smoothing=label_smoothing)

        self.text_embeddings_path = self.config['text_embeddings_path']
        text_categories = self.config['text_categories']
        if self.text_embeddings_path is None:
            self.text_embeddings = nn.Parameter(torch.zeros(text_categories, 512))
            nn.init.normal_(self.text_embeddings, mean=0.0, std=0.01)
        else:
            self.register_buffer('text_embeddings', torch.randn(text_categories, 512))
            loaded = torch.load(self.text_embeddings_path, map_location='cuda')
            self.text_embeddings[:, :] = loaded[:, :]
        self.text_embeddings = torch.cat((self.text_embeddings[0, :].unsqueeze(0)*0, self.text_embeddings), dim=0)

        self.point_mapping_local = nn.Linear(480, 512)
        # self.point_mapping_local = nn.Sequential(
        #     nn.Linear(480, 64),
        #     # nn.ReLU(),
        #     nn.Linear(64, 512),
        # )
        self.point_mapping_global = nn.Linear(480, 512)
        self.point_mapping_global_random = nn.Linear(480, 512)
        # self.point_mapping_global = nn.Sequential(
        #     nn.Linear(480, 64),
        #     # nn.ReLU(),
        #     nn.Linear(64, 512),
        # )


    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.SyncBatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # def forward(self, x):
    def forward(self, batch_dict, return_logit=False, return_tta=False): 
        """, previous_memory=[None, None, None, None], previous_offset=None, return_memory=False):"""

        """if self.temporal and not return_memory:
            self.eval()
            with torch.no_grad():
                for i in range(self.temporal_previous_frame):
                    previous_memory, previous_offset = self.forward(
                        batch_dict={
                            'lidar': batch_dict[f'temporal_{i}'],
                            'offset': batch_dict[f'offset_{i}'],
                        }, 
                        previous_memory=previous_memory, 
                        previous_offset=previous_offset,
                        return_memory=True,
                    )
            self.train()

        if return_memory:
            next_memory = []"""

        """save_visual.update(dict(
            image = batch_dict['images'].detach().cpu(),
        ))"""

        """if self.deep_fusion:
            B, I, C, H, W = batch_dict['images'].shape
            encoded = self.image_encoder(batch_dict['images'].reshape(B * I, C, H, W))
            #e0, e1, e2, e3, e4 = encoded
            decoded = self.image_decoder(encoded)
            #d4, d3, d2, d1, d0 = decoded
            #d3, d2, d1, d0 = self.image_model(batch_dict['images'].reshape(B * I, C, H, W))
            #d3, d2, d1, d0 = d3.detach(), d2.detach(), d1.detach(), d0.detach()

            multistage = []
            for i, feature_map in enumerate(decoded[::-1]):
                multistage.append(self.up_sample[str(i)](feature_map))
            multistage = torch.cat(multistage, dim=1)"""
        x = batch_dict
        # print(x.C.size())

        # batch_id = x.C[:,0].reshape(-1,1).clone()
        # xyz = x.C[:,1:].clone()
        # x.C = torch.cat((xyz, batch_id),dim=1)

        # x: SparseTensor z: PointTensor
        z = PointTensor(x.F, x.C.float())
        x0 = initial_voxelize(z, self.pres, self.vres)
        x0 = self.stem(x0)
        z0 = voxel_to_point(x0, z, nearest=False)
        z0.F = z0.F

        x1 = point_to_voxel(x0, z0)
        x1 = self.stage1(x1)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        z1 = voxel_to_point(x4, z0)
        z1.F = z1.F + self.point_transforms[0](z0.F)

        y1 = point_to_voxel(x4, z1)
        y1.F = self.dropout(y1.F)
        y1 = self.up1[0](y1)
        y1 = torchsparse.cat([y1, x3])
        y1 = self.up1[1](y1)

        y2 = self.up2[0](y1)
        y2 = torchsparse.cat([y2, x2])
        y2 = self.up2[1](y2)
        z2 = voxel_to_point(y2, z1)
        z2.F = z2.F + self.point_transforms[1](z1.F)

        y3 = point_to_voxel(y2, z2)
        y3.F = self.dropout(y3.F)
        y3 = self.up3[0](y3)
        y3 = torchsparse.cat([y3, x1])
        y3 = self.up3[1](y3)

        y4 = self.up4[0](y3)
        y4 = torchsparse.cat([y4, x0])
        y4 = self.up4[1](y4)
        z3 = voxel_to_point(y4, z2)
        z3.F = z3.F + self.point_transforms[2](z2.F)

        # import pdb
        # pdb.set_trace()


        if self.multi_scale == 'concat':
            # return torch.cat([z1.F, z2.F, z3.F], dim=1)
            feat = torch.cat([z1.F, z2.F, z3.F], dim=1)
            if self.config['mode'] == 'pretrain':
                point_local = self.point_mapping_local(feat)
                point_global = self.point_mapping_global(feat)
                # text_embeddings_tmp = self.text_embeddings[1:, :]
                # out = F1.conv1d(point_local.unsqueeze(-1), text_embeddings_tmp[:, :, None]).squeeze()
                return point_local, point_global

            elif self.config['mode'] == 'finetune':
                # feat = feat / feat.norm(dim=1, keepdim=True)
                out = self.classifier(feat)
                # out = F.conv2d(feat, self.text_embeddings[:, :, None, None])
                # out = self.classifier(feat)
                return out
            elif self.config['mode'] == 'source_free':

                feat = self.point_mapping_global(feat)
                # feat = feat / feat.norm(dim=1, keepdim=True)
                out = F1.conv1d(feat.unsqueeze(-1), self.text_embeddings[:, :, None]).squeeze()

                # out = torch.matmul(feat, self.text_embeddings.permute(1, 0))

                # out = F1.conv2d(feat, self.text_embeddings[:, :, None, None])
                # out = self.classifier(feat)
                return out
            elif self.config['mode'] == 'zero_shot':


                feat = self.point_mapping_global(feat)
                # feat = self.point_mapping_global_random(feat)
                # feat = feat / feat.norm(dim=1, keepdim=True)
                out = F1.conv1d(feat.unsqueeze(-1), self.text_embeddings[:, :, None]).squeeze()

                # out = torch.matmul(feat, self.text_embeddings.permute(1, 0))

                # out = F1.conv2d(feat, self.text_embeddings[:, :, None, None])
                # out = self.classifier(feat)
                return out
            """out = self.classifier(self.last_transform([
                z3.C[:, :3].contiguous(),
                torch.cat([z1.F, z2.F, z3.F], dim=1),
                batch_dict['offset']
            ]))"""
        elif self.multi_scale == 'sum':
            out = self.classifier(self.l1(z1.F) + self.l2(z2.F) + z3.F)
        elif self.multi_scale == 'se':
            attn = torch.cat([z1.F, z2.F, z3.F], dim=1)
            attn = self.pool(attn.permute(1, 0)).permute(1, 0)
            attn = self.attn(attn)
            out = self.classifier(torch.cat([z1.F, z2.F, z3.F], dim=1) * attn)
        else:
            out = self.classifier(z3.F)

        # if self.training:
        #     target = batch_dict['targets'].F.long().cuda(non_blocking=True)
        #
        #     coords_xyz = batch_dict['lidar'].C[:, :3].float()
        #     offset = batch_dict['offset']
        #     loss = self.criterion_losses(out, target, xyz=coords_xyz, offset=offset)
        #
        #     ret_dict = {'loss': loss}
        #     disp_dict = {'loss': loss.item()}
        #     tb_dict = {'loss': loss.item()}
        #     return ret_dict, tb_dict, disp_dict
        # else:
        #     """if self.tta and not return_tta:
        #         returns = [self.forward({
        #             'lidar': batch_dict[f'augmented_lidar_{k.split("_")[-1]}'],
        #             'targets_mapped': batch_dict[f'augmented_targets_mapped_{k.split("_")[-1]}'], # dummy
        #             'inverse_map': batch_dict[f'augmented_inverse_map_{k.split("_")[-1]}'],
        #             'num_points': batch_dict['num_points'],
        #             'offset': batch_dict[f'augmented_offset_{k.split("_")[-1]}'],
        #         }, return_tta=True) for k in batch_dict if k.startswith('augmented_lidar')]
        #
        #         point_labels = returns[0]['point_labels']
        #         for i in range(1, len(returns)):
        #             for j in range(len(returns[0]['point_predict'])):
        #                 returns[0]['point_predict'][j] += returns[i]['point_predict'][j]
        #         for j in range(len(returns[0]['point_predict'])): # normalise
        #             returns[0]['point_predict'][j] /= len(returns)
        #
        #         if return_logit or self.store_logits:
        #             point_predict = [logit for logit in returns[0]['point_predict']]
        #         else:
        #             point_predict = [logit.argmax(1) for logit in returns[0]['point_predict']]
        #
        #     else:"""
        #
        #
        #     invs = batch_dict['inverse_map']
        #     all_labels = batch_dict['targets_mapped']
        #     point_predict = []
        #     point_labels = []
        #     for idx in range(invs.C[:, -1].max() + 1):
        #         cur_scene_pts = (x.C[:, -1] == idx).cpu().numpy()
        #         cur_inv = invs.F[invs.C[:, -1] == idx].cpu().numpy()
        #         cur_label = (all_labels.C[:, -1] == idx).cpu().numpy()
        #         if return_logit or return_tta:
        #             outputs_mapped = out[cur_scene_pts][cur_inv].softmax(1)
        #         else:
        #             outputs_mapped = out[cur_scene_pts][cur_inv].argmax(1)
        #         targets_mapped = all_labels.F[cur_label]
        #         point_predict.append(outputs_mapped[:batch_dict['num_points'][idx]].cpu().numpy())
        #         point_labels.append(targets_mapped[:batch_dict['num_points'][idx]].cpu().numpy())
        #
        #     """### save to ceph
        #     if save_ceph:
        #         for i, name in enumerate(batch_dict['name']):
        #             filename = name
        #             gt = point_labels[i]
        #             pred = torch.tensor(point_predict[i]).softmax(dim=1)
        #             pred_score, pred_label = pred.max(dim=1)
        #             pred_score = pred_score.cpu().numpy()
        #             pred_label = pred_label.cpu().numpy()
        #             pred = pred.cpu().numpy()
        #
        #             ret = {
        #                 'filename': filename,
        #                 'gt_score': pred[np.arange(gt.shape[0]), gt],
        #                 'gt_label': gt,
        #                 'pred_score': pred_score,
        #                 'pred_label': pred_label
        #             }
        #             ceph_client.put(f's3://waymo_v_1_3/infer/trainval/{name}.pkl', pickle.dumps(ret))
        #
        #     ### save to ceph
        #     if self.store_logits:
        #         assert (self.store_config != None) and (self.store_root != None) \
        #                 and (self.model_name != None)
        #
        #         for i, name in enumerate(batch_dict['name']):
        #             pred = torch.tensor(point_predict[i]).cpu().numpy()
        #             pred_ = pred * np.iinfo(np.uint16).max
        #             pred_uint16 = pred_.astype(np.uint16)
        #             self.ceph_client.put(f'{self.store_root}/{self.store_config}/{self.model_name}/val/{name}',
        #                                 pred_uint16.tobytes())
        #
        #             point_predict[i] = point_predict[i].argmax(1)"""
        #
        #     return {'point_predict': point_predict, 'point_labels': point_labels, 'name': batch_dict['name']}
        return out

    def forward_ensemble(self, batch_dict):
        return self.forward(batch_dict, ensemble=True)
