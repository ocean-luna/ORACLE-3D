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
class fusionNet(nn.Module):
    """
    Dilated ResNet Feature Extractor
    """

    def __init__(self, config):
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

        self.img_size = (224, 416)

        # todo t = 8
        self.t = 1


    def forward(self, feature_packages):

        # feature_packages size: voxelSize * 8 * 1537
        # pixel_feature, point_feature, text_embedding, pred = feature_packages[:, :, :512], feature_packages[:, :, 512:1024], feature_packages[:, :, 1024:1536], feature_packages[:, :, -1]
        pixel_feature, point_feature, pred = feature_packages[:, :, :512], feature_packages[:, :, 512:1024], feature_packages[:, :, -1]

        # todo max voting
        pixel_pred = pred[:, 0].long()
        text_embedding = self.text_embeddings[pixel_pred].unsqueeze(1)

        # text_embedding = torch.cat((text_embedding.unsqueeze(1), text_embedding.unsqueeze(1)), dim=1)
        # pixel_point_feature = torch.cat((pixel_feature.unsqueeze(1), point_feature.unsqueeze(1)), dim=1)
        pixel_point_feature = point_feature
        pixel_point_attention = torch.sum(pixel_point_feature * text_embedding, dim=2)

        # pixel_point_attention = torch.flatten(pixel_point_attention, start_dim=1, end_dim=-1)

        # attention_pixel = torch.sum(pixel_feature * text_embedding, dim=2)
        # attention_point = torch.sum(point_feature * text_embedding, dim=2)

        # index_pixel_sum = torch.sum(attention_pixel, dim=1) != 0
        index_point_sum = torch.sum(pixel_point_attention, dim=1) != 0
        pixel_point_attention = pixel_point_attention[index_point_sum] / self.t
        pixel_point_feature = pixel_point_feature[index_point_sum]
        pixel_pred = pixel_pred[index_point_sum]
        # pixel_point_feature = torch.flatten(pixel_point_feature, start_dim=1, end_dim=2)
        # assert index_pixel_sum == index_point_sum
        # attention_pixel = attention_pixel[index_point_sum]
        # attention_point = attention_point[index_point_sum]

        # attention_union = torch.cat((attention_pixel, attention_point), dim=1) / self.t
        attention_union_sparse = pixel_point_attention.to_sparse()
        attention_union_dense = torch.sparse.softmax(attention_union_sparse, dim=1).to_dense()

        # feature_union = torch.cat((pixel_feature, point_feature), dim=1)
        # feature_union = feature_union[index_point_sum]

        fusion_feature = torch.sum(attention_union_dense.unsqueeze(-1) * pixel_point_feature, dim=1)
        inner_products = torch.sigmoid(torch.sum(fusion_feature.unsqueeze(1) * pixel_point_feature, dim=2))

        return fusion_feature, inner_products, pixel_pred
# #
# class fusionNet(nn.Module):
#     """
#     Dilated ResNet Feature Extractor
#     """
#
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.text_embeddings_path = self.config['text_embeddings_path']
#         text_categories = self.config['text_categories']
#         if self.text_embeddings_path is None:
#             self.text_embeddings = nn.Parameter(torch.zeros(text_categories, 512))
#             nn.init.normal_(self.text_embeddings, mean=0.0, std=0.01)
#         else:
#             self.register_buffer('text_embeddings', torch.randn(text_categories, 512))
#             loaded = torch.load(self.text_embeddings_path, map_location='cuda')
#             self.text_embeddings[:, :] = loaded[:, :]
#
#         self.img_size = (224, 416)
#         self.t = 1
#
#
#     def forward(self, feature_packages):
#
#         # feature_packages size: voxelSize * 8 * 1537
#         # pixel_feature, point_feature, text_embedding, pred = feature_packages[:, :, :512], feature_packages[:, :, 512:1024], feature_packages[:, :, 1024:1536], feature_packages[:, :, -1]
#         pixel_feature, point_feature, pred = feature_packages[:, :, :512], feature_packages[:, :, 512:1024], feature_packages[:, :, -1]
#
#         pixel_pred = pred[:, 0].long()
#         text_embedding = self.text_embeddings[pixel_pred].unsqueeze(1).unsqueeze(1)
#
#         # text_embedding = torch.cat((text_embedding.unsqueeze(1), text_embedding.unsqueeze(1)), dim=1)
#         pixel_point_feature = torch.cat((pixel_feature.unsqueeze(1), point_feature.unsqueeze(1)), dim=1)
#         pixel_point_attention = torch.sum(pixel_point_feature * text_embedding, dim=3)
#
#         pixel_point_attention = torch.flatten(pixel_point_attention, start_dim=1, end_dim=-1)
#
#         # attention_pixel = torch.sum(pixel_feature * text_embedding, dim=2)
#         # attention_point = torch.sum(point_feature * text_embedding, dim=2)
#
#         # index_pixel_sum = torch.sum(attention_pixel, dim=1) != 0
#         index_point_sum = torch.sum(pixel_point_attention, dim=1) != 0
#         pixel_point_attention = pixel_point_attention[index_point_sum] / self.t
#         pixel_point_feature = pixel_point_feature[index_point_sum]
#         pixel_pred = pixel_pred[index_point_sum]
#         pixel_point_feature = torch.flatten(pixel_point_feature, start_dim=1, end_dim=2)
#         # assert index_pixel_sum == index_point_sum
#         # attention_pixel = attention_pixel[index_point_sum]
#         # attention_point = attention_point[index_point_sum]
#
#         # attention_union = torch.cat((attention_pixel, attention_point), dim=1) / self.t
#         attention_union_sparse = pixel_point_attention.to_sparse()
#         attention_union_dense = torch.sparse.softmax(attention_union_sparse, dim=1).to_dense()
#
#         # feature_union = torch.cat((pixel_feature, point_feature), dim=1)
#         # feature_union = feature_union[index_point_sum]
#
#         fusion_feature = torch.sum(attention_union_dense.unsqueeze(-1) * pixel_point_feature, dim=1)
#         inner_products = torch.sigmoid(torch.sum(fusion_feature.unsqueeze(1) * pixel_point_feature, dim=2))
#
#         return fusion_feature, inner_products, pixel_pred
# #
# class fusionNet(nn.Module):
#     """
#     Dilated ResNet Feature Extractor
#     """
#
#     def __init__(self, config):
#         super().__init__()
#         self.prototype_dimension = 64
#         self.prototype1_num = config["prototype_num"]
#         self.t = 0.5
#         self.prototype1 = nn.Parameter(torch.zeros(size=(self.prototype1_num, 64)))
#         nn.init.kaiming_uniform_(self.prototype1.data, a=math.sqrt(5))
#
#         self.k1 = nn.Parameter(torch.zeros(size=(64, 16)))
#         nn.init.kaiming_uniform_(self.k1.data, a=math.sqrt(5))
#
#         self.q1 = nn.Parameter(torch.zeros(size=(64, 16)))
#         nn.init.kaiming_uniform_(self.q1.data, a=math.sqrt(5))
#
#
#         self.prototype2_num = 128
#         self.prototype2 = nn.Parameter(torch.zeros(size=(self.prototype2_num, 64)))
#         nn.init.kaiming_uniform_(self.prototype2.data, a=math.sqrt(5))
#
#         self.k2 = nn.Parameter(torch.zeros(size=(64, 16)))
#         nn.init.kaiming_uniform_(self.k2.data, a=math.sqrt(5))
#
#         self.q2 = nn.Parameter(torch.zeros(size=(64, 16)))
#         nn.init.kaiming_uniform_(self.q2.data, a=math.sqrt(5))
#
#
#         self.prototype3_num = 32
#         self.prototype3 = nn.Parameter(torch.zeros(size=(self.prototype3_num, 64)))
#         nn.init.kaiming_uniform_(self.prototype3.data, a=math.sqrt(5))
#
#         self.k3 = nn.Parameter(torch.zeros(size=(64, 16)))
#         nn.init.kaiming_uniform_(self.k3.data, a=math.sqrt(5))
#
#         self.q3 = nn.Parameter(torch.zeros(size=(64, 16)))
#         nn.init.kaiming_uniform_(self.q3.data, a=math.sqrt(5))
#
#         self.kk1 = nn.Parameter(torch.zeros(size=(64, 3)))
#         nn.init.kaiming_uniform_(self.kk1.data, a=math.sqrt(5))
#
#         self.qq1 = nn.Parameter(torch.zeros(size=(64, 3)))
#         nn.init.kaiming_uniform_(self.qq1.data, a=math.sqrt(5))
#
#
#     def forward(self, points_f, images_f):
#         simiI1 = torch.matmul(images_f, self.prototype1.permute(1, 0))
#         # simiI2 = torch.matmul(images_f, self.prototype2.permute(1, 0))
#         # simiI3 = torch.matmul(images_f, self.prototype3.permute(1, 0))
#
#         simiP1 = torch.matmul(points_f, self.prototype1.permute(1, 0))
#         # simiP2 = torch.matmul(points_f, self.prototype2.permute(1, 0))
#         # simiP3 = torch.matmul(points_f, self.prototype3.permute(1, 0))
#
#
#         # simiI1 = torch.matmul(torch.matmul(images_f, self.k1), torch.matmul(self.prototype1, self.q1).permute(1, 0))
#         # simiI2 = torch.matmul(torch.matmul(images_f, self.k2), torch.matmul(self.prototype2, self.q2).permute(1, 0))
#         # simiI3 = torch.matmul(torch.matmul(images_f, self.k3), torch.matmul(self.prototype3, self.q3).permute(1, 0))
#
#         cluster1 = torch.argmax(simiI1, dim=1)
#         # cluster2 = torch.argmax(simiI2, dim=1)
#         # cluster3 = torch.argmax(simiI3, dim=1)
#         # attentions = torch.matmul(feature, self.base.permute(1, 0))
#         # attentions = torch.softmax(attentions / self.t, dim=1)
#
#         prototype_reduce = torch.matmul(self.prototype1, self.qq1)
#         images_f_reduce = torch.matmul(images_f, self.kk1)
#
#         return points_f, images_f_reduce, simiI1, simiP1, cluster1, prototype_reduce, self.prototype1
#
