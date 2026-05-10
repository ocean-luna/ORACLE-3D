import os
import re
import torch
import time
import numpy as np
import torch.optim as optim
import MinkowskiEngine as ME
import pytorch_lightning as pl
from utils.chamfer_distance import ComputeCDLoss
from pretrain.criterion import NCELoss, DistillKL, semantic_NCELoss
from pytorch_lightning.utilities import rank_zero_only
# from torchsparse import SparseTensor
# from torchsparse.nn import functional as sparseF
# from torchsparse.utils.collate import sparse_collate_fn
# from torchsparse.utils.quantize import sparse_quantize

# import pc_utils
from plyfile import PlyData, PlyElement
# import scannet_utils
import math
# from pc_utils import write_ply_rgb

from torch import nn
import torch.nn.functional as F
import random
import numba as nb
from utils.metrics import confusion_matrix, compute_IoU_from_cmatrix
from utils.metrics import compute_IoU
from utils.scannet_utils import create_color_palette as create_color_palette
import datetime
import matplotlib.pyplot as plt

@nb.jit()
def nb_pack(counts):
    return [np.array(list(range(i))) for i in counts]


def show_anns(img, masks, mask_data, clip_orig, deeplab, mask_labels, image_name, seeds_index):
    img = img.cpu().contiguous().numpy()
    rand_color = (np.random.rand(10000, 3) * 255).astype(int)
    img_vis = np.zeros_like(img)
    # num_masks = len(mask_data['segmentation'])
    plt.imsave("visual/%s.png" % image_name, img)

    for i, mask in enumerate(masks):
        seg_i = mask['segmentation']
        seg_color_map = rand_color[i][None, None, :] * seg_i[:, :, None]
        img_vis += seg_color_map

    img_vis = np.clip(img_vis, 0, 255)
    img_vis = img_vis / 255.0
    img_vis = img_vis * 0.35 + img * 0.65

    plt.imsave("visual/%s_sam.png" % (image_name), img_vis)

    color_template = create_color_palette()
    rand_color = color_template

    img_vis = np.zeros_like(img)
    img_vis_label = np.zeros_like(img)
    img_vis_oriclip = np.zeros_like(img)
    img_vis_deeplab = np.zeros_like(img)

    for i, mask in enumerate(mask_data):
        # seg_i = mask['segmentation']

        # seg_color_map = rand_color[i][None, None, :] * seg_i[:, :, None]
        # img_vis += seg_color_map
        #
        # seg_i = mask_labels[i]['segmentation']
        # seg_color_map = rand_color[i][None, None, :] * seg_i[:, :, None]
        # img_vis_label += seg_color_map
        #
        # seg_i = clip_orig[i]['segmentation']
        # seg_color_map = rand_color[i][None, None, :] * seg_i[:, :, None]
        # img_vis_oriclip += seg_color_map
        #
        # seg_i = deeplab[i]['segmentation']
        # seg_color_map = rand_color[i][None, None, :] * seg_i[:, :, None]
        # img_vis_deeplab += seg_color_map



        seg_i = mask['segmentation']
        img_vis[seg_i] = color_template[i]

        # seg_i = mask_labels[i]['segmentation']
        # img_vis_label[seg_i] = color_template[i]

        seg_i = clip_orig[i]['segmentation']
        img_vis_oriclip[seg_i] = color_template[i]

        seg_i = deeplab[i]['segmentation']
        img_vis_deeplab[seg_i] = color_template[i]



    img_vis = np.clip(img_vis, 0, 255)
    img_vis = img_vis / 255.0
    img_vis = img_vis * 0.35 + img * 0.65
    plt.imsave("visual/%s_clip.png" % (image_name), img_vis)

    #logical_not()
    # img_vis[~(seeds_index.cpu())] = 0
    # img_vis[(seeds_index.cpu())] = img[(seeds_index.cpu())]
    # plt.imsave("visual/%s_seeds.png" % (image_name), img_vis)

    # img_vis_label = np.clip(img_vis_label, 0, 255)
    # img_vis_label = img_vis_label / 255.0
    # img_vis_label = img_vis_label * 0.35 + img * 0.65
    # plt.imsave("visual/%s_label.png" % (image_name), img_vis_label)

    img_vis_oriclip = np.clip(img_vis_oriclip, 0, 255)
    img_vis_oriclip = img_vis_oriclip / 255.0
    img_vis_oriclip = img_vis_oriclip * 0.35 + img * 0.65
    plt.imsave("visual/%s_orgclip.png" % (image_name), img_vis_oriclip)

    img_vis_deeplab = np.clip(img_vis_deeplab, 0, 255)
    img_vis_deeplab = img_vis_deeplab / 255.0
    img_vis_deeplab = img_vis_deeplab * 0.35 + img * 0.65
    plt.imsave("visual/%s_deeplab.png" % (image_name), img_vis_deeplab)

    # img_vis_oriclip[~(seeds_index.cpu())] = 0
    # img_vis[(seeds_index.cpu())] = img[(seeds_index.cpu())]
    # plt.imsave("visual/%s_seeds_oriclip.png" % (image_name), img_vis_oriclip)


def visual_masks(image, masks_sam, output_images, clip_orig, deeplab_pred, img_labels, image_name, seeds_index):

    mask_clip = []
    mask_labels = []
    mask_cliporig = []
    mask_deeplab = []

    for i in range(21):
        index = output_images == i
        mask_clip.append({'segmentation': index.detach().cpu().numpy()})

        # index = img_labels == i
        # mask_labels.append({'segmentation': index.detach().cpu().numpy()})

        index = clip_orig == i
        mask_cliporig.append({'segmentation': index.detach().cpu().numpy()})

        index = deeplab_pred == i
        mask_deeplab.append({'segmentation': index.detach().cpu().numpy()})

    show_anns(image, masks_sam, mask_clip, mask_cliporig, mask_deeplab, mask_labels, image_name, seeds_index)



def write_ply_rgb(points, colors, filename, text=True):
    """ input: Nx3, Nx3 write points and colors to filename as PLY format. """
    num_points = len(points)
    assert len(colors) == num_points

    points = [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])]
    colors = [(colors[i, 0], colors[i, 1], colors[i, 2]) for i in range(colors.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    color = np.array(colors, dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(num_points, vertex.dtype.descr + color.dtype.descr)

    for prop in vertex.dtype.names:
        vertex_all[prop] = vertex[prop]

    for prop in color.dtype.names:
        vertex_all[prop] = color[prop]

    el = PlyElement.describe(vertex_all, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)



def visual_prediction(coords_scannet, feats_scannet, labels_scannet, predictions, prediction_pre, prediction_CLIPSAMs, lidar_name):

    # import pdb
    # pdb.set_trace()

    # coords_scannet = coords_scannet[:, 1:]
    random_samples = torch.randperm(coords_scannet.size()[0])
    sample_points = 200000

    predictions = predictions[random_samples[:sample_points]].long()
    prediction_pre = prediction_pre[random_samples[:sample_points]].long()
    prediction_CLIPSAMs = prediction_CLIPSAMs[random_samples[:sample_points]].long()
    coords_scannet = coords_scannet[random_samples[:sample_points]]
    feats_scannet = feats_scannet[random_samples[:sample_points]]
    labels_scannet = labels_scannet[random_samples[:sample_points]].long()

    label2color = torch.zeros(coords_scannet.size()).long()
    pred2color = torch.zeros(coords_scannet.size()).long()
    pred_pre2color = torch.zeros(coords_scannet.size()).long()
    pred_CLIPSAM2color = torch.zeros(coords_scannet.size()).long()
    input = torch.zeros(coords_scannet.size()).long()

    # heatmap = np.uint8(255*predictions.detach().cpu().numpy())
    # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # print(feats_scannet)

    color_template = create_color_palette()

    r = 0 * predictions
    g = 0 * predictions
    b = 0 * predictions

    # index = labels_scannet[:, 1].long() == scannet_utils.CLASS_LABELS.index(object_visual)

    for i in range(20):
        index_label = labels_scannet == i
        index_pred = predictions == i
        index_pred_pre = prediction_pre == i
        index_pred_CLIPSAMs = prediction_CLIPSAMs == i
        if index_label.sum() > 0:
            label2color[index_label] = torch.tensor(color_template[i]).long()
        if index_pred.sum() > 0:
            pred2color[index_pred] = torch.tensor(color_template[i]).long()
        if index_pred_pre.sum() > 0:
            pred_pre2color[index_pred_pre] = torch.tensor(color_template[i]).long()
        if index_pred_CLIPSAMs.sum() > 0:
            pred_CLIPSAM2color[index_pred_CLIPSAMs] = torch.tensor(color_template[i]).long()

    if not os.path.exists('visual_result_nuscenes_suplementary'): os.makedirs('visual_result_nuscenes_suplementary')

    write_ply_rgb(coords_scannet, label2color, 'visual_result_nuscenes_suplementary/%s_%s.ply' % (lidar_name, 'GT'), text=True)
    write_ply_rgb(coords_scannet, pred2color, 'visual_result_nuscenes_suplementary/%s_%s.ply' % (lidar_name, 'ours'), text=True)
    write_ply_rgb(coords_scannet, pred_CLIPSAM2color, 'visual_result_nuscenes_suplementary/%s_%s.ply' % (lidar_name, 'CLIPSAM'), text=True)
    # write_ply_rgb(coords_scannet, feats_scannet, 'visual_result_nuscenes_suplementary/%s_%s.ply' % (lidar_name, 'origin'), text=True)
    write_ply_rgb(coords_scannet, input, 'visual_result_nuscenes_suplementary/%s_%s.ply' % (lidar_name, 'origin'), text=True)
    write_ply_rgb(coords_scannet, pred_pre2color, 'visual_result_nuscenes_suplementary/%s_%s.ply' % (lidar_name, 'clip'), text=True)


class LightningPretrain(pl.LightningModule):
    def __init__(self, model_points, model_clip, model_images, feature_mapping, model_SAM, config):
        super().__init__()
        self.model_points = model_points
        self.model_clip = model_clip
        self.model_images = model_images
        self.feature_mapping = feature_mapping
        self.model_SAM = model_SAM
        self.config = config
        self.losses = config["losses"]
        self.train_losses = []
        self.val_losses = []
        self.num_matches = config["num_matches"]
        self.batch_size = config["batch_size"]
        self.num_epochs = config["num_epochs"]
        self.superpixel_size = config["superpixel_size"]
        self.epoch = 0
        self.cot = 0
        self.CE = nn.CrossEntropyLoss()
        self.CD_loss = ComputeCDLoss()
        self.KLloss = DistillKL(T=1)
        if config["resume_path"] is not None:
            self.epoch = int(
                re.search(r"(?<=epoch=)[0-9]+", config["resume_path"])[0]
            )
        self.criterion = NCELoss(temperature=config["NCE_temperature"])
        self.sem_NCE = semantic_NCELoss(temperature=config["NCE_temperature"])

        self.ignore_index = config["ignore_index"]
        self.working_dir = os.path.join(config["working_dir"], config["datetime"])
        if os.environ.get("LOCAL_RANK", 0) == 0:
            os.makedirs(self.working_dir, exist_ok=True)

        self.text_embeddings_path = config['text_embeddings_path']
        text_categories = config['text_categories']
        if self.text_embeddings_path is None:
            self.text_embeddings = nn.Parameter(torch.zeros(text_categories, 512))
            nn.init.normal_(self.text_embeddings, mean=0.0, std=0.01)
        else:
            self.register_buffer('text_embeddings', torch.randn(text_categories, 512))
            loaded = torch.load(self.text_embeddings_path, map_location='cuda')
            self.text_embeddings[:, :] = loaded[:, :]

        self.saved = False

        self.max_size = config["max_images"]

        self.thr = 0.8
        self.n_classes = config["n_classes"]
        # self.mask_embedding = torch.tensor(list(range(self.max_size))).cuda()
        # self.zeros = torch.zeros(self.max_size, 2000).cuda()
        # self.pre_packing_feature = torch.zeros(60000, self.max_size, 2000).cuda()
    def get_in_field(self, coords, feats):
        in_field = ME.TensorField(coordinates=coords.float(), features=feats.int(),
                                  # coordinate_map_key=A.coordiante_map_key, coordinate_manager=A.coordinate_manager,
                                  quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                                  minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                                  # minkowski_algorithm=ME.MinkowskiAlgorithm.MEMORY_EFFICIENT,
                                  # device=self.config.device,
                                  ).float()
        return in_field


    def configure_optimizers(self):
        optimizer = optim.SGD(
            list(self.model_points.parameters()) + list(self.model_clip.parameters())
            + list(self.model_images.parameters()) + list(self.feature_mapping.parameters()),
            lr=self.config["lr"],
            momentum=self.config["sgd_momentum"],
            dampening=self.config["sgd_dampening"],
            weight_decay=self.config["weight_decay"],
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.num_epochs)
        return [optimizer], [scheduler]

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)


    def process_sam(self, image_names, input_img):

        # image_name = lidar_names + "_" + frame_names[id]
        torch.cuda.empty_cache()

        for id, image_name in enumerate(image_names):
            image = input_img[id].permute(1, 2, 0)
            image = (image * 255).cpu().numpy()

            image_name = image_name.split('/')[-1]

            print(image_name, image.shape)
            save_name = "sam_preprocess/" + image_name + ".npy"

            #(320, 240)
            # print("before ", image.shape)
            # image = cv2.resize(image, self.imageDim)
            # print("after ", image.shape)

            token = save_name + "token"
            if os.path.exists(save_name):
                print("exsists")
                continue


            # try:
            #     os.mknod(token)
            # except:
            #     pass

            masks_sam = self.model_SAM.generate(image.astype("uint8"))
            np.save(save_name, masks_sam)
            # embeddings_sam = np.load(save_name, allow_pickle=True)
            # print(embeddings_sam)

            try:
                os.remove(token)
            except:
                pass

            print(type(masks_sam))
            print(len(masks_sam))


    def training_step(self, batch, batch_idx):

        self.model_points.train()

        sinput_C = batch["sinput_C"]
        sinput_F = batch["sinput_F"]
        # masks_sams = batch["masks_sams"]

        sparse_input = ME.SparseTensor(sinput_F.float(), coordinates=sinput_C.int())

        # image_names = batch["image_names"][0]
        # print(len(image_names), image_names)

        # self.process_sam(image_names, batch["input_I"])

        # print("before ", sinput_F.float().shape)
        # st=time.time()
        output_points = self.model_points(sparse_input)


        # dt = time.time()
        # print("model_points",dt-st)
        # print("after ", output_points[0].shape)

        output_images = self.model_clip(batch["input_I"].float())
        # dt2 = time.time()
        # print("model_clip", dt2 - dt)
        output_images_deeplab = self.model_images(batch["input_I"].float())
        # dt3 = time.time()
        # print("model_image", dt3 - dt2)
        # output_images = output_images_deeplab = 0
        # output_points = sinput_F, sinput_F
        # output_images_deeplab = 0


        # assert output_images_deeplab.shape == output_images.shape
        # print("output_images ", output_images.shape)
        # print("output_images_deeplab.shape ", output_images_deeplab.shape)

        # print(batch["input_I"].shape, output_images[1].shape)


        del batch["sinput_F"]
        del batch["sinput_C"]
        del batch["input_I"]
        del sparse_input
        # each loss is applied independtly on each GPU
        losses = [
            getattr(self, loss)(batch, output_points, output_images, output_images_deeplab)
            for loss in self.losses
        ]
        # dt4 = time.time()
        # print("loss", dt4 - dt3)
        loss = torch.mean(torch.stack(losses))
        # print(loss.detach().cpu())
        torch.cuda.empty_cache()
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size
        )


        self.saved = True

        self.train_losses.append(loss.detach().cpu())
        return loss


    def validation_step(self, batch, batch_idx):
        self.model_points.eval()
        self.model_images.eval()

        sparse_input = ME.SparseTensor(batch["sinput_F"].float(), coordinates=batch["sinput_C"].int())
        output_points = self.model_points(sparse_input)
        output_clip = self.model_clip(batch["input_I"].float())
        output_deeplab = self.model_images(batch["input_I"].float())

        image_logists_deeplab, image_feature_deeplab = output_deeplab
        image_preds_deeplab = image_logists_deeplab.argmax(dim=1).view(-1).int() + 1

        image_logists_clip, image_feats_clip = output_clip
        # image_preds_clip = (F.softmax(image_logists_deeplab * 100, dim=1) + F.softmax(image_logists_clip * 100, dim=1)).argmax(dim=1)
        image_preds_clip = image_logists_clip.argmax(dim=1).int() + 1

        point_logists, point_feats = output_points

        masks_sams = batch["masks_sams"][0]
        for i in range(1, len(batch["masks_sams"])):
            masks_sams.extend(batch["masks_sams"][i])

        # image_refinedpreds_clip = image_preds_clip
        image_refinedpreds_clip, corrected_areas_clip = self.label_refinement_sam(image_preds_clip, masks_sams)
        assert image_refinedpreds_clip.shape == image_preds_clip.shape
        # image_preds_clip = image_refinedpreds_clip

        # Ensure we ignore the index 0
        # (probably not necessary after some training)

        point_preds = []
        point_labels = []
        offset = 0

        if self.config["ignore_index"]:
            point_logists[:, config["ignore_index"]] = -1e6
        point_pred = point_logists.argmax(1)

        prediction_pre = torch.zeros(point_pred.shape[0]).int()
        prediction_CLIPSAM = torch.zeros(point_pred.shape[0]).int()
        pairing_points = batch["pairing_points"]
        pairing_images = batch["pairing_images"]
        k_point_logists = point_logists[pairing_points]
        m = tuple(pairing_images.T.long())
        # prediction_pre[pairing_points] = image_preds_clip[m].detach().cpu().int()
        # prediction_CLIPSAM[pairing_points] = image_refinedpreds_clip[m].detach().cpu().int()
        # prediction_pres = []
        prediction_CLIPSAMs = []

        for i, lb in enumerate(batch["len_batch"]):
            point_preds.append(point_pred[batch["inverse_indexes"][i] + offset])
            # prediction_pres.append(prediction_pre[batch["inverse_indexes"][i] + offset])
            # prediction_CLIPSAMs.append(prediction_CLIPSAM[batch["inverse_indexes"][i] + offset])
            point_labels.append(batch["evaluation_labels"][i])
            offset += lb

        point_preds = torch.cat(point_preds, dim=0).int()
        # prediction_pres = torch.cat(prediction_pres, dim=0).int()
        # prediction_CLIPSAMs = torch.cat(prediction_CLIPSAMs, dim=0).int()
        point_labels = torch.cat(point_labels, dim=0).int()
        assert point_preds.shape == point_labels.shape

        image_preds_clip = image_preds_clip.view(-1).int()

        if self.config['dataset'] == "nuscenes":
            image_labels = torch.zeros(image_preds_clip.shape).int().to(device="cuda")
        elif self.config['dataset'] == "scannet":
            image_labels = batch["imgs_labels"].view(-1).int()
        assert image_preds_clip.shape == image_labels.shape




        images_batch = batch["input_I"]
        image_names = batch["image_names"]
        deeplab_preds = image_logists_deeplab.argmax(dim=1).int() + 1
        clip_preds = image_logists_clip.argmax(dim=1).int() + 1
        clipSAM_preds = image_refinedpreds_clip
        # import pdb
        # pdb.set_trace()

        # for id in range(batch["input_I"].shape[0]):
        #     image = images_batch[id].permute(1, 2, 0)
        #     masks_sam = masks_sams[id]
        #     image_name = image_names[id]
        #     image_name = image_name.split('/')[-1]
        #     deeplab_pred = deeplab_preds[id]
        #     clip_pred = clip_preds[id]
        #     clipSAM_pred = clipSAM_preds[id]
        #     seeds_index = 0
        #
        #     visual_masks(image, masks_sam, clipSAM_pred, clip_pred, deeplab_pred, image_labels, image_name,
        #                  seeds_index)

        # point_preds, point_labels, image_preds_clip, image_preds_deeplab, image_labels

        c_matrix_point = confusion_matrix(point_preds, point_labels, self.n_classes)
        c_matrix_clip = confusion_matrix(image_preds_clip, image_labels, self.n_classes)
        c_matrix_deeplab = confusion_matrix(image_preds_deeplab, image_labels, self.n_classes)

        return c_matrix_point, c_matrix_clip, c_matrix_deeplab

        # point_preds = prediction_pres.to(device="cuda")

        # return point_preds, point_labels, image_preds_clip, image_preds_deeplab, image_labels

    def validation_epoch_end(self, outputs):
        #
        # print("len(outputs) ", len(outputs))
        # point_preds = torch.cat([o[0] for o in outputs], dim=0)
        # point_labels = torch.cat([o[1] for o in outputs], dim=0)
        # image_preds_clip = torch.cat([o[2] for o in outputs], dim=0)
        # image_preds_deeplab = torch.cat([o[3] for o in outputs], dim=0)
        # image_labels = torch.cat([o[4] for o in outputs], dim=0)

        # c_matrix_point, c_matrix_clip, c_matrix_deeplab
        c_matrix_point = sum([o[0] for o in outputs])
        c_matrix_clip = sum([o[1] for o in outputs])
        c_matrix_deeplab = sum([o[2] for o in outputs])

        c_matrix_point = torch.sum(self.all_gather(c_matrix_point), 0)
        c_matrix_clip = torch.sum(self.all_gather(c_matrix_clip), 0)
        c_matrix_deeplab = torch.sum(self.all_gather(c_matrix_deeplab), 0)

        point_m_IoU, point_fw_IoU, per_class_IoU = compute_IoU_from_cmatrix(
            c_matrix_point, self.ignore_index
        )
        #
        # point_m_IoU, point_fw_IoU, per_class_IoU = compute_IoU(
        #     point_preds,
        #     point_labels,
        #     self.config["n_classes"],
        #     ignore_index=0,
        # )

        self.log("point_m_IoU", point_m_IoU, prog_bar=True, logger=True, sync_dist=False)
        self.log("point_fw_IoU", point_fw_IoU, prog_bar=True, logger=True, sync_dist=False)

        image_m_IoU, image_fw_IoU, per_class_IoU = compute_IoU_from_cmatrix(
            c_matrix_clip, self.ignore_index
        )
        # image_m_IoU, image_fw_IoU, per_class_IoU = compute_IoU(
        #     image_preds_clip,
        #     image_labels,
        #     self.config["n_classes"],
        #     ignore_index=0,
        # )
        self.log("image_m_IoU", image_m_IoU, prog_bar=True, logger=True, sync_dist=False)
        self.log("image_fw_IoU", image_fw_IoU, prog_bar=True, logger=True, sync_dist=False)

        image_deeplab_m_IoU, image_deeplab_fw_IoU, per_class_IoU = compute_IoU_from_cmatrix(
            c_matrix_deeplab, self.ignore_index
        )
        # image_deeplab_m_IoU, image_deeplab_fw_IoU, per_class_IoU = compute_IoU(
        #     image_preds_deeplab,
        #     image_labels,
        #     self.config["n_classes"],
        #     ignore_index=0,
        # )
        self.log("image_deeplab_m_IoU", image_deeplab_m_IoU, prog_bar=True, logger=True, sync_dist=False)
        self.log("image_deeplab_fw_IoU", image_deeplab_fw_IoU, prog_bar=True, logger=True, sync_dist=False)

        # output_file_name = os.path.join(self.working_dir, "results.txt")
        # if not os.path.exists(output_file_name):
        #     try:
        #         os.mknod(output_file_name)
        #     except:
        #         pass
        #     # output_file = open(output_file_name, 'a')
        #     # print('date time: %s' % datetime.datetime.now(), file=output_file)
        #     # output_file.close()
        #
        # output_file = open(output_file_name, 'a')
        # print('epochs: %s point_m_IoU: %.4f point_fw_IoU: %.4f image_m_IoU: %.4f image_fw_IoU: %.4f image_deeplab_m_IoU: %.4f image_deeplab_fw_IoU: %.4f' %
        #       (self.epoch, point_m_IoU, point_fw_IoU, image_m_IoU, image_fw_IoU, image_deeplab_m_IoU, image_deeplab_fw_IoU), file=output_file)
        # output_file.close()

        if self.epoch == self.config["num_epochs"]:
            self.save()

    def label_refinement_sam(self, prediction, masks_sams):

        prediction = prediction.detach()
        predictions = []
        corrected_areas = []
        for id in range(len(masks_sams)):
            masks_sam = masks_sams[id]
            output_images = prediction[id]
            # posibi = logists[id].unsqueeze(0)
            clip_orig = output_images.clone()

            # # get seeds
            # output_images_fla = output_images.view(-1, 1)
            # posibi_fla = torch.flatten(posibi.permute(0, 2, 3, 1), 0, 2)
            # order = torch.tensor(list(range(output_images_fla.shape[0]))).view(-1, 1)
            # m = torch.cat((order, output_images_fla.detach().cpu()), dim=1)
            # m = tuple(m.T.long())
            # posibi_fla = posibi_fla[m]
            # seeds_index = posibi_fla > self.thr
            # seeds_index = seeds_index.view(output_images.shape).detach().cpu()
            # # tt = 0

            tot_mask = torch.from_numpy(masks_sam[0]['segmentation'] * 0)
            # seeds_index = output_images != 0
            # seeds_index = seeds_index.detach().cpu()
            # if seeds_index.sum() == 0: continue
            # print(seeds_index.sum())
            # print(seeds_index.sum() / seeds_index.view(-1).shape[0])

            masks_sam = sorted(masks_sam, key=(lambda x: x['area']), reverse=False)
            for i, mask in enumerate(masks_sam):

                seg_i = torch.from_numpy(mask['segmentation'])
                # seg_uni = seg_i & seeds_index

                # print(seg_i.shape, seeds_index.shape)
                # if seg_uni.sum() != 0:
                #     tem_mask = clip_orig[seg_uni]
                # else:
                #     continue
                    # tem_mask = clip_orig[seg_i]

                tem_mask = clip_orig[seg_i]
                output_images[seg_i] = torch.mode(tem_mask).values
                tot_mask = tot_mask | seg_i

            # if tot_mask.sum() == 0: continue
            tot_mask = tot_mask.contiguous()
            corrected_area = tot_mask == 1
            corrected_areas.append(corrected_area.unsqueeze(0))
            predictions.append(output_images.contiguous().unsqueeze(0))

        predictions = torch.cat(predictions, dim=0)
        corrected_areas = torch.cat(corrected_areas, dim=0)

        return predictions, corrected_areas


    def noisy_supervision(self, batch, output_points, output_clip, output_deeplab):
        # output_images.shape: torch.Size([96, 64, 224, 416])
        # output_points.shape: torch.Size([225648, 64])

        # pairing_points.shape: torch.Size([214155])
        # pairing_images.shape: torch.Size([214155, 3])
        pairing_points = batch["pairing_points"]
        pairing_images = batch["pairing_images"]
        masks_sam = batch["masks_sams"][0]
        for i in range(1, len(batch["masks_sams"])):
            masks_sam.extend(batch["masks_sams"][i])

        image_logists_clip, image_feats_clip = output_clip
        image_preds_clip = image_logists_clip.argmax(dim=1)
        point_logist, point_feats_local = output_points
        image_logists_deeplab, image_feats_deeplab = output_deeplab

        # print(len(batch["masks_sams"][0]), (len(batch["masks_sams"][1])))
        # print(image_logists_clip.shape, len(masks_sam))

        image_preds_deeplab = image_logists_deeplab.argmax(dim=1)
        image_refinedPreds_deeplab, corrected_areas_deeplab = self.label_refinement_sam(image_preds_deeplab, masks_sam)
        image_logists_deeplab = torch.flatten(image_logists_deeplab.permute(0, 2, 3, 1), start_dim=0, end_dim=2)

        image_refinedPreds_clip, corrected_areas = self.label_refinement_sam(image_preds_clip, masks_sam)
        # print(image_refinedPreds_clip.shape, image_preds_clip.shape, len(masks_sam))
        assert image_refinedPreds_clip.shape == image_preds_clip.shape
        assert len(masks_sam) == image_preds_clip.shape[0]

        image_preds_final = image_refinedPreds_clip

        if self.epoch >= 10:
            rd = random.randint(1, 10)
            if rd > 5: image_preds_final = image_refinedPreds_deeplab

        # global
        if self.config["ignore_index"] == 0:
            point_logist = point_logist[:, 1:]

        k_point_logist = point_logist[pairing_points]
        m = tuple(pairing_images.T.long())
        point_preds_final = image_refinedPreds_clip[m]
        # switchable training strategy
        if self.epoch >= 10:
            rd = random.randint(1, 10)
            if rd >= 3 and rd <= 6: point_preds_final = k_point_logist.argmax(dim=1)
            elif rd > 6: point_preds_final = image_refinedPreds_deeplab[m]
            # if rd <= 5: point_preds_final = k_point_logist.argmax(dim=1)
            # else: point_preds_final = image_preds_deeplab[m]

        loss_points = self.CE(k_point_logist, point_preds_final)
        loss_images = self.CE(image_logists_deeplab, image_preds_final.view(-1))

        k_point_feats_local = point_feats_local[pairing_points]
        image_feats_clip_toPoint = image_feats_clip.permute(0, 2, 3, 1)[m]

        image_shape = image_feats_deeplab.shape[-2:]
        image_feats_clip = F.interpolate(image_feats_clip, size=image_shape, mode='bilinear', align_corners=False)

        loss_local_point = torch.mean(1 - F.cosine_similarity(image_feats_clip_toPoint, k_point_feats_local, dim=1))
        loss_local_image = torch.mean(1 - F.cosine_similarity(image_feats_clip, image_feats_deeplab, dim=1))

        return loss_points + loss_local_point + loss_images + loss_local_image


    def no_supervision(self, batch, output_points, output_clip, output_deeplab):

        logists, feats = output_points

        loss = torch.mean(1 - F.cosine_similarity(feats, feats, dim=1)) * 0

        return loss


    def noisy_supervision_fixed_without_feature_distilation(self, batch, output_points, output_clip, output_deeplab):
        # output_images.shape: torch.Size([96, 64, 224, 416])
        # output_points.shape: torch.Size([225648, 64])

        # pairing_points.shape: torch.Size([214155])
        # pairing_images.shape: torch.Size([214155, 3])
        pairing_points = batch["pairing_points"]
        pairing_images = batch["pairing_images"]
        masks_sam = batch["masks_sams"][0]
        # embeddings_sam = batch["embeddings_sams"]

        # print(type(masks_sam))

        for i in range(1, len(batch["masks_sams"])):
            masks_sam.extend(batch["masks_sams"][i])

        image_logists_clip, image_feats_clip = output_clip
        image_preds_clip = image_logists_clip.argmax(dim=1)

        point_logists, point_feats = output_points
        image_logists_deeplab, image_feats_deeplab = output_deeplab

        image_preds_deeplab = image_logists_deeplab.argmax(dim=1)
        image_refinedPreds_deeplab, corrected_areas_deeplab = self.label_refinement_sam(image_preds_deeplab, masks_sam)
        image_logists_deeplab = torch.flatten(image_logists_deeplab.permute(0, 2, 3, 1), start_dim=0, end_dim=2)

        image_refinedPreds_clip, corrected_areas = self.label_refinement_sam(image_preds_clip, masks_sam)
        # print(image_refinedPreds_clip.shape, image_preds_clip.shape, len(masks_sam))
        assert image_refinedPreds_clip.shape == image_preds_clip.shape
        assert len(masks_sam) == image_preds_clip.shape[0]

        # global
        if self.config["ignore_index"] == 0:
            point_logists = point_logists[:, 1:]

        k_point_logists = point_logists[pairing_points].contiguous()
        m = tuple(pairing_images.T.long())
        point_preds_final = image_refinedPreds_clip[m].contiguous()
        # switchable training strategy
        if self.epoch >= 10:
            rd = random.randint(1, 10)
            if rd >= 3 and rd <= 6: point_preds_final = k_point_logists.argmax(dim=1)
            elif rd > 6: point_preds_final = image_refinedPreds_deeplab[m].contiguous()
            # if rd <= 5: point_preds_final = k_point_logists.argmax(dim=1)
            # else: point_preds_final = image_preds_deeplab[m]


        image_preds_final = image_refinedPreds_clip
        if self.epoch >= 10:
            rd = random.randint(1, 10)
            if rd >= 3 and rd <= 6: image_preds_final = image_refinedPreds_deeplab
            elif rd > 6: image_preds_final[m] = k_point_logists.argmax(dim=1)

        loss_points = self.CE(k_point_logists, point_preds_final)
        loss_images = self.CE(image_logists_deeplab, image_preds_final.view(-1))


        # torch.Size([10, 768, 15, 20]), torch.Size([8985, 96]), torch.Size([10, 2048, 15, 20]), torch.Size([10, 256, 64, 64])
        # print(image_feats_clip.shape, point_feats.shape, image_feats_deeplab.shape, embeddings_sam.shape)

        # mappings = self.feature_mapping(image_feats_clip, image_feats_deeplab, point_feats, embeddings_sam, pairing_points, m)

        loss_feature_distilation = 0
        # loss_feature_distilation += torch.mean(1 - F.cosine_similarity(mappings["point_to_clip"], mappings["clip_to_point"], dim=1))
        # loss_feature_distilation += torch.mean(1 - F.cosine_similarity(mappings["point_to_sam"], mappings["sam_to_point"], dim=1))
        # loss_feature_distilation += torch.mean(1 - F.cosine_similarity(mappings["deeplab_to_clip"], mappings["clip_to_deeplab"], dim=1))
        # loss_feature_distilation += torch.mean(1 - F.cosine_similarity(mappings["deeplab_to_sam"], mappings["sam_to_deeplab"], dim=1))

        # embeddings_sam



        return loss_points + loss_images + loss_feature_distilation


    def noisy_supervision_fixed_with_sam_andclip_features(self, batch, output_points, output_clip, output_deeplab):
        # output_images.shape: torch.Size([96, 64, 224, 416])
        # output_points.shape: torch.Size([225648, 64])

        # pairing_points.shape: torch.Size([214155])
        # pairing_images.shape: torch.Size([214155, 3])
        pairing_points = batch["pairing_points"]
        pairing_images = batch["pairing_images"]
        masks_sam = batch["masks_sams"][0]
        embeddings_sam = batch["embeddings_sams"]

        for i in range(1, len(batch["masks_sams"])):
            masks_sam.extend(batch["masks_sams"][i])

        image_logists_clip, image_feats_clip = output_clip
        image_preds_clip = image_logists_clip.argmax(dim=1)

        point_logists, point_feats = output_points
        image_logists_deeplab, image_feats_deeplab = output_deeplab

        image_preds_deeplab = image_logists_deeplab.argmax(dim=1)
        image_refinedPreds_deeplab, corrected_areas_deeplab = self.label_refinement_sam(image_preds_deeplab, masks_sam)
        image_logists_deeplab = torch.flatten(image_logists_deeplab.permute(0, 2, 3, 1).contiguous(), start_dim=0, end_dim=2)

        image_refinedPreds_clip, corrected_areas = self.label_refinement_sam(image_preds_clip, masks_sam)
        # print(image_refinedPreds_clip.shape, image_preds_clip.shape, len(masks_sam))
        assert image_refinedPreds_clip.shape == image_preds_clip.shape
        assert len(masks_sam) == image_preds_clip.shape[0]

        # global
        if self.config["ignore_index"] == 0:
            point_logists = point_logists[:, 1:]

        k_point_logists = point_logists[pairing_points]
        m = tuple(pairing_images.T.long())
        point_preds_final = image_refinedPreds_clip[m]
        # switchable training strategy
        if self.epoch >= 10:
            rd = random.randint(1, 10)
            if rd >= 3 and rd <= 6: point_preds_final = k_point_logists.argmax(dim=1)
            elif rd > 6: point_preds_final = image_refinedPreds_deeplab[m]
            # if rd <= 5: point_preds_final = k_point_logists.argmax(dim=1)
            # else: point_preds_final = image_preds_deeplab[m]


        image_preds_final = image_refinedPreds_clip
        if self.epoch >= 10:
            rd = random.randint(1, 10)
            if rd >= 3 and rd <= 6: image_preds_final = image_refinedPreds_deeplab
            elif rd > 6: image_preds_final[m] = k_point_logists.argmax(dim=1)

        loss_points = self.CE(k_point_logists, point_preds_final)
        loss_images = self.CE(image_logists_deeplab, image_preds_final.view(-1))


        # torch.Size([10, 768, 15, 20]), torch.Size([8985, 96]), torch.Size([10, 2048, 15, 20]), torch.Size([10, 256, 64, 64])
        # print(image_feats_clip.shape, point_feats.shape, image_feats_deeplab.shape, embeddings_sam.shape)

        mappings = self.feature_mapping(image_feats_clip, image_feats_deeplab.contiguous(), point_feats, embeddings_sam, pairing_points, m)

        loss_feature_distilation = 0
        # loss_feature_distilation += torch.mean(1 - F.cosine_similarity(mappings["point_to_clip"], mappings["clip_to_point"], dim=1))
        loss_feature_distilation += torch.mean(1 - F.cosine_similarity(mappings["point_to_sam"], mappings["sam_to_point"], dim=1))
        # loss_feature_distilation += torch.mean(1 - F.cosine_similarity(mappings["deeplab_to_clip"], mappings["clip_to_deeplab"], dim=1))
        loss_feature_distilation += torch.mean(1 - F.cosine_similarity(mappings["deeplab_to_sam"], mappings["sam_to_deeplab"], dim=1))

        # embeddings_sam



        return loss_points + loss_images + loss_feature_distilation


    def noisy_supervision_fixed_without_clip_features(self, batch, output_points, output_clip, output_deeplab):
        # output_images.shape: torch.Size([96, 64, 224, 416])
        # output_points.shape: torch.Size([225648, 64])

        # pairing_points.shape: torch.Size([214155])
        # pairing_images.shape: torch.Size([214155, 3])
        pairing_points = batch["pairing_points"]
        pairing_images = batch["pairing_images"]
        masks_sam = batch["masks_sams"][0]
        embeddings_sam = batch["embeddings_sams"]

        for i in range(1, len(batch["masks_sams"])):
            masks_sam.extend(batch["masks_sams"][i])

        image_logists_clip, image_feats_clip = output_clip
        image_preds_clip = image_logists_clip.argmax(dim=1)

        point_logists, point_feats = output_points
        image_logists_deeplab, image_feats_deeplab = output_deeplab

        image_preds_deeplab = image_logists_deeplab.argmax(dim=1)
        image_refinedPreds_deeplab, corrected_areas_deeplab = self.label_refinement_sam(image_preds_deeplab, masks_sam)
        image_logists_deeplab = torch.flatten(image_logists_deeplab.permute(0, 2, 3, 1), start_dim=0, end_dim=2)

        image_refinedPreds_clip, corrected_areas = self.label_refinement_sam(image_preds_clip, masks_sam)
        # print(image_refinedPreds_clip.shape, image_preds_clip.shape, len(masks_sam))
        assert image_refinedPreds_clip.shape == image_preds_clip.shape
        assert len(masks_sam) == image_preds_clip.shape[0]

        # global
        if self.config["ignore_index"] == 0:
            point_logists = point_logists[:, 1:]

        k_point_logists = point_logists[pairing_points]
        m = tuple(pairing_images.T.long())
        point_preds_final = image_refinedPreds_clip[m]
        # switchable training strategy
        if self.epoch >= 10:
            rd = random.randint(1, 10)
            if rd >= 3 and rd <= 6: point_preds_final = k_point_logists.argmax(dim=1)
            elif rd > 6: point_preds_final = image_refinedPreds_deeplab[m]
            # if rd <= 5: point_preds_final = k_point_logists.argmax(dim=1)
            # else: point_preds_final = image_preds_deeplab[m]


        image_preds_final = image_refinedPreds_clip
        if self.epoch >= 10:
            rd = random.randint(1, 10)
            if rd >= 3 and rd <= 6: image_preds_final = image_refinedPreds_deeplab
            elif rd > 6: image_preds_final[m] = k_point_logists.argmax(dim=1)

        loss_points = self.CE(k_point_logists, point_preds_final)
        loss_images = self.CE(image_logists_deeplab, image_preds_final.view(-1))


        # torch.Size([10, 768, 15, 20]), torch.Size([8985, 96]), torch.Size([10, 2048, 15, 20]), torch.Size([10, 256, 64, 64])
        # print(image_feats_clip.shape, point_feats.shape, image_feats_deeplab.shape, embeddings_sam.shape)

        mappings = self.feature_mapping(image_feats_clip, image_feats_deeplab, point_feats, embeddings_sam, pairing_points, m)

        loss_feature_distilation = 0
        # loss_feature_distilation += torch.mean(1 - F.cosine_similarity(mappings["point_to_clip"], mappings["clip_to_point"], dim=1))
        loss_feature_distilation += torch.mean(1 - F.cosine_similarity(mappings["point_to_sam"], mappings["sam_to_point"], dim=1))
        # loss_feature_distilation += torch.mean(1 - F.cosine_similarity(mappings["deeplab_to_clip"], mappings["clip_to_deeplab"], dim=1))
        loss_feature_distilation += torch.mean(1 - F.cosine_similarity(mappings["deeplab_to_sam"], mappings["sam_to_deeplab"], dim=1))

        # embeddings_sam



        return loss_points + loss_images + loss_feature_distilation


    def noisy_supervision_fixed_without_SAM_features(self, batch, output_points, output_clip, output_deeplab):
        # output_images.shape: torch.Size([96, 64, 224, 416])
        # output_points.shape: torch.Size([225648, 64])

        # pairing_points.shape: torch.Size([214155])
        # pairing_images.shape: torch.Size([214155, 3])
        pairing_points = batch["pairing_points"]
        pairing_images = batch["pairing_images"]
        masks_sam = batch["masks_sams"][0]
        embeddings_sam = batch["embeddings_sams"]

        for i in range(1, len(batch["masks_sams"])):
            masks_sam.extend(batch["masks_sams"][i])

        image_logists_clip, image_feats_clip = output_clip
        image_preds_clip = image_logists_clip.argmax(dim=1)

        point_logists, point_feats = output_points
        image_logists_deeplab, image_feats_deeplab = output_deeplab

        image_preds_deeplab = image_logists_deeplab.argmax(dim=1)
        image_refinedPreds_deeplab, corrected_areas_deeplab = self.label_refinement_sam(image_preds_deeplab, masks_sam)
        image_logists_deeplab = torch.flatten(image_logists_deeplab.permute(0, 2, 3, 1), start_dim=0, end_dim=2)

        image_refinedPreds_clip, corrected_areas = self.label_refinement_sam(image_preds_clip, masks_sam)
        # print(image_refinedPreds_clip.shape, image_preds_clip.shape, len(masks_sam))
        assert image_refinedPreds_clip.shape == image_preds_clip.shape
        assert len(masks_sam) == image_preds_clip.shape[0]

        # global
        if self.config["ignore_index"] == 0:
            point_logists = point_logists[:, 1:]

        k_point_logists = point_logists[pairing_points]
        m = tuple(pairing_images.T.long())
        point_preds_final = image_refinedPreds_clip[m]
        # switchable training strategy
        if self.epoch >= 10:
            rd = random.randint(1, 10)
            if rd >= 3 and rd <= 6: point_preds_final = k_point_logists.argmax(dim=1)
            elif rd > 6: point_preds_final = image_refinedPreds_deeplab[m]
            # if rd <= 5: point_preds_final = k_point_logists.argmax(dim=1)
            # else: point_preds_final = image_preds_deeplab[m]


        image_preds_final = image_refinedPreds_clip
        if self.epoch >= 10:
            rd = random.randint(1, 10)
            if rd >= 3 and rd <= 6: image_preds_final = image_refinedPreds_deeplab
            elif rd > 6: image_preds_final[m] = k_point_logists.argmax(dim=1)

        loss_points = self.CE(k_point_logists, point_preds_final)
        loss_images = self.CE(image_logists_deeplab, image_preds_final.view(-1))


        # torch.Size([10, 768, 15, 20]), torch.Size([8985, 96]), torch.Size([10, 2048, 15, 20]), torch.Size([10, 256, 64, 64])
        # print(image_feats_clip.shape, point_feats.shape, image_feats_deeplab.shape, embeddings_sam.shape)

        mappings = self.feature_mapping(image_feats_clip, image_feats_deeplab, point_feats, embeddings_sam, pairing_points, m)

        loss_feature_distilation = 0
        loss_feature_distilation += torch.mean(1 - F.cosine_similarity(mappings["point_to_clip"], mappings["clip_to_point"], dim=1))
        # loss_feature_distilation += torch.mean(1 - F.cosine_similarity(mappings["point_to_sam"], mappings["sam_to_point"], dim=1))
        loss_feature_distilation += torch.mean(1 - F.cosine_similarity(mappings["deeplab_to_clip"], mappings["clip_to_deeplab"], dim=1))
        # loss_feature_distilation += torch.mean(1 - F.cosine_similarity(mappings["deeplab_to_sam"], mappings["sam_to_deeplab"], dim=1))

        # embeddings_sam



        return loss_points + loss_images + loss_feature_distilation


    def noisy_supervision_fixed_without_refinement(self, batch, output_points, output_clip, output_deeplab):
        # output_images.shape: torch.Size([96, 64, 224, 416])
        # output_points.shape: torch.Size([225648, 64])

        # pairing_points.shape: torch.Size([214155])
        # pairing_images.shape: torch.Size([214155, 3])
        pairing_points = batch["pairing_points"]
        pairing_images = batch["pairing_images"]
        masks_sam = batch["masks_sams"][0]
        embeddings_sam = batch["embeddings_sams"]

        for i in range(1, len(batch["masks_sams"])):
            masks_sam.extend(batch["masks_sams"][i])

        image_logists_clip, image_feats_clip = output_clip
        image_preds_clip = image_logists_clip.argmax(dim=1)

        point_logists, point_feats = output_points
        image_logists_deeplab, image_feats_deeplab = output_deeplab

        image_preds_deeplab = image_logists_deeplab.argmax(dim=1)
        # image_refinedPreds_deeplab, corrected_areas_deeplab = self.label_refinement_sam(image_preds_deeplab, masks_sam)
        image_refinedPreds_deeplab = image_preds_deeplab
        image_logists_deeplab = torch.flatten(image_logists_deeplab.permute(0, 2, 3, 1), start_dim=0, end_dim=2)

        # image_refinedPreds_clip, corrected_areas = self.label_refinement_sam(image_preds_clip, masks_sam)
        image_refinedPreds_clip = image_preds_clip
        # print(image_refinedPreds_clip.shape, image_preds_clip.shape, len(masks_sam))
        assert image_refinedPreds_clip.shape == image_preds_clip.shape
        assert len(masks_sam) == image_preds_clip.shape[0]

        # global
        if self.config["ignore_index"] == 0:
            point_logists = point_logists[:, 1:]

        k_point_logists = point_logists[pairing_points]
        m = tuple(pairing_images.T.long())
        point_preds_final = image_refinedPreds_clip[m]
        # switchable training strategy
        if self.epoch >= 10:
            rd = random.randint(1, 10)
            if rd >= 3 and rd <= 6: point_preds_final = k_point_logists.argmax(dim=1)
            elif rd > 6: point_preds_final = image_refinedPreds_deeplab[m]
            # if rd <= 5: point_preds_final = k_point_logists.argmax(dim=1)
            # else: point_preds_final = image_preds_deeplab[m]


        image_preds_final = image_refinedPreds_clip
        if self.epoch >= 10:
            rd = random.randint(1, 10)
            if rd >= 3 and rd <= 6: image_preds_final = image_refinedPreds_deeplab
            elif rd > 6: image_preds_final[m] = k_point_logists.argmax(dim=1)

        loss_points = self.CE(k_point_logists, point_preds_final)
        loss_images = self.CE(image_logists_deeplab, image_preds_final.view(-1))


        # torch.Size([10, 768, 15, 20]), torch.Size([8985, 96]), torch.Size([10, 2048, 15, 20]), torch.Size([10, 256, 64, 64])
        # print(image_feats_clip.shape, point_feats.shape, image_feats_deeplab.shape, embeddings_sam.shape)

        mappings = self.feature_mapping(image_feats_clip, image_feats_deeplab, point_feats, embeddings_sam, pairing_points, m)

        loss_feature_distilation = 0
        loss_feature_distilation += torch.mean(1 - F.cosine_similarity(mappings["point_to_clip"], mappings["clip_to_point"], dim=1))
        loss_feature_distilation += torch.mean(1 - F.cosine_similarity(mappings["point_to_sam"], mappings["sam_to_point"], dim=1))
        loss_feature_distilation += torch.mean(1 - F.cosine_similarity(mappings["deeplab_to_clip"], mappings["clip_to_deeplab"], dim=1))
        loss_feature_distilation += torch.mean(1 - F.cosine_similarity(mappings["deeplab_to_sam"], mappings["sam_to_deeplab"], dim=1))

        # embeddings_sam



        return loss_points + loss_images + loss_feature_distilation


    def noisy_supervision_fixed(self, batch, output_points, output_clip, output_deeplab):
        # output_images.shape: torch.Size([96, 64, 224, 416])
        # output_points.shape: torch.Size([225648, 64])

        # pairing_points.shape: torch.Size([214155])
        # pairing_images.shape: torch.Size([214155, 3])
        pairing_points = batch["pairing_points"]
        pairing_images = batch["pairing_images"]
        masks_sam = batch["masks_sams"][0]
        for i in range(1, len(batch["masks_sams"])):
            masks_sam.extend(batch["masks_sams"][i])

        image_logists_clip, image_feats_clip = output_clip
        image_preds_clip = image_logists_clip.argmax(dim=1)
        point_logist, point_feats_local = output_points
        image_logists_deeplab, image_feats_deeplab = output_deeplab

        # print(len(batch["masks_sams"][0]), (len(batch["masks_sams"][1])))
        # print(image_logists_clip.shape, len(masks_sam))

        # print(image_logists_deeplab.shape, image_preds_clip.shape)

        image_preds_deeplab = image_logists_deeplab.argmax(dim=1)
        image_refinedPreds_deeplab, corrected_areas_deeplab = self.label_refinement_sam(image_preds_deeplab, masks_sam)
        image_logists_deeplab = torch.flatten(image_logists_deeplab.permute(0, 2, 3, 1), start_dim=0, end_dim=2)

        image_refinedPreds_clip, corrected_areas = self.label_refinement_sam(image_preds_clip, masks_sam)
        # print(image_refinedPreds_clip.shape, image_preds_clip.shape, len(masks_sam))
        assert image_refinedPreds_clip.shape == image_preds_clip.shape
        assert len(masks_sam) == image_preds_clip.shape[0]

        # global
        if self.config["ignore_index"] == 0:
            point_logist = point_logist[:, 1:]

        k_point_logist = point_logist[pairing_points]
        m = tuple(pairing_images.T.long())
        point_preds_final = image_refinedPreds_clip[m]
        # switchable training strategy
        if self.epoch >= 10:
            rd = random.randint(1, 10)
            if rd >= 3 and rd <= 6: point_preds_final = k_point_logist.argmax(dim=1)
            elif rd > 6: point_preds_final = image_refinedPreds_deeplab[m]
            # if rd <= 5: point_preds_final = k_point_logist.argmax(dim=1)
            # else: point_preds_final = image_preds_deeplab[m]


        image_preds_final = image_refinedPreds_clip

        if self.epoch >= 10:
            rd = random.randint(1, 10)
            if rd >= 3 and rd <= 6: image_preds_final = image_refinedPreds_deeplab
            elif rd > 6: image_preds_final[m] = k_point_logist.argmax(dim=1)

        loss_points = self.CE(k_point_logist, point_preds_final)
        loss_images = self.CE(image_logists_deeplab, image_preds_final.view(-1))

        k_point_feats_local = point_feats_local[pairing_points]
        image_feats_clip_toPoint = image_feats_clip.permute(0, 2, 3, 1)[m]

        image_shape = image_feats_deeplab.shape[-2:]
        image_feats_clip = F.interpolate(image_feats_clip, size=image_shape, mode='bilinear', align_corners=False)

        loss_local_point = torch.mean(1 - F.cosine_similarity(image_feats_clip_toPoint, k_point_feats_local, dim=1))
        loss_local_image = torch.mean(1 - F.cosine_similarity(image_feats_clip, image_feats_deeplab, dim=1))

        return loss_points + loss_local_point + loss_images + loss_local_image



    def noisy_supervision_without_random_switch_without_clipssupervision_after10epoche(self, batch, output_points, output_clip, output_deeplab):
        # output_images.shape: torch.Size([96, 64, 224, 416])
        # output_points.shape: torch.Size([225648, 64])

        # pairing_points.shape: torch.Size([214155])
        # pairing_images.shape: torch.Size([214155, 3])
        pairing_points = batch["pairing_points"]
        pairing_images = batch["pairing_images"]
        masks_sam = batch["masks_sams"][0]
        for i in range(1, len(batch["masks_sams"])):
            masks_sam.extend(batch["masks_sams"][i])

        image_logists_clip, image_feats_clip = output_clip
        image_preds_clip = image_logists_clip.argmax(dim=1)
        point_logist, point_feats_local = output_points
        image_logists_deeplab, image_feats_deeplab = output_deeplab

        # print(len(batch["masks_sams"][0]), (len(batch["masks_sams"][1])))
        # print(image_logists_clip.shape, len(masks_sam))

        image_preds_deeplab = image_logists_deeplab.argmax(dim=1)
        image_refinedPreds_deeplab, corrected_areas_deeplab = self.label_refinement_sam(image_preds_deeplab, masks_sam)
        image_logists_deeplab = image_logists_deeplab.permute(0, 2, 3, 1)

        image_refinedPreds_clip, corrected_areas = self.label_refinement_sam(image_preds_clip, masks_sam)
        # print(image_refinedPreds_clip.shape, image_preds_clip.shape, len(masks_sam))
        assert image_refinedPreds_clip.shape == image_preds_clip.shape
        assert len(masks_sam) == image_preds_clip.shape[0]

        image_preds_final = image_refinedPreds_clip


        # global
        if self.config["ignore_index"] == 0:
            point_logist = point_logist[:, 1:]

        k_point_logist = point_logist[pairing_points]
        m = tuple(pairing_images.T.long())
        if self.epoch >= 10:
            image_preds_final = k_point_logist.argmax(dim=1)
            image_logists_deeplab = image_logists_deeplab[m]
        else:
            image_logists_deeplab = torch.flatten(image_logists_deeplab, start_dim=0, end_dim=2)

        point_preds_final = image_refinedPreds_clip[m]
        # switchable training strategy
        if self.epoch >= 10:
            point_preds_final = image_refinedPreds_deeplab[m]
            # if rd <= 5: point_preds_final = k_point_logist.argmax(dim=1)
            # else: point_preds_final = image_preds_deeplab[m]

        loss_points = self.CE(k_point_logist, point_preds_final)
        loss_images = self.CE(image_logists_deeplab, image_preds_final.view(-1))

        k_point_feats_local = point_feats_local[pairing_points]
        image_feats_clip_toPoint = image_feats_clip.permute(0, 2, 3, 1)[m]

        image_shape = image_feats_deeplab.shape[-2:]
        image_feats_clip = F.interpolate(image_feats_clip, size=image_shape, mode='bilinear', align_corners=False)

        loss_local_point = torch.mean(1 - F.cosine_similarity(image_feats_clip_toPoint, k_point_feats_local, dim=1))
        loss_local_image = torch.mean(1 - F.cosine_similarity(image_feats_clip, image_feats_deeplab, dim=1))

        return loss_points + loss_local_point + loss_images + loss_local_image

    def noisy_supervision_prediction_fusion(self, batch, output_points, output_clip, output_deeplab):
        # output_images.shape: torch.Size([96, 64, 224, 416])
        # output_points.shape: torch.Size([225648, 64])

        # pairing_points.shape: torch.Size([214155])
        # pairing_images.shape: torch.Size([214155, 3])
        pairing_points = batch["pairing_points"]
        pairing_images = batch["pairing_images"]
        masks_sam = batch["masks_sams"][0]
        for i in range(1, len(batch["masks_sams"])):
            masks_sam.extend(batch["masks_sams"][i])

        image_logists_clip, image_feats_clip = output_clip
        image_preds_clip = image_logists_clip.argmax(dim=1)
        point_logist, point_feats_local = output_points
        image_logists_deeplab, image_feats_deeplab = output_deeplab

        image_preds_deeplab = (F.softmax(image_logists_deeplab * 100, dim=1) + F.softmax(image_logists_clip * 100, dim=1)).argmax(dim=1)
        image_refinedPreds_deeplab, corrected_areas_deeplab = self.label_refinement_sam(image_preds_deeplab, masks_sam)
        image_logists_deeplab = torch.flatten(image_logists_deeplab.permute(0, 2, 3, 1), start_dim=0, end_dim=2)

        image_refinedPreds_clip, corrected_areas = self.label_refinement_sam(image_preds_clip, masks_sam)
        assert image_refinedPreds_clip.shape == image_preds_clip.shape
        assert len(masks_sam) == image_preds_clip.shape[0]

        image_preds_final = image_refinedPreds_clip

        if self.epoch >= 10:
            # rd = random.randint(1, 10)
            image_preds_final = image_refinedPreds_deeplab


        # global
        if self.config["ignore_index"] == 0:
            point_logist = point_logist[:, 1:]

        k_point_logist = point_logist[pairing_points]
        m = tuple(pairing_images.T.long())
        point_preds_final = image_refinedPreds_clip[m]
        # switchable training strategy
        if self.epoch >= 10:
            # rd = random.randint(1, 10)
            # if rd >= 3 and rd <= 6: point_preds_final = k_point_logist.argmax(dim=1)
            # elif rd > 6: point_preds_final = image_refinedPreds_deeplab[m]
            # if rd <= 5: point_preds_final = k_point_logist.argmax(dim=1)
            point_preds_final = image_refinedPreds_deeplab[m]

        loss_points = self.CE(k_point_logist, point_preds_final)
        loss_images = self.CE(image_logists_deeplab, image_preds_final.view(-1))

        k_point_feats_local = point_feats_local[pairing_points]
        image_feats_clip_toPoint = image_feats_clip.permute(0, 2, 3, 1)[m]

        image_shape = image_feats_deeplab.shape[-2:]
        image_feats_clip = F.interpolate(image_feats_clip, size=image_shape, mode='bilinear', align_corners=False)

        loss_local_point = torch.mean(1 - F.cosine_similarity(image_feats_clip_toPoint, k_point_feats_local, dim=1))
        loss_local_image = torch.mean(1 - F.cosine_similarity(image_feats_clip, image_feats_deeplab, dim=1))

        return loss_points + loss_local_point + loss_images + loss_local_image

    def noisy_supervision_point_without_clipsupervison_after10_epoches(self, batch, output_points, output_clip, output_deeplab):
        # output_images.shape: torch.Size([96, 64, 224, 416])
        # output_points.shape: torch.Size([225648, 64])

        # pairing_points.shape: torch.Size([214155])
        # pairing_images.shape: torch.Size([214155, 3])
        pairing_points = batch["pairing_points"]
        pairing_images = batch["pairing_images"]
        masks_sam = batch["masks_sams"][0]
        for i in range(1, len(batch["masks_sams"])):
            masks_sam.extend(batch["masks_sams"][i])

        image_logists_clip, image_feats_clip = output_clip
        image_preds_clip = image_logists_clip.argmax(dim=1)
        point_logist, point_feats_local = output_points
        image_logists_deeplab, image_feats_deeplab = output_deeplab

        image_preds_deeplab = image_logists_deeplab.argmax(dim=1)
        image_refinedPreds_deeplab, corrected_areas_deeplab = self.label_refinement_sam(image_preds_deeplab, image_posibi_clip, masks_sam)
        image_logists_deeplab = torch.flatten(image_logists_deeplab.permute(0, 2, 3, 1), start_dim=0, end_dim=2)

        image_refinedPreds_clip, corrected_areas = self.label_refinement_sam(image_preds_clip, image_posibi_clip, masks_sam)
        assert image_refinedPreds_clip.shape == image_preds_clip.shape
        assert len(masks_sam) == image_preds_clip.shape[0]

        image_preds_final = image_refinedPreds_clip

        if self.epoch >= 10:
            rd = random.randint(1, 10)
            if rd > 5: image_preds_final = image_refinedPreds_deeplab

        # global
        if self.config["ignore_index"] == 0:
            point_logist = point_logist[:, 1:]

        k_point_logist = point_logist[pairing_points]
        m = tuple(pairing_images.T.long())
        point_preds_final = image_refinedPreds_clip[m]
        # switchable training strategy
        if self.epoch >= 10:
            rd = random.randint(1, 10)
            # if rd >= 3 and rd <= 6: point_preds_final = k_point_logist.argmax(dim=1)
            # elif rd > 6: point_preds_final = image_refinedPreds_deeplab[m]
            if rd <= 5: point_preds_final = k_point_logist.argmax(dim=1)
            else: point_preds_final = image_refinedPreds_deeplab[m]

        loss_points = self.CE(k_point_logist, point_preds_final)
        loss_images = self.CE(image_logists_deeplab, image_preds_final.view(-1))

        k_point_feats_local = point_feats_local[pairing_points]
        image_feats_clip_toPoint = image_feats_clip.permute(0, 2, 3, 1)[m]

        image_shape = image_feats_deeplab.shape[-2:]
        image_feats_clip = F.interpolate(image_feats_clip, size=image_shape, mode='bilinear', align_corners=False)

        loss_local_point = torch.mean(1 - F.cosine_similarity(image_feats_clip_toPoint, k_point_feats_local, dim=1))
        loss_local_image = torch.mean(1 - F.cosine_similarity(image_feats_clip, image_feats_deeplab, dim=1))

        return loss_points + loss_local_point + loss_images + loss_local_image

    def noisy_supervision_without_clipsupervison_after10_epoches(self, batch, output_points, output_clip, output_deeplab):
        # output_images.shape: torch.Size([96, 64, 224, 416])
        # output_points.shape: torch.Size([225648, 64])

        # pairing_points.shape: torch.Size([214155])
        # pairing_images.shape: torch.Size([214155, 3])
        pairing_points = batch["pairing_points"]
        pairing_images = batch["pairing_images"]
        masks_sam = batch["masks_sams"][0]
        for i in range(1, len(batch["masks_sams"])):
            masks_sam.extend(batch["masks_sams"][i])

        image_logists_clip, image_feats_clip = output_clip
        image_preds_clip = image_logists_clip.argmax(dim=1)
        point_logist, point_feats_local = output_points
        image_logists_deeplab, image_feats_deeplab = output_deeplab

        image_preds_deeplab = image_logists_deeplab.argmax(dim=1)
        image_refinedPreds_deeplab, corrected_areas_deeplab = self.label_refinement_sam(image_preds_deeplab, image_posibi_clip, masks_sam)
        image_logists_deeplab = torch.flatten(image_logists_deeplab.permute(0, 2, 3, 1), start_dim=0, end_dim=2)

        image_refinedPreds_clip, corrected_areas = self.label_refinement_sam(image_preds_clip, image_posibi_clip, masks_sam)
        assert image_refinedPreds_clip.shape == image_preds_clip.shape
        assert len(masks_sam) == image_preds_clip.shape[0]

        image_preds_final = image_refinedPreds_clip

        if self.epoch >= 10:
            image_preds_final = image_refinedPreds_deeplab

        # global
        if self.config["ignore_index"] == 0:
            point_logist = point_logist[:, 1:]

        k_point_logist = point_logist[pairing_points]
        m = tuple(pairing_images.T.long())
        point_preds_final = image_refinedPreds_clip[m]
        # switchable training strategy
        if self.epoch >= 10:
            rd = random.randint(1, 10)
            # if rd >= 3 and rd <= 6: point_preds_final = k_point_logist.argmax(dim=1)
            # elif rd > 6: point_preds_final = image_refinedPreds_deeplab[m]
            if rd <= 5: point_preds_final = k_point_logist.argmax(dim=1)
            else: point_preds_final = image_preds_deeplab[m]

        loss_points = self.CE(k_point_logist, point_preds_final)
        loss_images = self.CE(image_logists_deeplab, image_preds_final.view(-1))

        k_point_feats_local = point_feats_local[pairing_points]
        image_feats_clip_toPoint = image_feats_clip.permute(0, 2, 3, 1)[m]

        image_shape = image_feats_deeplab.shape[-2:]
        image_feats_clip = F.interpolate(image_feats_clip, size=image_shape, mode='bilinear', align_corners=False)

        loss_local_point = torch.mean(1 - F.cosine_similarity(image_feats_clip_toPoint, k_point_feats_local, dim=1))
        loss_local_image = torch.mean(1 - F.cosine_similarity(image_feats_clip, image_feats_deeplab, dim=1))

        return loss_points + loss_local_point + loss_images + loss_local_image

    def noisy_supervision_without_cross_training(self, batch, output_points, output_clip, output_deeplab):
        # output_images.shape: torch.Size([96, 64, 224, 416])
        # output_points.shape: torch.Size([225648, 64])

        # pairing_points.shape: torch.Size([214155])
        # pairing_images.shape: torch.Size([214155, 3])
        pairing_points = batch["pairing_points"]
        pairing_images = batch["pairing_images"]
        masks_sam = batch["masks_sams"][0]
        for i in range(1, len(batch["masks_sams"])):
            masks_sam.extend(batch["masks_sams"][i])

        image_logists_clip, image_feats_clip = output_clip
        image_preds_clip = image_logists_clip.argmax(dim=1)
        point_logist, point_feats_local = output_points
        image_logists_deeplab, image_feats_deeplab = output_deeplab

        image_preds_deeplab = image_logists_deeplab.argmax(dim=1)
        image_refinedPreds_deeplab, corrected_areas_deeplab = self.label_refinement_sam(image_preds_deeplab, masks_sam)
        image_logists_deeplab = torch.flatten(image_logists_deeplab.permute(0, 2, 3, 1), start_dim=0, end_dim=2)

        image_refinedPreds_clip, corrected_areas = self.label_refinement_sam(image_preds_clip, masks_sam)
        assert image_refinedPreds_clip.shape == image_preds_clip.shape
        assert len(masks_sam) == image_preds_clip.shape[0]

        image_preds_final = image_refinedPreds_clip

        if self.epoch >= 10:
            rd = random.randint(1, 10)
            if rd > 5: image_preds_final = image_refinedPreds_deeplab

        # global
        if self.config["ignore_index"] == 0:
            point_logist = point_logist[:, 1:]

        k_point_logist = point_logist[pairing_points]
        m = tuple(pairing_images.T.long())
        point_preds_final = image_refinedPreds_clip[m]
        # switchable training strategy
        if self.epoch >= 10:
            rd = random.randint(1, 10)
            # if rd >= 3 and rd <= 6: point_preds_final = k_point_logist.argmax(dim=1)
            # elif rd > 6: point_preds_final = image_refinedPreds_deeplab[m]
            if rd <= 5: point_preds_final = k_point_logist.argmax(dim=1)
            # else: point_preds_final = image_preds_deeplab[m]

        loss_points = self.CE(k_point_logist, point_preds_final)
        loss_images = self.CE(image_logists_deeplab, image_preds_final.view(-1))

        k_point_feats_local = point_feats_local[pairing_points]
        image_feats_clip_toPoint = image_feats_clip.permute(0, 2, 3, 1)[m]

        image_shape = image_feats_deeplab.shape[-2:]
        image_feats_clip = F.interpolate(image_feats_clip, size=image_shape, mode='bilinear', align_corners=False)

        loss_local_point = torch.mean(1 - F.cosine_similarity(image_feats_clip_toPoint, k_point_feats_local, dim=1))
        loss_local_image = torch.mean(1 - F.cosine_similarity(image_feats_clip, image_feats_deeplab, dim=1))

        return loss_points + loss_local_point + loss_images + loss_local_image

    def noisy_supervision_without_self_andcross_training(self, batch, output_points, output_clip, output_deeplab):
        # output_images.shape: torch.Size([96, 64, 224, 416])
        # output_points.shape: torch.Size([225648, 64])

        # pairing_points.shape: torch.Size([214155])
        # pairing_images.shape: torch.Size([214155, 3])
        pairing_points = batch["pairing_points"]
        pairing_images = batch["pairing_images"]
        masks_sam = batch["masks_sams"][0]
        for i in range(1, len(batch["masks_sams"])):
            masks_sam.extend(batch["masks_sams"][i])

        image_logists_clip, image_feats_clip = output_clip
        image_preds_clip = image_logists_clip.argmax(dim=1)
        point_logist, point_feats_local = output_points
        image_logists_deeplab, image_feats_deeplab = output_deeplab

        image_preds_deeplab = image_logists_deeplab.argmax(dim=1)
        image_refinedPreds_deeplab, corrected_areas_deeplab = self.label_refinement_sam(image_preds_deeplab, masks_sam)
        image_logists_deeplab = torch.flatten(image_logists_deeplab.permute(0, 2, 3, 1), start_dim=0, end_dim=2)

        image_refinedPreds_clip, corrected_areas = self.label_refinement_sam(image_preds_clip, masks_sam)
        assert image_refinedPreds_clip.shape == image_preds_clip.shape
        assert len(masks_sam) == image_preds_clip.shape[0]

        image_preds_final = image_refinedPreds_clip

        # if self.epoch >= 10:
        #     rd = random.randint(1, 10)
        #     if rd > 5: image_preds_final = image_refinedPreds_deeplab

        # global
        if self.config["ignore_index"] == 0:
            point_logist = point_logist[:, 1:]

        k_point_logist = point_logist[pairing_points]
        m = tuple(pairing_images.T.long())
        point_preds_final = image_refinedPreds_clip[m]
        # switchable training strategy
        # if self.epoch >= 10:
        #     rd = random.randint(1, 10)
            # if rd >= 3 and rd <= 6: point_preds_final = k_point_logist.argmax(dim=1)
            # elif rd > 6: point_preds_final = image_refinedPreds_deeplab[m]
            # if rd <= 5: point_preds_final = k_point_logist.argmax(dim=1)
            # else: point_preds_final = image_preds_deeplab[m]

        loss_points = self.CE(k_point_logist, point_preds_final)
        loss_images = self.CE(image_logists_deeplab, image_preds_final.view(-1))

        k_point_feats_local = point_feats_local[pairing_points]
        image_feats_clip_toPoint = image_feats_clip.permute(0, 2, 3, 1)[m]

        image_shape = image_feats_deeplab.shape[-2:]
        image_feats_clip = F.interpolate(image_feats_clip, size=image_shape, mode='bilinear', align_corners=False)

        loss_local_point = torch.mean(1 - F.cosine_similarity(image_feats_clip_toPoint, k_point_feats_local, dim=1))
        loss_local_image = torch.mean(1 - F.cosine_similarity(image_feats_clip, image_feats_deeplab, dim=1))

        return loss_points + loss_local_point + loss_images + loss_local_image



    def noisy_supervision_without_refinement_without_cross_training_without_selfsupervison_without_knowledge_distialltion(self, batch, output_points, output_clip, output_deeplab):
        # output_images.shape: torch.Size([96, 64, 224, 416])
        # output_points.shape: torch.Size([225648, 64])

        # pairing_points.shape: torch.Size([214155])
        # pairing_images.shape: torch.Size([214155, 3])
        pairing_points = batch["pairing_points"]
        pairing_images = batch["pairing_images"]
        masks_sam = batch["masks_sams"][0]
        for i in range(1, len(batch["masks_sams"])):
            masks_sam.extend(batch["masks_sams"][i])

        image_logists_clip, image_feats_clip = output_clip
        image_preds_clip = image_logists_clip.argmax(dim=1)
        point_logist, point_feats_local = output_points
        image_logists_deeplab, image_feats_deeplab = output_deeplab

        image_preds_deeplab = image_logists_deeplab.argmax(dim=1)
        image_refinedPreds_deeplab = image_preds_deeplab
        # image_refinedPreds_deeplab, corrected_areas_deeplab = self.label_refinement_sam(image_preds_deeplab, image_posibi_clip, masks_sam)
        image_logists_deeplab = torch.flatten(image_logists_deeplab.permute(0, 2, 3, 1), start_dim=0, end_dim=2)

        # image_refinedPreds_clip, corrected_areas = self.label_refinement_sam(image_preds_clip, image_posibi_clip, masks_sam)
        image_refinedPreds_clip = image_preds_clip
        assert image_refinedPreds_clip.shape == image_preds_clip.shape
        assert len(masks_sam) == image_preds_clip.shape[0]

        image_preds_final = image_refinedPreds_clip

        # if self.epoch >= 10:
        #     rd = random.randint(1, 10)
        #     if rd > 5: image_preds_final = image_refinedPreds_deeplab

        # global
        if self.config["ignore_index"] == 0:
            point_logist = point_logist[:, 1:]

        k_point_logist = point_logist[pairing_points]
        m = tuple(pairing_images.T.long())
        point_preds_final = image_refinedPreds_clip[m]
        # switchable training strategy
        # if self.epoch >= 10:
        #     rd = random.randint(1, 10)
        #     if rd >= 3 and rd <= 6: point_preds_final = k_point_logist.argmax(dim=1)
        #     elif rd > 6: point_preds_final = image_refinedPreds_deeplab[m]
            # if rd <= 5: point_preds_final = k_point_logist.argmax(dim=1)
            # else: point_preds_final = image_preds_deeplab[m]

        loss_points = self.CE(k_point_logist, point_preds_final)
        loss_images = self.CE(image_logists_deeplab, image_preds_final.view(-1))

        # k_point_feats_local = point_feats_local[pairing_points]
        # image_feats_clip_toPoint = image_feats_clip.permute(0, 2, 3, 1)[m]

        # image_shape = image_feats_deeplab.shape[-2:]
        # image_feats_clip = F.interpolate(image_feats_clip, size=image_shape, mode='bilinear', align_corners=False)

        loss_local_point, loss_local_image = 0, 0
        # loss_local_point = torch.mean(1 - F.cosine_similarity(image_feats_clip_toPoint, k_point_feats_local, dim=1))
        # loss_local_image = torch.mean(1 - F.cosine_similarity(image_feats_clip, image_feats_deeplab, dim=1))

        return loss_points + loss_local_point + loss_images + loss_local_image


    def noisy_supervision_without_feature_distillation(self, batch, output_points, output_clip, output_deeplab):
        # output_images.shape: torch.Size([96, 64, 224, 416])
        # output_points.shape: torch.Size([225648, 64])

        # pairing_points.shape: torch.Size([214155])
        # pairing_images.shape: torch.Size([214155, 3])
        pairing_points = batch["pairing_points"]
        pairing_images = batch["pairing_images"]
        masks_sam = batch["masks_sams"][0]
        for i in range(1, len(batch["masks_sams"])):
            masks_sam.extend(batch["masks_sams"][i])

        image_logists_clip, image_feats_clip = output_clip
        image_preds_clip = image_logists_clip.argmax(dim=1)
        point_logist, point_feats_local = output_points
        image_logists_deeplab, image_feats_deeplab = output_deeplab

        image_preds_deeplab = image_logists_deeplab.argmax(dim=1)
        image_refinedPreds_deeplab, corrected_areas_deeplab = self.label_refinement_sam(image_preds_deeplab, image_posibi_clip, masks_sam)
        image_logists_deeplab = torch.flatten(image_logists_deeplab.permute(0, 2, 3, 1), start_dim=0, end_dim=2)

        image_refinedPreds_clip, corrected_areas = self.label_refinement_sam(image_preds_clip, image_posibi_clip, masks_sam)
        assert image_refinedPreds_clip.shape == image_preds_clip.shape
        assert len(masks_sam) == image_preds_clip.shape[0]

        image_preds_final = image_refinedPreds_clip

        if self.epoch >= 10:
            rd = random.randint(1, 10)
            if rd > 5: image_preds_final = image_refinedPreds_deeplab

        # global
        if self.config["ignore_index"] == 0:
            point_logist = point_logist[:, 1:]

        k_point_logist = point_logist[pairing_points]
        m = tuple(pairing_images.T.long())
        point_preds_final = image_refinedPreds_clip[m]
        # switchable training strategy
        if self.epoch >= 10:
            rd = random.randint(1, 10)
            if rd >= 3 and rd <= 6: point_preds_final = k_point_logist.argmax(dim=1)
            elif rd > 6: point_preds_final = image_refinedPreds_deeplab[m]
            # if rd <= 5: point_preds_final = k_point_logist.argmax(dim=1)
            # else: point_preds_final = image_preds_deeplab[m]

        loss_points = self.CE(k_point_logist, point_preds_final)
        loss_images = self.CE(image_logists_deeplab, image_preds_final.view(-1))

        # k_point_feats_local = point_feats_local[pairing_points]
        # image_feats_clip_toPoint = image_feats_clip.permute(0, 2, 3, 1)[m]

        # image_shape = image_feats_deeplab.shape[-2:]
        # image_feats_clip = F.interpolate(image_feats_clip, size=image_shape, mode='bilinear', align_corners=False)

        loss_local_point, loss_local_image = 0, 0
        # loss_local_point = torch.mean(1 - F.cosine_similarity(image_feats_clip_toPoint, k_point_feats_local, dim=1))
        # loss_local_image = torch.mean(1 - F.cosine_similarity(image_feats_clip, image_feats_deeplab, dim=1))

        return loss_points + loss_local_point + loss_images + loss_local_image


    def training_epoch_end(self, outputs):
        self.epoch += 1
        self.save()
        if self.epoch == self.num_epochs:
            self.save()
        return super().training_epoch_end(outputs)

    @rank_zero_only
    def save(self):
        path = os.path.join(self.working_dir, f"model_epoch_{self.epoch}.pt")
        torch.save(
            {
                "model_points": self.model_points.state_dict(),
                "model_clip": self.model_clip.state_dict(),
                "model_images": self.model_images.state_dict(),
                "epoch": self.epoch,
                "config": self.config,
            },
            path,
        )
