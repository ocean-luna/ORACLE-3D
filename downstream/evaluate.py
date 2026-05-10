import numpy as np
import torch
from tqdm import tqdm
from copy import deepcopy
from MinkowskiEngine import SparseTensor
import MinkowskiEngine as ME
# from torchsparse import SparseTensor
from utils.metrics import compute_IoU
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from utils.scannet_utils import create_color_palette as create_color_palette

CLASSES_NUSCENES = [
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
    "driveable_surface",
    "other_flat",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
]

CLASSES_KITTI = [
    "car",
    "bicycle",
    "motorcycle",
    "truck",
    "other-vehicle",
    "person",
    "bicyclist",
    "motorcyclist",
    "road",
    "parking",
    "sidewalk",
    "other-ground",
    "building",
    "fence",
    "vegetation",
    "trunk",
    "terrain",
    "pole",
    "traffic-sign",
]

CLASSES_scannet = [
    'wall',
    'floor',
    'cabinet',
    'bed',
    'chair',
    'sofa',
    'table',
    'door',
    'window',
    'bookshelf',
    'picture',
    'counter',
    'desk',
    'curtain',
    'refrigerator',
    'shower curtain',
    'toilet',
    'sink',
    'bathtub',
    'other furniture'
]
#
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

        seg_i = mask_labels[i]['segmentation']
        img_vis_label[seg_i] = color_template[i]

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

    img_vis_label = np.clip(img_vis_label, 0, 255)
    img_vis_label = img_vis_label / 255.0
    img_vis_label = img_vis_label * 0.35 + img * 0.65
    plt.imsave("visual/%s_label.png" % (image_name), img_vis_label)

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


# def show_anns(image, anns, lidar_names):
#     if len(anns) == 0:
#         return
#     sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
#     ax = plt.gca()
#     ax.set_autoscale_on(False)
#     polygons = []
#     color = []
#     # img = np.ones((image.shape[0], image.shape[1], 3))
#     image = image.cpu().contiguous().numpy()
#     plt.imsave("visual/%s.png" % lidar_names, image)
#
#     print("image", image.shape)
#     img_vis = np.zeros_like(image)
#     for id, ann in enumerate(sorted_anns):
#         m = ann['segmentation']
#         print("m: ", m.shape)
#         img = np.ones((m.shape[0], m.shape[1], 3))
#         color_mask = np.random.random((1, 3)).tolist()[0]
#         for i in range(3):
#             img[:,:,i] = color_mask[i]
#
#         img = np.dstack((img, m * 0.35))
#         print("img_after: ", img.shape)
#         plt.imsave("visual/%s_masked.png" % (lidar_names), img_vis)
#
#         img_vis += img
#
#     img_vis = image * 0.65 + img_vis*0.35
#     plt.imsave("visual/%s_masked.png" % (lidar_names), img_vis)

    # img.save(lidar_names + )
    # ax.imshow(np.dstack((img, m*0.35)))

def visual_masks(image, masks_sam, output_images, clip_orig, deeplab_pred, img_labels, image_name, seeds_index):

    mask_clip = []
    mask_labels = []
    mask_cliporig = []
    mask_deeplab = []

    for i in range(21):
        index = output_images == i
        mask_clip.append({'segmentation': index.detach().cpu().numpy()})

        index = img_labels == i
        mask_labels.append({'segmentation': index.detach().cpu().numpy()})

        index = clip_orig == i
        mask_cliporig.append({'segmentation': index.detach().cpu().numpy()})

        index = deeplab_pred == i
        mask_deeplab.append({'segmentation': index.detach().cpu().numpy()})

    show_anns(image, masks_sam, mask_clip, mask_cliporig, mask_deeplab, mask_labels, image_name, seeds_index)

def extract_sedd_points():
    print((posibi > thr).sum(),
          (posibi > thr).sum() / (posibi.shape[0] * posibi.shape[1] * posibi.shape[2] * posibi.shape[3]))
    output_images_fla = output_images.view(-1, 1)
    posibi_fla = torch.flatten(posibi.permute(0, 2, 3, 1), 0, 2)
    order = torch.tensor(list(range(output_images_fla.shape[0]))).view(-1, 1)
    m = torch.cat((order, output_images_fla.detach().cpu()), dim=1)
    m = tuple(m.T.long())
    posibi_fla = posibi_fla[m]
    posibi_fla_index = posibi_fla > thr


def sam_preprocess():
    for id in range(len(frame_names)):
        torch.cuda.empty_cache()

        image_name = lidar_names + "_" + frame_names[id]
        image = input_img[id].permute(1, 2, 0)
        image = (image * 255).cpu().numpy()

        print(image_name)
        save_name = "sam_preprocess/" + image_name + ".npy"

        token = save_name + "token"
        if os.path.exists(token) or os.path.exists(save_name):
            print("exsists")
            continue

        try:
            os.mknod(token)
        except:
            continue

        masks_sam = SAM.generate(image.astype("uint8"))
        np.save(save_name, masks_sam)

        try:
            os.remove(token)
        except:
            continue

        print(type(masks_sam))
        print(len(masks_sam))

        continue

        for id in range(len(frame_names)):
            torch.cuda.empty_cache()

            image_name = lidar_names + "_" + frame_names[id]
            image = input_img[id].permute(1, 2, 0)
            image = (image * 255).cpu().numpy()

            print(image_name)
            save_name = "sam_preprocess/" + image_name + "_samImage_embedding.npy"

            token = save_name + "token"
            if os.path.exists(token) or os.path.exists(save_name):
                print("exsists")
                continue

            try:
                os.mknod(token)
            except:
                continue

            SAM.set_image(image.astype("uint8"))
            embedding_sam = SAM.get_image_embedding().detach().cpu().numpy()
            np.save(save_name, embedding_sam)

            try:
                os.remove(token)
            except:
                continue

            print(type(embedding_sam))
            print(embedding_sam.shape)

            continue


        # image_name = lidar_names + "_" + frame_names[id]
        # image = input_img[id].permute(1, 2, 0)
        # image = (image * 255).cpu().numpy()

        # print(image_name)
        # save_name = "sam_preprocess/" + image_name + "_samImage_embedding.npy"
        #
        # token = save_name + "token"
        # if os.path.exists(token) or os.path.exists(save_name):
        #     print("exsists")
        #     continue
        #
        # try:
        #     os.mknod(token)
        # except:
        #     continue
        #
        # SAM.set_image(image.astype("uint8"))
        # embedding_sam = SAM.get_image_embedding().detach().cpu().numpy()
        # np.save(save_name, embedding_sam)
        #
        # try:
        #     os.remove(token)
        # except:
        #     continue
        #
        # #
        # # try:
        # #     embeddings_sam = np.load(save_name, allow_pickle=True)
        # # except:
        # #
        # #     print("re_generation")
        # #     SAM.set_image(image.astype("uint8"))
        # #     embedding_sam = SAM.get_image_embedding().detach().cpu().numpy()
        # #     np.save(save_name, embedding_sam)
        #
        # print(type(embedding_sam))
        # print(embedding_sam.shape)
        #
        # continue


#
#
def evaluate_images_and_point(model, SAM, dataloader, config):
    """
    Function to evaluate the performances of a downstream training.
    It prints the per-class IoU, mIoU and fwIoU.
    """
    thr = 0.3

    model.eval()
    with torch.no_grad():
        i = 0
        full_predictions = []
        ground_truth = []
        t = 0

        ratios = 0
        cot = 0
        for batch in tqdm(dataloader):

            t += 1
            # if t >= 5: break

            lidar_names = batch["lidar_name"][0]
            frame_names = batch["frame_names"][0]
            input_img = batch["input_I"].to(device=0).float()

            img_labels_b = batch["imgs_labels"]
            image_clip, _ = model(input_img)

            posibi_b = F.softmax(output_images_b * 100, dim=1)
            output_images_b = output_images_b.argmax(dim=1)





            sparse_input = SparseTensor(batch["sinput_F"].float(), batch["sinput_C"].int(), device=0)
            # print(sparse_input, model)
            output_points = model(sparse_input)

            # for spvcnn
            # sparse_input = SparseTensor(batch["sinput_F"], batch["sinput_C"])
            # output_points = model(sparse_input.to(0))
            # print("output_points.shape", output_points.shape)
            if config["ignore_index"]:
                output_points[:, config["ignore_index"]] = -1e6


            preds = output_points.argmax(1).cpu()
            offset = 0



            for id in range(len(frame_names)):
                torch.cuda.empty_cache()

                image_name = lidar_names + "_" + frame_names[id]
                masks_sam = np.load("sam_preprocess/" + image_name + ".npy", allow_pickle=True)

                output_images = output_images_b[id]
                img_labels = img_labels_b[id]
                posibi = posibi_b[id].unsqueeze(0)
                image = input_img[id].permute(1, 2, 0)

                clip_orig = output_images.clone()

                # get seeds
                print((posibi > thr).sum(), (posibi > thr).sum() / (posibi.shape[0] * posibi.shape[1] * posibi.shape[2] * posibi.shape[3]))
                output_images_fla = output_images.view(-1, 1)
                posibi_fla = torch.flatten(posibi.permute(0, 2, 3, 1), 0, 2)
                order = torch.tensor(list(range(output_images_fla.shape[0]))).view(-1, 1).to(device=0)
                m = torch.cat((order, output_images_fla), dim=1)
                m = tuple(m.T.long())
                posibi_fla = posibi_fla[m]
                seeds_index = posibi_fla > thr

                # print(len(masks_sam))
                # print(masks_sam[0])
                tot_mask = masks_sam[0]['segmentation']
                seeds_index = seeds_index.view(output_images.shape)
                print(seeds_index.shape)
                # tt = 0

                # print(tot_mask.shape, clip_orig.shape)

                masks_sam = sorted(masks_sam, key=(lambda x: x['area']), reverse=False)
                for i, mask in enumerate(masks_sam):

                    seg_i = mask['segmentation']
                    # tt += seg_i.sum()
                    # seg_uni = seg_i & seeds_index

                    # print(seeds_index.sum(), seg_i.sum(), seg_uni.sum())

                    # print(seg_i.shape, seeds_index.shape)
                    # if seg_uni.sum() != 0:
                    #     tem_mask = clip_orig[seg_i]
                    # else:
                    #     continue
                        # tem_mask = clip_orig[seg_i]

                    # tem_mask = clip_orig[seg_i]
                    # print(tem_mask, torch.mode(tem_mask).values)
                    # print("tem_mask: ", tem_mask.shape)
                    # print("before output_images[seg_i] ", output_images[seg_i])
                    # output_images[seg_i] = torch.mode(tem_mask).values
                    # print("output_images[seg_i] ", output_images[seg_i].shape)
                    tot_mask = tot_mask | seg_i
                    # print(tot_mask.sum())


                if tot_mask.sum() == 0: continue
                tot_mask = torch.from_numpy(tot_mask).contiguous()
                corrected_area = tot_mask == 1

                # tot_mask[tot_mask == 0] = False
                # tot_mask[tot_mask == 1] = True

                # print("output_images: ", (output_images - kk_image).sum())
                clip_orig += 1
                output_images += 1

                # output_images[1 - tot_mask] = 0
                # img_labels[1 - tot_mask] = 0

                output_images = output_images.contiguous()

                ratio = tot_mask.sum() / tot_mask.view(-1).shape[0]
                ratios += ratio
                cot += 1

                print("", ratio)
                # if ratio > 0.8:
                #     visual_masks(image, masks_sam, output_images, clip_orig, img_labels, image_name, seeds_index)
                # else:
                #     continue
                # if ratio < 0.9: continue

                # output_images = output_images[seeds_index]

                output_images = output_images[seeds_index]
                img_labels = img_labels[seeds_index]

                # output_images = output_images[corrected_area]
                # img_labels = img_labels[corrected_area]


                full_predictions.append(output_images.cpu().view(-1))
                ground_truth.append(deepcopy(img_labels.view(-1)))

                torch.cuda.empty_cache()
                # del tot_mask
                # del seeds_index
                # del masks_sam

        print("avg ration: ", ratios / cot)

        full_predictions = torch.cat(full_predictions).int()
        ground_truth = torch.cat(ground_truth).int()


        print(ground_truth)

        m_IoU, fw_IoU, per_class_IoU = compute_IoU(
            full_predictions,
            ground_truth,
            config["model_n_out"],
            ignore_index=0,
        )

        # import pdb
        # pdb.set_trace()

        print("Per class IoU:")
        if config["dataset"].lower() == "nuscenes":
            print(
                *[
                    f"{a:20} - {b:.3f}"
                    for a, b in zip(CLASSES_NUSCENES, (per_class_IoU).numpy())
                ],
                sep="\n",
            )
        elif config["dataset"].lower() == "kitti":
            print(
                *[
                    f"{a:20} - {b:.3f}"
                    for a, b in zip(CLASSES_KITTI, (per_class_IoU).numpy())
                ],
                sep="\n",
            )
        elif config["dataset"].lower() == "scannet":
            print(
                *[
                    f"{a:20} - {b:.3f}"
                    for a, b in zip(CLASSES_scannet, (per_class_IoU).numpy())
                ],
                sep="\n",
            )
        print()
        print(f"mIoU: {m_IoU}")
        print(f"fwIoU: {fw_IoU}")

    return m_IoU

def label_refinement_sam( prediction, masks_sams):

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
def evaluate_visual_points(model, dataloader, config):
    """
    Function to evaluate the performances of a downstream training.
    It prints the per-class IoU, mIoU and fwIoU.
    """
    model.eval()
    with torch.no_grad():
        i = 0
        full_predictions = []
        ground_truth = []
        for batch in tqdm(dataloader):
            lidar_names = batch["lidar_name"]
            sinput_C = batch["sinput_C"]
            sinput_F = batch["sinput_F"]
            # masks_sams = batch["masks_sams"]

            sparse_input = ME.SparseTensor(sinput_F.float(), coordinates=sinput_C.int(),device="cuda:0")

            # print(sparse_input, model)
            output_points = model(sparse_input)

            # for spvcnn
            # sparse_input = SparseTensor(batch["sinput_F"], batch["sinput_C"])
            # output_points = model(sparse_input.to(0))
            # print("output_points.shape", output_points.shape)
            if config["ignore_index"]:
                output_points[:, config["ignore_index"]] = -1e6


            preds = output_points.argmax(1).cpu()
            offset = 0

            # print(output_points)
            # print(batch["evaluation_labels"][0].max())
            # print(batch["evaluation_labels"][0].min())


            for j, lb in enumerate(batch["len_batch"]):
                # print(batch["len_batch"], j)
                inverse_indexes = batch["inverse_indexes"][j]
                predictions = preds[inverse_indexes + offset]

                # print(predictions.shape, batch["evaluation_labels"][j].shape)
                # remove the ignored index entirely
                full_predictions.append(predictions)
                ground_truth.append(deepcopy(batch["evaluation_labels"][j]))
                offset += lb

                # m_IoU, fw_IoU, per_class_IoU = compute_IoU(
                #     torch.cat([predictions]),
                #     torch.cat([deepcopy(batch["evaluation_labels"][j])]),
                #     config["model_n_out"],
                #     ignore_index=0,
                # )

                '''
                class_ind = 4
                lidar_name = lidar_names[j].split('/')[-1]
                root_path = '/mnt/lustre/chenrunnan/projects/SLidR/visual/annotation_free/'
                # lidar_name_path = root_path + str(per_class_IoU[class_ind]) + lidar_name
                lidar_name_path = root_path + lidar_name
                save_file = predictions.unsqueeze(-1).numpy()
                # save_file = np.expand_dims(predictions)
                # if per_class_IoU[class_ind] != 1 and per_class_IoU[class_ind] > 0.4:
                np.array(save_file).astype(np.uint8).tofile(lidar_name_path)
                '''

                # import pdb
                # pdb.set_trace()

            i += j

            torch.cuda.empty_cache()


        full_predictions = torch.cat(full_predictions).int()
        ground_truth = torch.cat(ground_truth).int()

        # if config["dataset"].lower() == "scannet":
        #     ground_truth += 1
        #     ground_truth[ground_truth == -99] = 0

        # print(full_predictions.shape, torch.cat(ground_truth).shape)
        # print(torch.cat(full_predictions), torch.cat(ground_truth))

        # print(ground_truth)

        m_IoU, fw_IoU, per_class_IoU = compute_IoU(
            full_predictions,
            ground_truth,
            config["model_n_out"],
            ignore_index=0,
        )

        # import pdb
        # pdb.set_trace()

        print("Per class IoU:")
        if config["dataset"].lower() == "nuscenes":
            print(
                *[
                    f"{a:20} - {b:.3f}"
                    for a, b in zip(CLASSES_NUSCENES, (per_class_IoU).numpy())
                ],
                sep="\n",
            )
        elif config["dataset"].lower() == "kitti":
            print(
                *[
                    f"{a:20} - {b:.3f}"
                    for a, b in zip(CLASSES_KITTI, (per_class_IoU).numpy())
                ],
                sep="\n",
            )
        elif config["dataset"].lower() == "scannet":
            print(
                *[
                    f"{a:20} - {b:.3f}"
                    for a, b in zip(CLASSES_scannet, (per_class_IoU).numpy())
                ],
                sep="\n",
            )
        print()
        print(f"mIoU: {m_IoU}")
        print(f"fwIoU: {fw_IoU}")

    return m_IoU


def validation_step(model_points, model_clip,model_images,dataloader, config):
    model_points.eval()
    model_images.eval()
    for batch in tqdm(dataloader):
        sparse_input = ME.SparseTensor(batch["sinput_F"].float(), coordinates=batch["sinput_C"].int(),device="cuda:0")
        output_points = model_points(sparse_input)
        output_clip = model_clip(batch["input_I"].to(device=0).float())
        output_deeplab = model_images(batch["input_I"].to(device=0).float())

        image_logists_deeplab, image_feature_deeplab = output_deeplab
        image_preds_deeplab = image_logists_deeplab.argmax(dim=1).view(-1).int() + 1

        image_logists_clip, image_feats_clip = output_clip
        # image_preds_clip = (F.softmax(image_logists_deeplab * 100, dim=1) + F.softmax(image_logists_clip * 100, dim=1)).argmax(dim=1)
        image_preds_clip = image_logists_clip.argmax(dim=1).int() + 1

        point_logists=output_points

        masks_sams = batch["masks_sams"][0]
        for i in range(1, len(batch["masks_sams"])):
            masks_sams.extend(batch["masks_sams"][i])

        # image_refinedpreds_clip = image_preds_clip
        image_refinedpreds_clip, corrected_areas_clip = label_refinement_sam(image_preds_clip, masks_sams)
        assert image_refinedpreds_clip.shape == image_preds_clip.shape
        # image_preds_clip = image_refinedpreds_clip

        # Ensure we ignore the index 0
        # (probably not necessary after some training)

        point_preds = []
        point_labels = []
        offset = 0

        if config["ignore_index"]:
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



        if config['dataset'] == "nuscenes":
            image_labels = torch.zeros(image_preds_clip.shape).int().to(device="cuda")
        elif config['dataset'] == "scannet":
            image_labels = batch["imgs_labels"].view(-1).int()
        assert image_preds_clip.shape == image_labels.shape

        image_preds_clip = image_preds_clip.view(-1).int()


        images_batch = batch["input_I"]
        image_names = batch["image_names"]
        deeplab_preds = image_logists_deeplab.argmax(dim=1).int() + 1
        clip_preds = image_logists_clip.argmax(dim=1).int() + 1
        clipSAM_preds = image_refinedpreds_clip
        # import pdb
        # pdb.set_trace()

        for id in range(batch["input_I"].shape[0]):
            image = images_batch[id].permute(1, 2, 0)
            image_label=image_labels[id]
            masks_sam = masks_sams[id]
            image_name = image_names[id]
            image_name = image_name.split('/')[-1]
            deeplab_pred = deeplab_preds[id]
            clip_pred = clip_preds[id]
            clipSAM_pred = clipSAM_preds[id]
            seeds_index = 0

            visual_masks(image, masks_sam, clipSAM_pred, clip_pred, deeplab_pred, image_label, image_name,
                         seeds_index)



    return 0

    # point_preds = prediction_pres.to(device="cuda")

    # return point_preds, point_labels, image_preds_clip, image_preds_deeplab, image_labels



def evaluate_points(model, dataloader, config):
    """
    Function to evaluate the performances of a downstream training.
    It prints the per-class IoU, mIoU and fwIoU.
    """
    model.eval()
    with torch.no_grad():
        i = 0
        full_predictions = []
        ground_truth = []
        for batch in tqdm(dataloader):
            lidar_names = batch["lidar_name"]

            sparse_input = SparseTensor(batch["sinput_F"].float(), batch["sinput_C"].int(), device=0)
            # print(sparse_input, model)
            output_points = model(sparse_input)

            # for spvcnn
            # sparse_input = SparseTensor(batch["sinput_F"], batch["sinput_C"])
            # output_points = model(sparse_input.to(0))
            # print("output_points.shape", output_points.shape)
            if config["ignore_index"]:
                output_points[:, config["ignore_index"]] = -1e6


            preds = output_points.argmax(1).cpu()
            offset = 0

            # print(output_points)
            # print(batch["evaluation_labels"][0].max())
            # print(batch["evaluation_labels"][0].min())


            for j, lb in enumerate(batch["len_batch"]):
                # print(batch["len_batch"], j)
                inverse_indexes = batch["inverse_indexes"][j]
                predictions = preds[inverse_indexes + offset]

                # print(predictions.shape, batch["evaluation_labels"][j].shape)
                # remove the ignored index entirely
                full_predictions.append(predictions)
                ground_truth.append(deepcopy(batch["evaluation_labels"][j]))
                offset += lb

                # m_IoU, fw_IoU, per_class_IoU = compute_IoU(
                #     torch.cat([predictions]),
                #     torch.cat([deepcopy(batch["evaluation_labels"][j])]),
                #     config["model_n_out"],
                #     ignore_index=0,
                # )

                '''
                class_ind = 4
                lidar_name = lidar_names[j].split('/')[-1]
                root_path = '/mnt/lustre/chenrunnan/projects/SLidR/visual/annotation_free/'
                # lidar_name_path = root_path + str(per_class_IoU[class_ind]) + lidar_name
                lidar_name_path = root_path + lidar_name
                save_file = predictions.unsqueeze(-1).numpy()
                # save_file = np.expand_dims(predictions)
                # if per_class_IoU[class_ind] != 1 and per_class_IoU[class_ind] > 0.4:
                np.array(save_file).astype(np.uint8).tofile(lidar_name_path)
                '''

                # import pdb
                # pdb.set_trace()

            i += j

            torch.cuda.empty_cache()


        full_predictions = torch.cat(full_predictions).int()
        ground_truth = torch.cat(ground_truth).int()

        # if config["dataset"].lower() == "scannet":
        #     ground_truth += 1
        #     ground_truth[ground_truth == -99] = 0

        # print(full_predictions.shape, torch.cat(ground_truth).shape)
        # print(torch.cat(full_predictions), torch.cat(ground_truth))

        print(ground_truth)

        m_IoU, fw_IoU, per_class_IoU = compute_IoU(
            full_predictions,
            ground_truth,
            config["model_n_out"],
            ignore_index=0,
        )

        # import pdb
        # pdb.set_trace()

        print("Per class IoU:")
        if config["dataset"].lower() == "nuscenes":
            print(
                *[
                    f"{a:20} - {b:.3f}"
                    for a, b in zip(CLASSES_NUSCENES, (per_class_IoU).numpy())
                ],
                sep="\n",
            )
        elif config["dataset"].lower() == "kitti":
            print(
                *[
                    f"{a:20} - {b:.3f}"
                    for a, b in zip(CLASSES_KITTI, (per_class_IoU).numpy())
                ],
                sep="\n",
            )
        elif config["dataset"].lower() == "scannet":
            print(
                *[
                    f"{a:20} - {b:.3f}"
                    for a, b in zip(CLASSES_scannet, (per_class_IoU).numpy())
                ],
                sep="\n",
            )
        print()
        print(f"mIoU: {m_IoU}")
        print(f"fwIoU: {fw_IoU}")

    return m_IoU


def evaluate_images(model_images, model_clip, SAM, dataloader, config):
    """
    Function to evaluate the performances of a downstream training.
    It prints the per-class IoU, mIoU and fwIoU.
    """
    thr = 0.3

    model_clip.eval()
    model_images.eval()
    with torch.no_grad():
        i = 0
        full_predictions = []
        ground_truth = []
        t = 0

        ratios = 0
        cot = 0

        # list_name = ["scene0204_01", "scene0269_01", "scene0219_00", "scene0701_01", "scene0182_01"]

        for batch in tqdm(dataloader):

            t += 1
            # if t >= 5: break

            lidar_names = batch["lidar_name"][0]
            frame_names = batch["image_names"][0]
            input_img = batch["input_I"].to(device=0).float()

            # if lidar_names not in list_name:
            #     print(lidar_names)
            #     continue


            # image = input_img[0].permute(1, 2, 0)
            # img = (image * 255).cpu().numpy()
            # masks_sam = SAM.generate(img.astype("uint8"))

            # image_name = lidar_names + "_" + frame_names[0]
            # masks_sam = np.load("sam_preprocess/" + image_name + ".npy", allow_pickle=True)
            # print(len(masks_sam))

            img_labels_b = batch["imgs_labels"]
            logists, feats = model_clip(input_img)
            logists_deeplab, feats_deeplab = model_images(input_img)
            posibi_b = F.softmax(logists * 100, dim=1)
            output_images_b = logists.argmax(dim=1)

            preds_deeplab = logists_deeplab.argmax(dim=1)

            for id in range(len(frame_names)):
                torch.cuda.empty_cache()
                image_name = frame_names[id].split('/')[-1]


                masks_sam = np.load("/dataset/nu-sam-all/0-1-0/sam_nuscenes/nuscenes_sam/" + image_name + ".npy", allow_pickle=True)

                output_images = output_images_b[id]
                img_labels = img_labels_b[id]
                posibi = posibi_b[id].unsqueeze(0)
                image = input_img[id].permute(1, 2, 0)

                clip_orig = output_images.clone()
                deeplab_pred = preds_deeplab[id]

                # get seeds
                # print((posibi > thr).sum(), (posibi > thr).sum() / (posibi.shape[0] * posibi.shape[1] * posibi.shape[2] * posibi.shape[3]))
                output_images_fla = output_images.view(-1, 1)
                posibi_fla = torch.flatten(posibi.permute(0, 2, 3, 1), 0, 2)
                order = torch.tensor(list(range(output_images_fla.shape[0]))).view(-1, 1).to(device=0)
                m = torch.cat((order, output_images_fla), dim=1)
                m = tuple(m.T.long())
                posibi_fla = posibi_fla[m]
                seeds_index = posibi_fla > thr

                # print(len(masks_sam))
                # print(masks_sam[0])
                tot_mask = masks_sam[0]['segmentation']
                seeds_index = seeds_index.view(output_images.shape)
                # print(seeds_index.shape)
                # tt = 0

                # print(tot_mask.shape, clip_orig.shape)

                masks_sam = sorted(masks_sam, key=(lambda x: x['area']), reverse=False)
                for i, mask in enumerate(masks_sam):

                    seg_i = mask['segmentation']
                    # tt += seg_i.sum()
                    # seg_uni = seg_i & seeds_index

                    # print(seeds_index.sum(), seg_i.sum(), seg_uni.sum())

                    # print(seg_i.shape, seeds_index.shape)
                    # if seg_uni.sum() != 0:
                    #     tem_mask = clip_orig[seg_i]
                    # else:
                    #     continue
                        # tem_mask = clip_orig[seg_i]

                    tem_mask = clip_orig[seg_i]
                    # print(tem_mask, torch.mode(tem_mask).values)
                    # print("tem_mask: ", tem_mask.shape)
                    # print("before output_images[seg_i] ", output_images[seg_i])
                    output_images[seg_i] = torch.mode(tem_mask).values
                    # print("output_images[seg_i] ", output_images[seg_i].shape)
                    tot_mask = tot_mask | seg_i
                    # print(tot_mask.sum())


                if tot_mask.sum() == 0: continue
                tot_mask = torch.from_numpy(tot_mask).contiguous()
                corrected_area = tot_mask == 1

                # tot_mask[tot_mask == 0] = False
                # tot_mask[tot_mask == 1] = True

                # print("output_images: ", (output_images - kk_image).sum())
                clip_orig += 1
                output_images += 1
                deeplab_pred += 1

                # output_images[1 - tot_mask] = 0
                # img_labels[1 - tot_mask] = 0

                output_images = output_images.contiguous()

                ratio = tot_mask.sum() / tot_mask.view(-1).shape[0]
                ratios += ratio
                cot += 1

                visual_masks(image, masks_sam, output_images, clip_orig, deeplab_pred, img_labels, image_name,
                             seeds_index)

                # print("", ratio)
                # ratio_b = (1 - tot_mask).sum() / tot_mask.view(-1).shape[0]
                # print("ratio: ", ratio, ratio_b, ratio + ratio_b)

                # print("output_images", output_images[tot_mask].shape)

                # print((tot_mask.sum() + (~tot_mask).sum()))
                # print(tot_mask.sum())
                # print(~tot_mask.sum())
                # print((tot_mask.sum() + (1 - tot_mask).sum()) / (tot_mask.shape[0] * tot_mask.shape[1]))
                # if ratio > 0.8:
                #     visual_masks(image, masks_sam, output_images, clip_orig, deeplab_pred, img_labels, image_name, seeds_index)
                # else:
                #     continue
                # if ratio < 0.9: continue

                # output_images = output_images[seeds_index]

                # output_images = output_images[seeds_index]
                # img_labels = img_labels[seeds_index]

                # output_images = output_images[corrected_area]
                # img_labels = img_labels[corrected_area]


                full_predictions.append(output_images.cpu().view(-1))
                ground_truth.append(deepcopy(img_labels.view(-1)))


                # m_IoU_clip, fw_IoU_clip, per_class_IoU = compute_IoU(
                #     output_images.cpu().view(-1).int(),
                #     img_labels.view(-1).int(),
                #     config["model_n_out"],
                #     ignore_index=0,
                # )
                #
                # m_IoU_deeplab, fw_IoU_deeplab, per_class_IoU = compute_IoU(
                #     deeplab_pred.cpu().view(-1).int(),
                #     img_labels.view(-1).int(),
                #     config["model_n_out"],
                #     ignore_index=0,
                # )
                #
                #
                # if fw_IoU_deeplab > fw_IoU_clip + 0.4 and fw_IoU_deeplab > 0.85:
                #     print(fw_IoU_deeplab, fw_IoU_clip)
                #     visual_masks(image, masks_sam, output_images, clip_orig, deeplab_pred, img_labels, image_name,
                #                  seeds_index)

                torch.cuda.empty_cache()
                # del tot_mask
                # del seeds_index
                # del masks_sam

        print("avg ration: ", ratios / cot)

        full_predictions = torch.cat(full_predictions).int()
        ground_truth = torch.cat(ground_truth).int()

        # if config["dataset"].lower() == "scannet":
        #     ground_truth += 1
        #     ground_truth[ground_truth == -99] = 0

        # print(full_predictions.shape, torch.cat(ground_truth).shape)
        # print(torch.cat(full_predictions), torch.cat(ground_truth))

        print(ground_truth)

        m_IoU, fw_IoU, per_class_IoU = compute_IoU(
            full_predictions,
            ground_truth,
            config["model_n_out"],
            ignore_index=0,
        )

        # import pdb
        # pdb.set_trace()

        print("Per class IoU:")
        if config["dataset"].lower() == "nuscenes":
            print(
                *[
                    f"{a:20} - {b:.3f}"
                    for a, b in zip(CLASSES_NUSCENES, (per_class_IoU).numpy())
                ],
                sep="\n",
            )
        elif config["dataset"].lower() == "kitti":
            print(
                *[
                    f"{a:20} - {b:.3f}"
                    for a, b in zip(CLASSES_KITTI, (per_class_IoU).numpy())
                ],
                sep="\n",
            )
        elif config["dataset"].lower() == "scannet":
            print(
                *[
                    f"{a:20} - {b:.3f}"
                    for a, b in zip(CLASSES_scannet, (per_class_IoU).numpy())
                ],
                sep="\n",
            )
        print()
        print(f"mIoU: {m_IoU}")
        print(f"fwIoU: {fw_IoU}")

    return m_IoU
