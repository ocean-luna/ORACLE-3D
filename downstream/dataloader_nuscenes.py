import os
import copy
import torch
import numpy as np
from PIL import Image
import MinkowskiEngine as ME
from pyquaternion import Quaternion
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud
# from torchsparse.utils.quantize import sparse_quantize
from abc import ABC, abstractmethod
import json
import cv2
import pickle


CUSTOM_SPLIT = [
    "scene-0008", "scene-0009", "scene-0019", "scene-0029", "scene-0032", "scene-0042",
    "scene-0045", "scene-0049", "scene-0052", "scene-0054", "scene-0056", "scene-0066",
    "scene-0067", "scene-0073", "scene-0131", "scene-0152", "scene-0166", "scene-0168",
    "scene-0183", "scene-0190", "scene-0194", "scene-0208", "scene-0210", "scene-0211",
    "scene-0241", "scene-0243", "scene-0248", "scene-0259", "scene-0260", "scene-0261",
    "scene-0287", "scene-0292", "scene-0297", "scene-0305", "scene-0306", "scene-0350",
    "scene-0352", "scene-0358", "scene-0361", "scene-0365", "scene-0368", "scene-0377",
    "scene-0388", "scene-0391", "scene-0395", "scene-0413", "scene-0427", "scene-0428",
    "scene-0438", "scene-0444", "scene-0452", "scene-0453", "scene-0459", "scene-0463",
    "scene-0464", "scene-0475", "scene-0513", "scene-0533", "scene-0544", "scene-0575",
    "scene-0587", "scene-0589", "scene-0642", "scene-0652", "scene-0658", "scene-0669",
    "scene-0678", "scene-0687", "scene-0701", "scene-0703", "scene-0706", "scene-0710",
    "scene-0715", "scene-0726", "scene-0735", "scene-0740", "scene-0758", "scene-0786",
    "scene-0790", "scene-0804", "scene-0806", "scene-0847", "scene-0856", "scene-0868",
    "scene-0882", "scene-0897", "scene-0899", "scene-0976", "scene-0996", "scene-1012",
    "scene-1015", "scene-1016", "scene-1018", "scene-1020", "scene-1024", "scene-1044",
    "scene-1058", "scene-1094", "scene-1098", "scene-1107",
]


def minkunet_collate_pair_fn(list_data):
    """
    Collate function adapted for creating batches with MinkowskiEngine.
    """
    (
        pc,
        coords,
        feats,
        images,
        pairing_points,
        pairing_images,
        inverse_indexes,
        image_names,
        masks_sams,
        points_labels,
        unique_labels,
        lidar_name,
        embeddings_sams,
    ) = list(zip(*list_data))
    batch_n_points, batch_n_pairings = [], []

    offset = 0
    len_batch = []
    for batch_id in range(len(coords)):

        # Move batchids to the beginning
        coords[batch_id][:, 0] = batch_id
        pairing_points[batch_id][:] += offset
        pairing_images[batch_id][:, 0] += batch_id * images[0].shape[0]

        batch_n_points.append(coords[batch_id].shape[0])
        batch_n_pairings.append(pairing_points[batch_id].shape[0])
        offset += coords[batch_id].shape[0]

        N = coords[batch_id].shape[0]
        len_batch.append(N)

    embeddings_sams = torch.from_numpy(np.stack(embeddings_sams, axis=0)).squeeze(0).squeeze(1)
    embeddings_sams=  embeddings_sams.contiguous().view(-1, 256, 64, 64)
    # Concatenate all lists
    coords_batch = torch.cat(coords, 0).int()
    unique_labels = torch.cat(unique_labels, 0).long()
    pairing_points = torch.tensor(np.concatenate(pairing_points))
    pairing_images = torch.tensor(np.concatenate(pairing_images)).int().contiguous()
    feats_batch = torch.cat(feats, 0).float()
    images_batch = torch.cat(images, 0).float()
    image_names = [name for sublist in image_names for name in sublist]
    return {
        "sinput_C": coords_batch,
        "sinput_F": feats_batch,
        "input_I": images_batch,
        "pairing_points": pairing_points,
        "pairing_images": pairing_images,
        "batch_n_pairings": batch_n_pairings,
        "inverse_indexes": inverse_indexes,
        "image_names": image_names,
        "masks_sams": masks_sams,
        "len_batch": len_batch,
        "embeddings_sams": embeddings_sams,
        "pc": pc,
        "lidar_name": lidar_name,
        "labels": unique_labels,
        "evaluation_labels": points_labels,  # labels for each point
    }


def spvcnn_collate_pair_fn(list_data):
    """
    Collate function adapted for creating batches with MinkowskiEngine.
    """
    (
        coords,
        feats,
        images,
        pairing_points,
        pairing_images,
        inverse_indexes,
        inverse_indexes_merged,
        sweepIds_group,
        sweep_pairing_group,
    ) = list(zip(*list_data))
    batch_n_points, batch_n_pairings = [], []

    offset = 0
    offset_inverse_indexes = 0

    for batch_id in range(len(coords)):

        # Move batchids to the beginning
        coords[batch_id][:, -1] = batch_id
        pairing_points[batch_id][:] += offset_inverse_indexes
        pairing_images[batch_id][:, 0] += batch_id * images[0].shape[0]
        inverse_indexes[batch_id][:] += offset
        inverse_indexes_merged[batch_id][:] += offset

        batch_n_points.append(coords[batch_id].shape[0])
        batch_n_pairings.append(pairing_points[batch_id].shape[0])
        offset += coords[batch_id].shape[0]
        offset_inverse_indexes += inverse_indexes[batch_id].shape[0]

    coords_batch = torch.cat(coords, 0).int()
    pairing_points = torch.cat(pairing_points, 0)
    pairing_images = torch.cat(pairing_images, 0)
    feats_batch = torch.cat(feats, 0).float()
    images_batch = torch.cat(images, 0).float()
    sweepIds_group = torch.cat(sweepIds_group, 0)
    inverse_indexes_merged = torch.cat(inverse_indexes_merged, 0)
    inverse_indexes_group = torch.cat(inverse_indexes, 0)

    return {
        "sinput_C": coords_batch,
        "sinput_F": feats_batch,
        "input_I": images_batch,
        "pairing_points": pairing_points,
        "pairing_images": pairing_images,
        "batch_n_pairings": batch_n_pairings,
        "inverse_indexes_group": inverse_indexes_group,
        "inverse_indexes_merged": inverse_indexes_merged,
        "sweepIds": sweepIds_group,
        "sweep_pairing_group": sweep_pairing_group,
    }


class NuScenesDataset(Dataset):
    """
    Dataset matching a 3D points cloud and an image using projection.
    """

    def __init__(
        self,
        phase,
        config,
        shuffle=False,
        cloud_transforms=None,
        mixed_transforms=None,
        **kwargs,
    ):
        self.phase = phase
        self.shuffle = shuffle
        self.cloud_transforms = cloud_transforms
        self.mixed_transforms = mixed_transforms
        self.voxel_size = config["voxel_size"]
        self.cylinder = config["cylindrical_coordinates"]
        self.superpixels_type = config["superpixels_type"]
        self.bilinear_decoder = config["decoder"] == "bilinear"
        self.config = config
        self.dataroot = config['dataRoot_nuscenes']
        self.eval_labels = {
            0: 0, 1: 0, 2: 7, 3: 7, 4: 7, 5: 0, 6: 7, 7: 0, 8: 0, 9: 1, 10: 0, 11: 0,
            12: 8, 13: 0, 14: 2, 15: 3, 16: 3, 17: 4, 18: 5, 19: 0, 20: 0, 21: 6, 22: 9,
            23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 0, 30: 16, 31: 0,
        }
        if "cached_nuscenes" in kwargs:
            self.nusc = kwargs["cached_nuscenes"]
        else:
            self.nusc = NuScenes(
                version="v1.0-trainval", dataroot=self.dataroot, verbose=False
            )

        self.list_keyframes = []
        # a skip ratio can be used to reduce the dataset size and accelerate experiments
        try:
            skip_ratio = config["dataset_skip_step"]
        except KeyError:
            skip_ratio = 1
        self.imageDim = (416, 224)#(416, 224)
        self.imageDim_origin = (1640, 900)
        skip_counter = 0
        if phase in ("train", "val", "test"):
            phase_scenes = create_splits_scenes()[phase]
        elif phase == "parametrizing":
            phase_scenes = list(
                set(create_splits_scenes()["train"]) - set(CUSTOM_SPLIT)
            )
        elif phase == "verifying":
            phase_scenes = CUSTOM_SPLIT
        # create a list of camera & lidar scans
        for scene_idx in range(len(self.nusc.scene)):
            scene = self.nusc.scene[scene_idx]
            if scene["name"] in phase_scenes:
                skip_counter += 1
                if skip_counter % skip_ratio == 0:
                    self.create_list_of_scans(scene)

        with open('/volumes/autopilot-root-muses/public_dataset/nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_val.pkl', 'rb') as f:
            self.sweeps_infos = pickle.load(f)
        tem = {}
        for info in self.sweeps_infos:
            tem[info['lidar_path']] = {'sweeps': info['sweeps']}
        self.sweeps_infos = tem
        self.max_sweeps = self.config['max_sweeps']



        print(phase)
        print(len(phase_scenes))

    def create_list_of_scans(self, scene):
        # Get first and last keyframe in the scene
        current_sample_token = scene["first_sample_token"]
        # print("current_sample_token", current_sample_token)
        # Loop to get all successive keyframes
        list_data = []
        while current_sample_token != "":
            current_sample = self.nusc.get("sample", current_sample_token)
            list_data.append(current_sample["data"])
            current_sample_token = current_sample["next"]

        # Add new scans in the list
        self.list_keyframes.extend(list_data)


    def get_sweep(self, sweep_info):
        def remove_ego_points(points, center_radius=1.0):
            mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
            return points[mask]

        lidar_name = sweep_info['lidar_path']
        lidar_path = os.path.join(self.dataroot, lidar_name)
        pc_original = LidarPointCloud.from_file(lidar_path)
        points_sweep = pc_original.points.T[:, :4]

        points_sweep = remove_ego_points(points_sweep).T
        if sweep_info['transform_matrix'] is not None:
            num_points = points_sweep.shape[1]
            points_sweep[:3, :] = sweep_info['transform_matrix'].dot(
                np.vstack((points_sweep[:3, :], np.ones(num_points))))[:3, :]

        cur_times = sweep_info['time_lag'] * np.ones((1, points_sweep.shape[1]))
        return points_sweep.T, cur_times.T

    def get_lidar_with_sweeps(self, lidar_name, max_sweeps=1):
        info = self.sweeps_infos[lidar_name]
        lidar_path = os.path.join(self.nusc.dataroot, lidar_name)

        pc_original = LidarPointCloud.from_file(lidar_path)
        points = pc_original.points.T[:, :4]

        name_list = [lidar_name]
        sweep_points_list = [points]

        for k in np.random.choice(len(info['sweeps']), max_sweeps - 1, replace=False):
            points_sweep, times_sweep = self.get_sweep(info['sweeps'][k])
            sweep_points_list.append(points_sweep)
            name_list.append(info['sweeps'][k]['lidar_path'])

        points = np.concatenate(sweep_points_list, axis=0)
        return sweep_points_list, points

    def map_pointcloud_to_image(self, data, min_dist: float = 1.0):
        """
        Given a lidar token and camera sample_data token, load pointcloud and map it to
        the image plane. Code adapted from nuscenes-devkit
        https://github.com/nutonomy/nuscenes-devkit.
        :param min_dist: Distance from the camera below which points are discarded.
        """
        pointsensor = self.nusc.get("sample_data", data["LIDAR_TOP"])
        # pc_original = LidarPointCloud.from_points(sweep_points)
        pcl_path = os.path.join(self.nusc.dataroot, pointsensor["filename"])
        pc_original = LidarPointCloud.from_file(pcl_path)
        pc_ref = pc_original.points

        images = []
        pairing_points = np.empty(0, dtype=np.int64)
        pairing_images = np.empty((0, 3), dtype=np.int64)
        camera_list = [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_FRONT_LEFT",
        ]
        if self.shuffle:
            np.random.shuffle(camera_list)

        image_names = []
        images = []
        masks_sams = []
        embeddings_sams = []
        for i, camera_name in enumerate(camera_list):
            pc = copy.deepcopy(pc_original)
            cam = self.nusc.get("sample_data", data[camera_name])
            # im = image_buffer[camera_name]
            image_names.append(cam["filename"])
            image_name = cam["filename"].split('/')[-1]
            save_path = "/down_sampled_images/" + image_name

            im = np.array(Image.open(os.path.join(self.nusc.dataroot, cam["filename"])))
            im = cv2.resize(im, self.imageDim)
            # try:
            #     im = cv2.imread(save_path)
            # except:
            #     im = np.array(Image.open(os.path.join(self.nusc.dataroot, cam["filename"])))
            #     im = cv2.resize(im, self.imageDim)
            #     # cv2.imwrite(save_path, im)

            save_name = "/dataset/nu-sam-all/0-1-0/sam_nuscenes/nuscenes_sam/" + image_name + ".npy"
            masks_sam = np.load(save_name, allow_pickle=True)
            for k, mask in enumerate(masks_sam):
                com=masks_sam[k]['segmentation'].astype(np.uint8) * 255
                masks_sam[k]['segmentation']=cv2.resize(com, self.imageDim)>127

            # print(len(masks_sam))
            # print(masks_sam[0].keys())
            # print(masks_sam['segmentation'].shape)
            # mask = masks_sam['segmentation']
            # mask = cv2.resize(im, self.)
            # print(mask.shape)

            masks_sams.append(masks_sam)

            embeddings_sam = np.load("/dataset/nu-sam-all/0-1-0/sam_nuscenes/nuscenes_sam/" + image_name + "_samImage_embedding.npy", allow_pickle=True)
            embeddings_sams.append(embeddings_sam.squeeze(0))

            # /sharedata/home/chenrn/projects/SAM2Point/down_sampled_images

            # print(os.path.join(self.nusc.dataroot, cam["filename"]), im.shape)
            # self.process_sam(cam["filename"], im)

            # Points live in the point sensor frame. So they need to be transformed via
            # global to the image plane.
            # First step: transform the pointcloud to the ego vehicle frame for the
            # timestamp of the sweep.
            cs_record = self.nusc.get(
                "calibrated_sensor", pointsensor["calibrated_sensor_token"]
            )
            pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix)
            pc.translate(np.array(cs_record["translation"]))

            # Second step: transform from ego to the global frame.
            poserecord = self.nusc.get("ego_pose", pointsensor["ego_pose_token"])
            pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix)
            pc.translate(np.array(poserecord["translation"]))

            # Third step: transform from global into the ego vehicle frame for the
            # timestamp of the image.
            poserecord = self.nusc.get("ego_pose", cam["ego_pose_token"])
            pc.translate(-np.array(poserecord["translation"]))
            pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix.T)

            # Fourth step: transform from ego into the camera.
            cs_record = self.nusc.get(
                "calibrated_sensor", cam["calibrated_sensor_token"]
            )
            pc.translate(-np.array(cs_record["translation"]))
            pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix.T)

            # Fifth step: actually take a "picture" of the point cloud.
            # Grab the depths (camera frame z axis points away from the camera).
            depths = pc.points[2, :]

            # Take the actual picture
            # (matrix multiplication with camera-matrix + renormalization).
            points = view_points(
                pc.points[:3, :],
                np.array(cs_record["camera_intrinsic"]),
                normalize=True,
            )

            # Remove points that are either outside or behind the camera.
            # Also make sure points are at least 1m in front of the camera to avoid
            # seeing the lidar points on the camera
            # casing for non-keyframes which are slightly out of sync.
            points = points[:2].T
            mask = np.ones(depths.shape[0], dtype=bool)
            mask = np.logical_and(mask, depths > min_dist)
            mask = np.logical_and(mask, points[:, 0] > 0)
            mask = np.logical_and(mask, points[:, 0] < self.imageDim_origin[0] - 1)
            mask = np.logical_and(mask, points[:, 1] > 0)
            mask = np.logical_and(mask, points[:, 1] < self.imageDim_origin[1] - 1)
            matching_points = np.where(mask)[0]


            matching_pixels = np.round(
                np.flip(points[matching_points], axis=1)
            ).astype(np.int64).astype(np.float64)

            ratio_x = self.imageDim_origin[1] / self.imageDim[1]
            ratio_y = self.imageDim_origin[0] / self.imageDim[0]
            # matching_pixels[:, 0][:] /= ratio_x
            # matching_pixels[:, 1][:] /= ratio_y

            matching_pixels[:, 0] = np.clip(matching_pixels[:, 0] / ratio_x, 0, self.imageDim[1] - 1)
            matching_pixels[:, 1] = np.clip(matching_pixels[:, 1] / ratio_y, 0, self.imageDim[0] - 1)

            matching_pixels = matching_pixels.astype(np.int64)


            images.append(im / 255)
            pairing_points = np.concatenate((pairing_points, matching_points))
            pairing_images = np.concatenate(
                (
                    pairing_images,
                    np.concatenate(
                        (
                            np.ones((matching_pixels.shape[0], 1), dtype=np.int64) * i,
                            matching_pixels,
                        ),
                        axis=1,
                    ),
                )
            )



        return pc_ref.T, images, pairing_points, pairing_images, image_names, masks_sams, embeddings_sams



    def __len__(self):
        return len(self.list_keyframes)


    def voxelizaton(self, pc):
        if self.cylinder:
            # Transform to cylinder coordinate and scale for voxel size
            x, y, z = pc.T
            rho = torch.sqrt(x ** 2 + y ** 2) / self.voxel_size
            phi = torch.atan2(y, x) * 180 / np.pi  # corresponds to a split each 1°
            z = z / self.voxel_size
            coords_aug = torch.cat((rho[:, None], phi[:, None], z[:, None]), 1)
        else:
            coords_aug = pc / self.voxel_size

        if self.config["model_points"] == "spvcnn":
            pass
            # discrete_coords, indexes, inverse_indexes = sparse_quantize(
            #     coords_aug.contiguous().numpy(), return_index=True, return_inverse=True
            # )
            # discrete_coords, indexes, inverse_indexes = torch.from_numpy(discrete_coords), torch.from_numpy(indexes), torch.from_numpy(inverse_indexes)
            # return discrete_coords, indexes, inverse_indexes

        elif self.config["model_points"] == "minkunet":
            discrete_coords, indexes, inverse_indexes = ME.utils.sparse_quantize(
                coords_aug.contiguous(), return_index=True, return_inverse=True
            )
            return discrete_coords, indexes, inverse_indexes

    def preload_images(self, data):
        camera_list = [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_FRONT_LEFT",
        ]
        image_buffer = {}
        for i, camera_name in enumerate(camera_list):
            cam = self.nusc.get("sample_data", data[camera_name])
            im = np.array(Image.open(os.path.join(self.nusc.dataroot, cam["filename"])))
            image_buffer[camera_name] = im

        return image_buffer

    def __getitem__(self, idx):
        (
            pc,
            images,
            pairing_points,
            pairing_images,
            image_names,
            masks_sams,
            embeddings_sams,
        ) = self.map_pointcloud_to_image(self.list_keyframes[idx])

        data = torch.from_numpy(pc)
        intensity = torch.tensor(pc[:, 3:])
        pc = torch.tensor(pc[:, :3])
        images = torch.tensor(np.array(images, dtype=np.float32).transpose(0, 3, 1, 2))

        # self.imageDim = (416, 224)
        # self.imageDim_origin = (1640, 900)
        # ratio_x = self.imageDim_origin[1] / self.imageDim[1]
        # ratio_y = self.imageDim_origin[0] / self.imageDim[0]
        # pairing_images[:, 1][:] /= ratio_x
        # pairing_images[:, 2][:] /= ratio_y

        # pairing_images = pairing_images / max(ratio_x, ratio_y)

        lidarseg_labels_filename = os.path.join(
            self.nusc.dataroot, self.nusc.get("lidarseg", self.list_keyframes[idx]["LIDAR_TOP"])["filename"]
        )

        # 30af13f6e00747998fc4a4f4fe7734d2_lidarseg.bin
        lidar_name = self.nusc.get("lidarseg", self.list_keyframes[idx]["LIDAR_TOP"])["filename"]
        lidar_name = lidar_name.split('/')[-1]


        # print(lidar_name)
        points_labels = np.fromfile(lidarseg_labels_filename, dtype=np.uint8)


        if self.cloud_transforms:
            pc = self.cloud_transforms(pc)
        # if self.mixed_transforms:
        #     (
        #         pc,
        #         intensity,
        #         images,
        #         pairing_points,
        #         pairing_images,
        #     ) = self.mixed_transforms(
        #         pc, intensity, images, pairing_points, pairing_images
        #     )

        if self.cylinder:
            # Transform to cylinder coordinate and scale for voxel size
            x, y, z = pc.T
            rho = torch.sqrt(x ** 2 + y ** 2) / self.voxel_size
            phi = torch.atan2(y, x) * 180 / np.pi  # corresponds to a split each 1°
            z = z / self.voxel_size
            coords_aug = torch.cat((rho[:, None], phi[:, None], z[:, None]), 1)
        else:
            coords_aug = pc / self.voxel_size

        # Voxelization with MinkowskiEngine
        discrete_coords, indexes, inverse_indexes = ME.utils.sparse_quantize(
            coords_aug.contiguous(), return_index=True, return_inverse=True
        )
        # indexes here are the indexes of points kept after the voxelization
        pairing_points = inverse_indexes[pairing_points]

        unique_feats = intensity[indexes]


        points_labels = torch.tensor(
            np.vectorize(self.eval_labels.__getitem__)(points_labels),
            dtype=torch.int32,
        )
        unique_labels = points_labels[indexes]


        discrete_coords = torch.cat(
            (
                torch.zeros(discrete_coords.shape[0], 1, dtype=torch.int32),
                discrete_coords,
            ),
            1,
        )

        return (
            data,
            discrete_coords,
            unique_feats,
            images,
            pairing_points,
            pairing_images,
            inverse_indexes,
            image_names,
            masks_sams,
            points_labels,
            unique_labels,
            lidar_name,
            embeddings_sams,
        )

def make_data_loader(config, phase, num_threads=0):
    """
    Create the data loader for a given phase and a number of threads.
    This function is not used with pytorch lightning, but is used when evaluating.
    """
    # select the desired transformations
    if phase == "train":
        transforms = make_transforms_clouds(config)
    else:
        transforms = None

    # instantiate the dataset
    dset = NuScenesDataset(phase=phase, transforms=transforms, config=config)
    if config['model_points'] == "spvcnn":
        collate_fn = spvcnn_collate_pair_fn
    elif config['model_points'] == "minkunet":
        collate_fn = minkunet_collate_pair_fn
    batch_size = config["batch_size"] // config["num_gpus"]

    # create the loader
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=phase == "train",
        num_workers=num_threads,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=phase == "train",
        worker_init_fn=lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
    )
    return loader
