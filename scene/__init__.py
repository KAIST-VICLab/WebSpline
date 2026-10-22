#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
import torch
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.dataset import FourDGSdataset
from scene.bundle_adjustment import bundle_adjustment
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from torch.utils.data import Dataset
from scene.dataset_readers import add_points
from utils.scene_utils import filter_track, query_buffers_by_track
from glob import glob

# The DAVIS reader lives in its own module because it imports helpers from
# dataset_readers; registering it here instead of there keeps that one-way.
from scene.dataset_readers_davis import readDavisInfo

sceneLoadTypeCallbacks["davis"] = readDavisInfo


# How many frames a track must be visible in to survive filter_track.  iPhone
# tracks are dense and long, so a strict threshold only drops noise; the NVIDIA
# sequences are ~12 frames, where the same threshold would throw away almost
# everything.
TRACK_MIN_VALID_CNT = {
    "iphone": 4,
    "nvidia": 1,
    "davis": 4,
}

# Substring identifying the tracker output to glob for in the source directory.
TRACK_NPZ_KEY = {
    "iphone": "cotracker",
    "nvidia": "cotracker",
    "davis": "bootstapir",
}


def detect_dataset_type(source_path, override=""):
    """Canonical (lowercase) dataset type for a source directory.

    Every `dataset_type ==` comparison downstream keys on the string returned
    here, so the spellings below are the only ones in use -- a mismatched case
    silently sends a dataset down another dataset's code path.

    `override` (--dataset_type) wins over sniffing, for layouts the markers
    below cannot tell apart.
    """
    if override:
        return override

    exists = lambda *p: os.path.exists(os.path.join(source_path, *p))

    if exists("emf.json"):
        return "iphone"
    # DAVIS is staged by scripts/prepare_davis.py, which always writes the
    # MegaSaM intrinsics next to the per-frame poses.
    if exists("pose", "intrinsics.npy"):
        return "davis"
    if exists("poses_bounds.npy"):
        return "nvidia"
    # Generic COLMAP layout, for custom captures.
    if exists("sparse"):
        return "colmap"
    raise ValueError(
        f"could not recognize the dataset at {source_path}; "
        f"pass --dataset_type to force one"
    )


class Scene:

    dyn_gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, static_gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], load_coarse=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.dyn_gaussians = gaussians
        self.stat_gaussians = static_gaussians
        
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.video_cameras = {}
        
        dataset_type = detect_dataset_type(args.source_path, getattr(args, "dataset_type", ""))
        print(f"Dataset type: {dataset_type}")

        if dataset_type == "iphone":
            scene_info = sceneLoadTypeCallbacks["iphone"](args.source_path, args)
        elif dataset_type == "davis":
            scene_info = sceneLoadTypeCallbacks["davis"](args.source_path, args)
        elif dataset_type == "nvidia":
            scene_info = sceneLoadTypeCallbacks["nvidia"](args.source_path)
        elif dataset_type == "colmap":
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        else:
            raise ValueError(f"no reader registered for dataset type {dataset_type!r}")

        # Track loading differs per dataset in three ways: which tracker produced the
        # npz, how many visible frames a track needs to survive filtering, and whether
        # a static track is needed at all.  Only bundle_adjustment consumes the static
        # track, and DAVIS does not bundle-adjust (its poses come from MegaSaM), so
        # there is nothing to load it for.
        min_valid_cnt = TRACK_MIN_VALID_CNT.get(dataset_type, 4)
        min_valid_cnt = getattr(args, "tap_loading_min_valid_cnt", min_valid_cnt)
        track_key = TRACK_NPZ_KEY.get(dataset_type, "cotracker")
        if dataset_type == "davis":
            self.load_track(args, scene_info, f"*{track_key}", min_valid_cnt=min_valid_cnt, static=False)
        else:
            self.load_track(args, scene_info, f"*uniform*{track_key}", min_valid_cnt=min_valid_cnt, static=True)
            self.load_track(args, scene_info, f"*{track_key}", min_valid_cnt=min_valid_cnt, static=False)
        self.maxtime = scene_info.maxtime
        self.dataset_type = dataset_type
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        print(f"Original scene extent {self.cameras_extent}")
        print("Loading Training Cameras")
        self.train_camera = FourDGSdataset(scene_info.train_cameras, args, dataset_type)
        print("Loading Test Cameras")
        self.test_camera = FourDGSdataset(scene_info.test_cameras, args, dataset_type)
        print("Loading Video Cameras")
        self.video_camera = FourDGSdataset(scene_info.video_cameras, args, dataset_type)

        # self.video_camera = cameraList_from_camInfos(scene_info.video_cameras,-1,args)
        xyz_max = scene_info.point_cloud.points.max(axis=0)
        xyz_min = scene_info.point_cloud.points.min(axis=0)
        if args.add_points:
            print("add points.")
            # breakpoint()
            scene_info = scene_info._replace(point_cloud=add_points(scene_info.point_cloud, xyz_max=xyz_max, xyz_min=xyz_min))
        self.dyn_gaussians._deformation.deformation_net.set_aabb(xyz_max, xyz_min)

        if self.loaded_iter:
            self.dyn_gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            self.dyn_gaussians.load_model(os.path.join(self.model_path,
                                                    "point_cloud",
                                                    "iteration_" + str(self.loaded_iter),
                                                       ))
            self.stat_gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud_static.ply"))
        else:
            self.dyn_gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, self.maxtime)
            self.stat_gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, self.maxtime)

    def save(self, iteration, stage, scaffold=None):
        if stage == "fine_static":
            point_cloud_path = os.path.join(self.model_path, "point_cloud/fine_static_iteration_{}".format(iteration))

        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.dyn_gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.dyn_gaussians.save_deformation(point_cloud_path)

        self.stat_gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud_static.ply"))
        if scaffold is not None:
            torch.save(scaffold.state_dict(), os.path.join(point_cloud_path, "scaffold_state_dict.pt"))

    def save_best_psnr(self, iteration, stage, scaffold=None):
        if stage == "fine_static":
            point_cloud_path = os.path.join(self.model_path, "point_cloud/fine_static_best")

        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/fine_best")
        self.dyn_gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.dyn_gaussians.save_deformation(point_cloud_path)

        self.stat_gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud_static.ply"))
        if scaffold is not None:
            torch.save(scaffold.state_dict(), os.path.join(point_cloud_path, "scaffold_state_dict.pt"))

    def recordpoints(self, iteration, string):
        numpoints_dyn = self.dyn_gaussians._xyz.shape[0]
        numpoints_stat = self.stat_gaussians._xyz.shape[0]
        txtpath = os.path.join(self.model_path, "exp_log.txt")
        with open(txtpath, 'a') as f:
            f.write("iteration at " + str(iteration) + "\n")
            f.write(string + " pointsnumber dyn. " + str(numpoints_dyn) + "\n")
            f.write(string + " pointsnumber stat. " + str(numpoints_stat) + "\n")

    def getTrainCameras(self, scale=1.0):
        return self.train_camera

    def getTestCameras(self, scale=1.0):
        return self.test_camera

    def getVideoCameras(self, scale=1.0):
        return self.video_camera


    def load_track(self, args, scene_info, track_npz_key="spatracker", min_valid_cnt=4, static= True):
        # * load from saved files
        long_track_fns = glob(os.path.join(args.source_path, f"*{track_npz_key}*.npz"))
        assert len(long_track_fns) > 0, "no TAP found!"
        track, track_mask = [], []
        for track_fn in long_track_fns:
            track_data = np.load(track_fn, allow_pickle=True)
            track.append(track_data["tracks"])
            track_mask.append(track_data["visibility"])
        # ! explicitly round to long
        track = torch.from_numpy(np.concatenate(track, 1)).float()  # T,N,2/3
        track[:, :, :2] = track[:, :, :2].long().float()
        track_mask = torch.from_numpy(np.concatenate(track_mask, 1)).bool()  # T,N
        W = scene_info.train_cameras[0].width
        H = scene_info.train_cameras[0].height
        T = len(scene_info.train_cameras)
        track_mask = track_mask * (track[..., 0] >= 0) * (track[..., 1] >= 0)
        track_mask = track_mask * (track[..., 0] < W) * (track[..., 1] < H)
        assert track.shape[:2] == track_mask.shape
        assert len(track) == T
        depth_masks_tensor = torch.stack([
            torch.from_numpy(cam.depth_mask).squeeze(-1)
            for cam in scene_info.train_cameras
        ], dim=0)
        static_masks_tensor = torch.stack([
            torch.from_numpy(cam.mask).squeeze(-1)
            for cam in scene_info.train_cameras
        ], dim=0)
        if static:
            static_depth_mask = (depth_masks_tensor >= 0.5) & (static_masks_tensor < 0.5)
        # * filter the load tracks
            if min_valid_cnt > 0:
                track, track_mask = filter_track(
                    track,
                    track_mask,
                    static_depth_mask.to(track.device),
                    min_valid_cnt=min_valid_cnt,
                )
            self.static_track = track.float()
            self.static_track_mask = track_mask
            if self.static_track.shape[-1] == 3 and hasattr(self, "scale_nw"):
                self.static_track[:, :, 2] = self.static_track[:, :, 2].clone() * self.scale_nw
        else:
            dyn_depth_mask = (depth_masks_tensor >= 0.5) & (static_masks_tensor > 0.5)
            if min_valid_cnt > 0:
                track, track_mask = filter_track(
                    track,
                    track_mask,
                    dyn_depth_mask.to(track.device),
                    min_valid_cnt=min_valid_cnt,
                )
            self.dyn_track = track.float()
            self.dyn_track_mask = track_mask
            if self.dyn_track.shape[-1] == 3 and hasattr(self, "scale_nw"):
                self.dyn_track[:, :, 2] = self.dyn_track[:, :, 2].clone() * self.scale_nw