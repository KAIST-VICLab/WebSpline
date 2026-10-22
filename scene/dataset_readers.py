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
import sys
import cv2
from PIL import Image
from scene.cameras import Camera
import glob

from typing import NamedTuple, Optional
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
import torchvision.transforms as transforms
import copy
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import torch
import torch.nn.functional as F
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from utils.general_utils import PILtoTorch, laplacian_filter_depth
from tqdm import tqdm

from easydict import EasyDict as edict
import imageio

import dycheck_geometry
from utils.dycheck_utils import io

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    time : float
    max_time: float
    mask: np.array
    sem_mask: Optional[np.array] = None
    metadata: Optional[dycheck_geometry.Camera] = None
    normal: Optional[np.array] = None
    depth: Optional[np.array] = None
    fwd_flow: Optional[np.array] = None
    bwd_flow: Optional[np.array] = None
    fwd_flow_mask: Optional[np.array] = None
    bwd_flow_mask: Optional[np.array] = None
    instance_mask: Optional[np.array] = None


    K: Optional[np.array] = None
    covisible: Optional[np.array] = None
    depth_mask: Optional[np.array] = None
class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    video_cameras: list
    nerf_normalization: dict
    ply_path: str
    maxtime: int


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center
    return {"translate": translate, "radius": radius}

def get_normals(z, camera_metadata):
    pixels = camera_metadata.get_pixels()
    y = (pixels[..., 1] - camera_metadata.principal_point_y) / camera_metadata.scale_factor_y
    x = (
                pixels[..., 0] - camera_metadata.principal_point_x - y * camera_metadata.skew
        ) / camera_metadata.scale_factor_x
    viewdirs = np.stack([x, y, np.ones_like(x)], axis=-1)
    viewdirs = torch.from_numpy(viewdirs).to(z.device)

    coords = viewdirs[None] * z[..., None]
    coords = coords.permute(0, 3, 1, 2)

    dxdu = coords[..., 0, :, 1:] - coords[..., 0, :, :-1]
    dydu = coords[..., 1, :, 1:] - coords[..., 1, :, :-1]
    dzdu = coords[..., 2, :, 1:] - coords[..., 2, :, :-1]
    dxdv = coords[..., 0, 1:, :] - coords[..., 0, :-1, :]
    dydv = coords[..., 1, 1:, :] - coords[..., 1, :-1, :]
    dzdv = coords[..., 2, 1:, :] - coords[..., 2, :-1, :]

    dxdu = torch.nn.functional.pad(dxdu, (0, 1), mode='replicate')
    dydu = torch.nn.functional.pad(dydu, (0, 1), mode='replicate')
    dzdu = torch.nn.functional.pad(dzdu, (0, 1), mode='replicate')

    dxdv = torch.cat([dxdv, dxdv[..., -1:, :]], dim=-2)
    dydv = torch.cat([dydv, dydv[..., -1:, :]], dim=-2)
    dzdv = torch.cat([dzdv, dzdv[..., -1:, :]], dim=-2)

    n_x = dydv * dzdu - dydu * dzdv
    n_y = dzdv * dxdu - dzdu * dxdv
    n_z = dxdv * dydu - dxdu * dydv

    pred_normal = torch.stack([n_x, n_y, n_z], dim=-3)
    pred_normal = torch.nn.functional.normalize(pred_normal, dim=-3)
    return pred_normal

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "OPENCV":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        image = PILtoTorch(image,None)
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              time = float(idx/len(cam_extrinsics)), mask=None) # default by monocular settings.
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    times = np.vstack([vertices['t']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals, times=times)

def storePly(path, xyzt, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),('t','f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    xyz = xyzt[:, :3]
    normals = np.zeros_like(xyz)

    elements = np.empty(xyzt.shape[0], dtype=dtype)
    attributes = np.concatenate((xyzt, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    # breakpoint()
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    
    try:
        pcd = fetchPly(ply_path)
        
    except:
        pcd = None
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=train_cam_infos,
                           maxtime=0,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info
def generateCamerasFromTransforms(path, template_transformsfile, extension, maxtime):
    trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

    rot_phi = lambda phi : torch.Tensor([
        [1,0,0,0],
        [0,np.cos(phi),-np.sin(phi),0],
        [0,np.sin(phi), np.cos(phi),0],
        [0,0,0,1]]).float()

    rot_theta = lambda th : torch.Tensor([
        [np.cos(th),0,-np.sin(th),0],
        [0,1,0,0],
        [np.sin(th),0, np.cos(th),0],
        [0,0,0,1]]).float()
    def pose_spherical(theta, phi, radius):
        c2w = trans_t(radius)
        c2w = rot_phi(phi/180.*np.pi) @ c2w
        c2w = rot_theta(theta/180.*np.pi) @ c2w
        c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
        return c2w
    cam_infos = []
    # generate render poses and times
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,160+1)[:-1]], 0)
    render_times = torch.linspace(0,maxtime,render_poses.shape[0])
    with open(os.path.join(path, template_transformsfile)) as json_file:
        template_json = json.load(json_file)
        try:
            fovx = template_json["camera_angle_x"]
        except:
            fovx = focal2fov(template_json["fl_x"], template_json['w'])
    print("hello!!!!")
    # breakpoint()
    # load a single image to get image info.
    for idx, frame in enumerate(template_json["frames"]):
        cam_name = os.path.join(path, frame["file_path"] + extension)
        image_path = os.path.join(path, cam_name)
        image_name = Path(cam_name).stem
        image = Image.open(image_path)
        im_data = np.array(image.convert("RGBA"))
        image = PILtoTorch(image,(800,800))
        break
    # format information
    for idx, (time, poses) in enumerate(zip(render_times,render_poses)):
        time = time/maxtime
        matrix = np.linalg.inv(np.array(poses))
        R = -np.transpose(matrix[:3,:3])
        R[:,0] = -R[:,0]
        T = -matrix[:3, 3]
        fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[2])
        FovY = fovy 
        FovX = fovx
        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=None, image_name=None, width=image.shape[1], height=image.shape[2],
                            time = time, mask=None))
    return cam_infos
def format_infos(dataset,split):
    # loading
    cameras = []
    image = dataset[0][0]
    if split == "train":
        for idx in tqdm(range(len(dataset))):
            image_path = None
            image_name = f"{idx}"
            time = dataset.image_times[idx]
            # matrix = np.linalg.inv(np.array(pose))
            R,T = dataset.load_pose(idx)
            FovX = focal2fov(dataset.focal[0], image.shape[1])
            FovY = focal2fov(dataset.focal[0], image.shape[2])
            cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
                                time = time, mask=None))

    return cameras
def format_render_poses(poses,data_infos):
    cameras = []
    tensor_to_pil = transforms.ToPILImage()
    len_poses = len(poses)
    times = [i/len_poses for i in range(len_poses)]
    image = data_infos[0][0]
    for idx, p in tqdm(enumerate(poses)):
        # image = None
        image_path = None
        image_name = f"{idx}"
        time = times[idx]
        pose = np.eye(4)
        pose[:3,:] = p[:3,:]
        # matrix = np.linalg.inv(np.array(pose))
        R = pose[:3,:3]
        R = - R
        R[:,0] = -R[:,0]
        T = -pose[:3,3].dot(R)
        FovX = focal2fov(data_infos.focal[0], image.shape[2])
        FovY = focal2fov(data_infos.focal[0], image.shape[1])
        cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
                            time = time, mask=None))
    return cameras

def add_points(pointsclouds, xyz_min, xyz_max):
    add_points = (np.random.random((100000, 3)))* (xyz_max-xyz_min) + xyz_min
    add_points = add_points.astype(np.float32)
    addcolors = np.random.random((100000, 3)).astype(np.float32)
    addnormals = np.random.random((100000, 3)).astype(np.float32)
    # breakpoint()
    new_points = np.vstack([pointsclouds.points,add_points])
    new_colors = np.vstack([pointsclouds.colors,addcolors])
    new_normals = np.vstack([pointsclouds.normals,addnormals])
    pointsclouds=pointsclouds._replace(points=new_points)
    pointsclouds=pointsclouds._replace(colors=new_colors)
    pointsclouds=pointsclouds._replace(normals=new_normals)
    return pointsclouds
    # breakpoint()
    # new_

def plot_camera_orientations(cam_list, xyz):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax2 = fig.add_subplot(122, projection='3d')
    # xyz = xyz[xyz[:,0]<1]
    threshold=2
    xyz = xyz[(xyz[:, 0] >= -threshold) & (xyz[:, 0] <= threshold) &
                         (xyz[:, 1] >= -threshold) & (xyz[:, 1] <= threshold) &
                         (xyz[:, 2] >= -threshold) & (xyz[:, 2] <= threshold)]

    ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c='r',s=0.1)
    for cam in tqdm(cam_list):
        # 提取 R 和 T
        R = cam.R
        T = cam.T

        direction = R @ np.array([0, 0, 1])

        ax.quiver(T[0], T[1], T[2], direction[0], direction[1], direction[2], length=1)

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.savefig("output.png")
    # breakpoint()

def readIphoneCameras(path, depth_std):
    # # iphone norm
    # coord_scale = np.array(scene_json['scale']).astype(np.float32)
    # scene_center = np.array(scene_json['center']).astype(np.float32)

    # som norm
    scene_json = io.load(os.path.join(path, "scene.json"))
    coord_scale = np.array(scene_json['scale']).astype(np.float32)
    scene_center = np.array(scene_json['center']).astype(np.float32)

    scene_factor = 2

    train_split_json = io.load(os.path.join(path, "splits", "train.json"))
    test_split_json = io.load(os.path.join(path, "splits", "val.json"))
    train_times = np.array(train_split_json["time_ids"], np.uint32)
    max_time = train_times.max()
    # max_time = 199
    train_cam_infos = []
    test_cam_infos = []

    os.makedirs(os.path.join(path, 'uni_normal'), exist_ok=True)

    # iphone depth
    depth_list = []
    for idx, (time_id, camera_id) in enumerate(zip(np.array(train_split_json["time_ids"], np.uint32), np.array(train_split_json["camera_ids"], np.uint32))):
        frame_name = "{}_{}".format(str(camera_id), str(time_id).zfill(5))
        depth_path = os.path.join(path, 'depth', "{}x".format(str(scene_factor)), frame_name + ".npy")
        depth = np.load(depth_path) * coord_scale
        depth_list.append(depth)
        
    mean_depth = np.mean(np.stack(depth_list, 0))
    # Load train infos
    for idx, (time_id, camera_id) in enumerate(zip(np.array(train_split_json["time_ids"], np.uint32), np.array(train_split_json["camera_ids"], np.uint32))):
        if time_id <= max_time:
            frame_name = "{}_{}".format(str(camera_id), str(time_id).zfill(5))
            img_path = os.path.join(
                            path,
                            'rgb',
                            "{}x".format(str(scene_factor)),
                            frame_name + ".png",
                        )

            curr_img_rgba = Image.open(img_path)
            curr_img = np.array(curr_img_rgba.convert('RGB'))
            curr_img = Image.fromarray((curr_img).astype(np.uint8))

            image_name = Path(img_path).stem

            # nerfies scale first, depth scale later
            # iphone cam
            camera = (
                        dycheck_geometry.Camera.fromjson(
                            os.path.join(path, 'camera', frame_name + ".json")
                        )
                        .rescale_image_domain(1 / scene_factor)
                        .translate(-scene_center)
                        .rescale(coord_scale)
                        .rescale(1/mean_depth)
                    )
            K = camera.intrin
            R = camera.orientation.T
            T = camera.translation #w2c
            focal = camera.focal_length
            fid = time_id / max_time
            FovY = focal2fov(focal, curr_img.size[1])
            FovX = focal2fov(focal, curr_img.size[0])

            depth_path = os.path.join(
                            path,
                            'depth',
                            "{}x".format(str(scene_factor)),
                            frame_name + ".npy",
                        )
            depth = np.load(depth_path) * coord_scale 
            depth = depth / mean_depth

            depth_mask = (depth > 1e-3) * (depth < 1000)
                            
            normal_path = img_path.replace('rgb/2x', 'uni_normal').replace('.png', '.npy')
            if not os.path.exists(normal_path):
                normal = get_normals(torch.from_numpy(depth).squeeze(-1)[None], camera)
                normal = normal.squeeze(0).cpu().numpy().transpose(1,2,0)
                np.save(normal_path, normal)
            else:
                normal = np.load(normal_path)
            if np.isnan(normal).any():
                breakpoint()

            normal = F.avg_pool2d(torch.from_numpy(normal).permute(2,0,1)[None], 5, stride=1, padding=2).squeeze(0).permute(1,2,0).numpy()

            instance_path = img_path.replace('rgb/2x', 'masks/cam0')
            instance_mask_list = []

            instance_mask = np.array(Image.open(instance_path))[..., None]

            if instance_mask.shape[2] != 1:
                instance_mask = instance_mask[..., 0, :]

            instance_mask[instance_mask > 0] = 1
            instance_mask_list.append(instance_mask)

            instance_mask_list = np.stack(instance_mask_list, 0)
            if instance_mask.sum() == 0:
                breakpoint()

            new_mask = instance_mask_list[0]
            if depth_std is not None:
                unidepth_path = img_path.replace('rgb/2x', 'uni_depth').replace('.png', '.npy')
                unidepth = np.load(unidepth_path)
                scale = np.median(depth[depth_mask]) / (np.median(unidepth[depth_mask]) + 1e-8)
                shift = np.median(depth[depth_mask] - scale * unidepth[depth_mask])
                aligned_uni = scale * unidepth + shift
                diff = depth - aligned_uni
                dynamic_region = (new_mask == 1) & depth_mask
                dynamic_diffs = diff[dynamic_region]
                diff_mean = np.mean(dynamic_diffs)
                diff_std = np.std(dynamic_diffs)
                threshold = diff_mean + depth_std * diff_std

                # Outliers: inside the dynamic region and with an error above the threshold
                outlier_mask = (new_mask == 1) & (diff > threshold)

                # Final depth_mask
                # Subtract the outliers from valid_sensor (which already drops 0 / inf)
                depth_mask = depth_mask & (~outlier_mask)
            
            # depth_mask = (depth > 0).astype(np.float32)

            cam_info = CameraInfo(uid=int(image_name[2:]), R=R.astype(np.float32), T=T.astype(np.float32), FovY=float(FovY), FovX=float(FovX), image=curr_img, image_path=img_path, max_time=float(max_time),
                                image_name=image_name, width=curr_img.size[0], height=curr_img.size[1],
                                time=float(fid), mask=new_mask, metadata=camera, normal=normal.astype(np.float32), depth=depth.astype(np.float32), sem_mask=None,
                                instance_mask=instance_mask_list, K=K, depth_mask=depth_mask
                                )
            train_cam_infos.append(cam_info)
            
    # Load test cams
    for idx, (time_id, camera_id) in enumerate(zip(np.array(test_split_json["time_ids"], np.uint32), np.array(test_split_json["camera_ids"], np.uint32))):
        if time_id <= max_time:
            frame_name = "{}_{}".format(str(camera_id), str(time_id).zfill(5))
            img_path = os.path.join(
                            path,
                            'rgb',
                            "{}x".format(str(scene_factor)),
                            frame_name + ".png",
                        )

            curr_img_rgba = Image.open(img_path)
            curr_img = np.array(curr_img_rgba.convert('RGB'))
            curr_img = Image.fromarray((curr_img).astype(np.uint8))

            image_name = Path(img_path).stem

            camera = (
                        dycheck_geometry.Camera.fromjson(
                            os.path.join(path, 'camera', frame_name + ".json")
                        )
                        .rescale_image_domain(1 / scene_factor)
                        .translate(-scene_center)
                        .rescale(coord_scale)
                        .rescale(1/mean_depth)
                    )
            K = camera.intrin
            R = camera.orientation.T
            T = camera.translation
            focal = camera.focal_length
            fid = time_id / max_time

            FovY = focal2fov(focal, curr_img.size[1])
            FovX = focal2fov(focal, curr_img.size[0])
            
            # covisible mask
            covisible = (imageio.imread(os.path.join(path, 'covisible/2x/val', f"{frame_name}.png")) > 0).astype(float)[..., None]           

            cam_info = CameraInfo(uid=int(image_name[2:]), R=R.astype(np.float32), T=T.astype(np.float32), FovY=float(FovY), FovX=float(FovX), image=curr_img, image_path=img_path, max_time=float(max_time),
                                image_name=image_name, width=curr_img.size[0], height=curr_img.size[1],
                                time=float(fid), mask=None, metadata=camera, normal=normal, depth=None, sem_mask=None, K=K,
                                instance_mask=None, covisible=covisible,
                                )
            test_cam_infos.append(cam_info)

    return train_cam_infos, test_cam_infos, max_time


def readIphoneInfo(path, args):
    print("Reading Iphone Info")
    
    train_cam_infos, test_cam_infos, max_time = readIphoneCameras(path, args.depth_std)
    nerf_normalization = getNerfppNorm(train_cam_infos)
    
    start_time = 0
    totalxyz = []
    totalrgb = []
    totaltime = []

    xyz = rgb = np.zeros((100, 3)) # dummy

    for i in range(start_time, start_time+max_time):        
        totalxyz.append(xyz)
        totalrgb.append(rgb)
        totaltime.append(np.ones((xyz.shape[0], 1)) * (i-start_time) / max_time)

    xyz = np.concatenate(totalxyz, axis=0)
    rgb = np.concatenate(totalrgb, axis=0)
    totaltime = np.concatenate(totaltime, axis=0)   

    ply_path = os.path.join(path, "points3D.ply")
    xyzt = np.concatenate((xyz, totaltime), axis=1)   
    assert xyz.shape[0] == rgb.shape[0]   
    storePly(ply_path, xyzt, rgb)    
    try:
        pcd = fetchPly(ply_path)
        
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=None,
                           nerf_normalization=nerf_normalization,
                           ply_path=None, maxtime=max_time)
    return scene_info

def read_flow(flow_path, img_size):
    flow_info = np.load(flow_path)
    flow, mask = flow_info['flow'], flow_info['mask']
    H,W,_ = flow.shape
    flow[...,0] = flow[...,0] / W
    flow[...,1] = flow[...,1] / H

    flow = cv2.resize(flow, (int(img_size[1]), int(img_size[0])), interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(mask.astype(np.float32), (int(img_size[1]), int(img_size[0])), interpolation=cv2.INTER_NEAREST)
    return flow, mask

def normalize_coords(coords, h, w):
    assert coords.shape[-1] == 2
    return coords / torch.tensor([w - 1.0, h - 1.0], device=coords.device) * 2 - 1.0

prior_list = [
    # [[(288, 70),(291, 248)],[(313,55),(315,235+ 4)],[(326,74),(328,251+ 4)],[(339,55),(336,230+ 4)],[(359,70),(349,248+ 4)],[(386,86),(379,263+ 4)],[(363,106),(344,261+ 4)],[(345,109),(357,256+ 4)],[(334,130),(347,267)],[(321,126),(334,254+ 4)],[(313,124),(325,247+ 4)],[(292,141),(305,263+ 4)]],
    [[(288, 70),(291, 243)],[(313,55),(315,230)],[(326,74),(328,246)],[(339,55),(336,225)],[(359,70),(349,243)],[(386,86),(379,258)],[(363,106),(344,256)],[(345,109),(357,251)],[(334,130),(347,262)],[(321,126),(334,249)],[(313,124),(325,242)],[(292,141),(305,257)]],
    [],
    []
]

def readNvidiaCameras(path):
    # load img factor of 2
    poses_arr = np.load(os.path.join(path, 'poses_bounds.npy'))
    scene_center = np.array(json.load(open(os.path.join(path, 'scene.json')))["center"])
    # poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
    # bds = poses_arr[:, -2:].transpose([1,0])
    factor = 2

    poses = poses_arr[:, :15].reshape(-1, 3, 5)
    hwf = poses[0, :, -1]
    sh = hwf[:2] / factor
    focal_length = hwf[-1] / factor
    cx = sh[1] / 2.0
    cy = sh[0] / 2.0
    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    max_time = poses.shape[0] - 1
    poses = np.concatenate(
        [poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
    bottoms = np.array([0, 0, 0, 1]).reshape(
        1, -1, 4).repeat(poses.shape[0], axis=0)
    poses = np.concatenate([poses, bottoms], axis=1)
    poses = poses @ np.diag([1, -1, -1, 1])
    c2ws = poses[:,:3,:4]

    train_cam_infos = []
    test_cam_infos = []
    normal_dir = os.path.join(path, 'normal')
    disp_dir = os.path.join(path, 'norm_disp')
    if not os.path.exists(normal_dir):
        os.mkdir(normal_dir)
    if not os.path.exists(disp_dir):
        os.mkdir(disp_dir)

    depth_list = []
    for idx in range(max_time + 1):
        frame_name = f"{idx:03d}.png"
        depth_path = os.path.join(path, 'metric3d_depth', frame_name.replace('.png', '.npz'))
        depth = np.load(depth_path)['dep'][..., None]
        depth_list.append(depth)
    mean_depth = np.mean(np.stack(depth_list, 0))

    for idx in range(max_time + 1):
        frame_name = f"{idx:03d}.png"
        img_path = os.path.join(path, 'images_2', frame_name)
        image_name = Path(img_path).stem

        curr_img = np.array(Image.open(img_path))
        curr_img = Image.fromarray((curr_img).astype(np.uint8))

        # load flow
        fwd_flow_path = os.path.join(path, 'flow', frame_name.replace('.png', '_fwd.npz'))
        bwd_flow_path = os.path.join(path, 'flow', frame_name.replace('.png', '_bwd.npz'))

        if idx == 0:
            fwd_flow, fwd_mask = read_flow(fwd_flow_path, sh)
            bwd_flow, bwd_mask = np.zeros_like(fwd_flow), np.zeros_like(fwd_mask)
        elif idx == max_time:
            bwd_flow, bwd_mask = read_flow(bwd_flow_path, sh)
            fwd_flow, fwd_mask = np.zeros_like(bwd_flow), np.zeros_like(bwd_mask)
        else:
            fwd_flow, fwd_mask = read_flow(fwd_flow_path, sh)
            bwd_flow, bwd_mask = read_flow(bwd_flow_path, sh)

        norm_disp_path = os.path.join(disp_dir, frame_name)
        if not os.path.exists(norm_disp_path):
            disp_path = os.path.join(path, 'disp', frame_name.replace('.png', '.npy'))
            disp = np.load(disp_path)

            disp_min = disp.min()
            disp_max = disp.max()

            max_val = (2 ** (8 * 2)) - 1

            if disp_max - disp_min > np.finfo("float").eps:
                norm_disp = max_val * (disp - disp_min) / (disp_max - disp_min)
            else:
                norm_disp = np.zeros(disp.shape, dtype=disp.dtype)
            cv2.imwrite(norm_disp_path, norm_disp.astype("uint16"), [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        norm_disp = Image.open(norm_disp_path)
        #norm_disp = torch.clamp(PILtoTorch(norm_disp, (int(sh[1]), int(sh[0]))).squeeze(0).unsqueeze(-1), 0, 255)
        # depth_from_disp = 255.0 - norm_disp.numpy()
        # depth_from_disp = cv2.resize(depth_from_disp, dsize=(int(sh[1]), int(sh[0])), interpolation=cv2.INTER_CUBIC)[..., None]
        
        disp_path = os.path.join(path, 'disp', frame_name.replace('.png', '.npy'))
        disp = np.load(disp_path)
        depth_from_disp = 1.0 / (disp + 1e-6)
        depth_from_disp = cv2.resize(depth_from_disp, dsize=(int(sh[1]), int(sh[0])), interpolation=cv2.INTER_CUBIC)[..., None]
        
        curr_c2w = c2ws[idx].copy()
        
        # translate and scale pose
        curr_c2w[:3, 3] -= scene_center
        curr_c2w[:3, 3] /= mean_depth
        
        C2W = np.eye(4).astype(np.float32)
        C2W[:3,:4] = curr_c2w
        W2C = np.linalg.inv(C2W)

        R = C2W[:3,:3]
        T = W2C[:3,3]
        fid = idx / max_time

        FovY = focal2fov(focal_length, curr_img.size[1])
        FovX = focal2fov(focal_length, curr_img.size[0])

        metadata =  dycheck_geometry.Camera(
            orientation=W2C[:3,:3],
            position=C2W[:3,3],
            focal_length=np.array([focal_length]).astype(np.float32),
            principal_point=np.array([sh[1]/2.0, sh[0]/2.0]).astype(np.float32),
            image_size=np.array([sh[1], sh[0]]).astype(np.float32),
        )

        depth_path = os.path.join(path, 'metric3d_depth', frame_name.replace('.png', '.npz'))
        uni_depth = np.load(depth_path)['dep'][..., None] / mean_depth

        
        depth = uni_depth
        depth_mask = (depth > 1e-3) * (depth < 1000)
        # cv2.imwrite("debug.png", (depth / depth.max() * 255).astype(np.uint8))

        normal_path = os.path.join(path, 'uni_normal', frame_name.replace('.png', '.npy'))
        if not os.path.exists(normal_path):
            normal = get_normals(torch.from_numpy(depth).squeeze(-1)[None], metadata)
            normal = normal.squeeze(0).cpu().numpy().transpose(1,2,0)
            np.save(normal_path, normal)
        else:
            normal = np.load(normal_path)
        if np.isnan(normal).any():
            breakpoint()
            
        # apply avg pooling to normal
        normal = F.avg_pool2d(torch.from_numpy(normal).permute(2,0,1)[None], 5, stride=1, padding=2).squeeze(0).permute(1,2,0).numpy()

        instance_path = os.path.join(path, 'instance_mask_manual', frame_name.split('.')[0] + '/*.png')
        instance_mask_list = []
        for mask_path in sorted(glob.glob(instance_path)):
            instance_mask = PILtoTorch(Image.open(mask_path), (int(sh[1]), int(sh[0]))).squeeze(0).unsqueeze(-1).numpy()
            instance_mask[instance_mask > 0] = 1
            instance_mask_list.append(instance_mask)
        instance_mask_list = np.stack(instance_mask_list, 0)

        new_mask = np.zeros_like(instance_mask_list[0])
        for instance in instance_mask_list:
            new_mask = np.maximum(new_mask, instance)
            
        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=curr_img, image_path=img_path, max_time=max_time,
                              image_name=image_name, width=curr_img.size[0], height=curr_img.size[1],
                              time=fid, mask=new_mask, metadata=metadata, normal=normal, depth=depth, sem_mask=None,
                              fwd_flow=fwd_flow, bwd_flow=bwd_flow, fwd_flow_mask=fwd_mask[...,None], bwd_flow_mask=bwd_mask[...,None],
                              instance_mask=instance_mask_list, depth_mask=depth_mask, K=K,
                              )
        train_cam_infos.append(cam_info)

    for idx in range(max_time + 1):
        frame_name = f"v000_t{idx:03d}.png"
        img_path = os.path.join(path, 'gt', frame_name)
        image_name = Path(img_path).stem

        curr_img = np.array(Image.open(img_path))
        curr_img = Image.fromarray((curr_img).astype(np.uint8))

        curr_c2w = c2ws[0].copy()
        
        # translate and scale pose
        curr_c2w[:3, 3] -= scene_center
        curr_c2w[:3, 3] /= mean_depth
        
        C2W = np.eye(4).astype(np.float32)
        C2W[:3,:4] = curr_c2w
        W2C = np.linalg.inv(C2W)

        R = C2W[:3,:3]
        T = W2C[:3,3]
        fid = idx / max_time

        FovY = focal2fov(focal_length, curr_img.size[1])
        FovX = focal2fov(focal_length, curr_img.size[0])

        metadata =  dycheck_geometry.Camera(
            orientation=W2C[:3,:3],
            position=C2W[:3,3],
            focal_length=np.array([focal_length]).astype(np.float32),
            principal_point=np.array([sh[1]/2.0, sh[0]/2.0]).astype(np.float32),
            image_size=np.array([sh[1], sh[0]]).astype(np.float32),
        )

        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=curr_img, image_path=img_path, max_time=max_time,
                              image_name=image_name, width=curr_img.size[0], height=curr_img.size[1],
                              time=fid, mask=None, metadata=metadata, normal=None, depth=None, sem_mask=None, K=K,)
        test_cam_infos.append(cam_info)
    return train_cam_infos, test_cam_infos, max_time

def readNvidiaInfo(path):
    print("Reading Nvidia Info")
    train_cam_infos, test_cam_infos, max_time = readNvidiaCameras(path)
    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")

    print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")

    totalxyz = []
    totalrgb = []
    totaltime = []
    
    start_time = 0

    try:
        xyz, rgb, _ = read_points3D_binary(bin_path)
    except:
        xyz, rgb, _ = read_points3D_text(txt_path)
    
    for i in range(start_time, start_time+max_time):        
        totalxyz.append(xyz)
        totalrgb.append(rgb)
        totaltime.append(np.ones((xyz.shape[0], 1)) * (i-start_time) / max_time)
    
    xyz = np.concatenate(totalxyz, axis=0)
    rgb = np.concatenate(totalrgb, axis=0)
    totaltime = np.concatenate(totaltime, axis=0)    
    assert xyz.shape[0] == rgb.shape[0]
    xyzt = np.concatenate((xyz, totaltime), axis=1)     
    storePly(ply_path, xyzt, rgb)    
    try:
        pcd = fetchPly(ply_path)
        
    except:
        pcd = None
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=None,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path, maxtime=max_time)
    return scene_info

sceneLoadTypeCallbacks = {
    # scene.Scene dispatches on the canonical dataset type; scene/__init__.py adds
    # the "davis" entry, which lives in its own module to keep the import one-way.
    "Colmap": readColmapSceneInfo,
    "iphone": readIphoneInfo,
    "nvidia": readNvidiaInfo,
}
