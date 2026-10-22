import os
import sys
import re
import copy
import torch
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from scipy.spatial.transform import Rotation, Slerp

sys.path.append(os.path.join(sys.path[0], ".."))

from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from gaussian_renderer import render
from scene import bundle_adjustment


torch.manual_seed(0)


def save_tensor_image(tensor_chw, save_path):
    img = torch.clamp(tensor_chw, 0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy()
    img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(img).save(save_path)


def _extract_int_tokens(name):
    if name is None:
        return []
    return [int(tok) for tok in re.findall(r"\d+", str(name))]


def _build_train_camera_maps(train_cams):
    by_name = {}
    by_uid = {}
    by_last_token = {}
    by_first_token = {}

    for cam in train_cams:
        name = str(getattr(cam, "image_name", ""))
        uid = getattr(cam, "uid", None)

        if name and name not in by_name:
            by_name[name] = cam

        if uid is not None:
            by_uid[int(uid)] = cam

        toks = _extract_int_tokens(name)
        if toks:
            by_last_token.setdefault(toks[-1], cam)
            by_first_token.setdefault(toks[0], cam)

    return by_name, by_uid, by_last_token, by_first_token


def _match_train_camera(test_cam, maps):
    by_name, by_uid, by_last_token, by_first_token = maps
    t_name = str(getattr(test_cam, "image_name", ""))
    t_uid = getattr(test_cam, "uid", None)

    if t_name in by_name:
        return by_name[t_name]

    if t_uid is not None and int(t_uid) in by_uid:
        return by_uid[int(t_uid)]

    toks = _extract_int_tokens(t_name)
    if toks and toks[-1] in by_last_token:
        return by_last_token[toks[-1]]

    if toks and toks[0] in by_first_token:
        return by_first_token[toks[0]]

    return None


def _interp_pose_train_to_test(train_cam, test_cam, alpha=0.2):
    r_train = np.asarray(train_cam.R, dtype=np.float32)
    r_test = np.asarray(test_cam.R, dtype=np.float32)
    t_train = np.asarray(train_cam.T, dtype=np.float32)
    t_test = np.asarray(test_cam.T, dtype=np.float32)

    key_rots = Rotation.from_matrix(np.stack([r_train, r_test], axis=0))
    slerp = Slerp([0.0, 1.0], key_rots)
    r_interp = slerp([alpha]).as_matrix()[0].astype(np.float32)
    t_interp = ((1.0 - alpha) * t_train + alpha * t_test).astype(np.float32)

    return r_interp, t_interp


def build_interpolated_cameras(train_cams, test_cams, alpha=0.2):
    maps = _build_train_camera_maps(train_cams)
    interp_cams = []
    unmatched = 0

    for test_cam in test_cams:
        train_cam = _match_train_camera(test_cam, maps)
        if train_cam is None:
            unmatched += 1
            continue

        interp_cam = copy.deepcopy(test_cam)
        interp_r, interp_t = _interp_pose_train_to_test(train_cam, test_cam, alpha=alpha)

        pixels = interp_cam.get_pixels(
            interp_cam.metadata.image_size_x,
            interp_cam.metadata.image_size_y,
            use_center=True,
        )
        batch_shape = pixels.shape[:-1]
        local_viewdirs = interp_cam.pixels_to_local_viewdirs(np.reshape(pixels, (-1, 2))).astype(np.float32)
        interp_cam.update_cam(interp_r, interp_t, local_viewdirs, batch_shape)
        interp_cams.append(interp_cam)

    if unmatched > 0:
        print(f"[WARN] Could not match {unmatched} test cameras to train cameras.")

    print(f"[*] Built {len(interp_cams)} interpolated cameras (alpha={alpha}).")
    return interp_cams


def build_multi_step_interpolated_groups(train_cams, test_cams):
    groups = {}
    for k in [1, 2, 3, 4]:
        alpha = k / 5.0
        split_name = f"interp_{k}_5"
        groups[split_name] = build_interpolated_cameras(train_cams, test_cams, alpha=alpha)
    return groups


def training_report(scene, train_cams, test_cams, interp_cam_groups, render_func, render_args, stage, out_dir):
    validation_configs = [
        {"name": "test", "cameras": test_cams},
        {"name": "train", "cameras": train_cams},
    ]
    for split_name, cams in interp_cam_groups.items():
        validation_configs.append({"name": split_name, "cameras": cams})

    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for config in validation_configs:
            cams = config["cameras"]
            if not cams or len(cams) == 0:
                continue

            split_dir = os.path.join(out_dir, config["name"])
            os.makedirs(split_dir, exist_ok=True)

            for idx, viewpoint in enumerate(cams):
                render_pkg = render_func(
                    viewpoint,
                    scene.stat_gaussians,
                    scene.dyn_gaussians,
                    render_args[1],
                    stage=stage,
                    get_static=True,
                )

                save_tensor_image(
                    render_pkg["render"],
                    os.path.join(split_dir, f"img_{idx:04d}.png"),
                )

                if "s_render" in render_pkg:
                    save_tensor_image(
                        render_pkg["s_render"],
                        os.path.join(split_dir, f"s_img_{idx:04d}.png"),
                    )

    torch.cuda.empty_cache()
    print(f"Saved rendered images only to: {out_dir}")


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = ArgumentParser(description="NVS image dump (no metrics, no TTO)")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)

    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument("--stage", type=str, default="fine")
    parser.add_argument("--run_ba", action="store_true")

    args = parser.parse_args(sys.argv[1:])

    if args.configs:
        import mmengine as mmcv
        from utils.params_utils import merge_hparams

        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    dataset = lp.extract(args)
    hyper = hp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    dyn_gaussians = GaussianModel(dataset.sh_degree, hyper)
    stat_gaussians = GaussianModel(dataset.sh_degree, hyper)

    scene = Scene(dataset, dyn_gaussians, stat_gaussians, load_coarse=None)

    dyn_gaussians.create_pose_network(hyper, scene.getTrainCameras())
    dyn_gaussians.training_setup(opt, stage="fine")
    stat_gaussians.training_setup(opt, stage="fine", is_static=True)

    if args.run_ba:
        bundle_adjustment(opt, scene, dataset.source_path)

    bg_color = [1] * 9 + [0] if dataset.white_background else [0] * 9 + [0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    test_cams = [c for c in scene.getTestCameras()]
    train_cams = [c for c in scene.getTrainCameras()]

    dyn_gaussians.load_ply(os.path.join(args.checkpoint, "point_cloud.ply"))
    stat_gaussians.load_ply(os.path.join(args.checkpoint, "point_cloud_static.ply"))
    dyn_gaussians.load_model(args.checkpoint)

    interp_cam_groups = build_multi_step_interpolated_groups(train_cams, test_cams)

    output_dir = os.path.join("output", args.expname, "nvs_images")
    training_report(
        scene=scene,
        train_cams=train_cams,
        test_cams=test_cams,
        interp_cam_groups=interp_cam_groups,
        render_func=render,
        render_args=[pipe, background],
        stage=args.stage,
        out_dir=output_dir,
    )
