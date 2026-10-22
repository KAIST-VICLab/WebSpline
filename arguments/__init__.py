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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = True
        self.add_points=False
        self.extension=".png"
        self.llffhold=8
        self.depth_std = None
        # Empty = sniff the type from marker files in the source directory
        # (see scene.detect_dataset_type); set it to override that.
        self.dataset_type = ""
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class ModelHiddenParams(ParamGroup):
    def __init__(self, parser):
        self.net_width = 64
        self.timebase_pe = 4
        self.defor_depth = 1
        self.posebase_pe = 10
        self.scale_rotation_pe = 2
        self.opacity_pe = 2
        self.timenet_width = 64
        self.timenet_output = 32
        self.bounds = 1.6
        self.plane_tv_weight = 0.0001
        self.time_smoothness_weight = 0.01
        self.l1_time_planes = 0.0001
        self.kplanes_config = {
                             'grid_dimensions': 2,
                             'input_coordinate_dim': 4,
                             'output_coordinate_dim': 32,
                             'resolution': [64, 64, 64, 25]
                            }
        self.multires = [1, 2, 4, 8]
        self.no_dx=False
        self.no_grid=False
        self.no_ds=False
        self.no_dr=False
        self.no_do=True
        self.no_dshs=True
        self.empty_voxel=False
        self.grid_pe=0
        self.static_mlp=False
        self.apply_rotation=False

        super().__init__(parser, "ModelHiddenParams")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.dataloader=False
        self.zerostamp_init=False
        self.custom_sampler=None
        self.iterations = 30_000
        self.coarse_iterations = 1000
        self.fine_static_iterations = 30000    

        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 20_000

        self.deformation_lr_init = 0.00016
        self.deformation_lr_final = 0.000016
        self.deformation_lr_delay_mult = 0.01

        self.grid_lr_init = 0.0016
        self.grid_lr_final = 0.00016

        self.pose_lr_init = 0.0005
        self.pose_lr_final = 0.00005
        self.pose_lr_delay_mult = 0.01

        self.feature_lr = 0.0025
        self.featuret_lr = 0.001
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.max_scale = 0.02
        self.min_scale = 0.0
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.p_lambda_dssim = 0.0
        self.lambda_lpips = 0
        self.lambda_direct_anchor = 10.0
        self.lambda_scf_spline_prior = 0.05
        self.scf_spline_vis_window = 30
        self.weight_constraint_init= 1
        self.weight_constraint_after = 0.2
        self.weight_decay_iteration = 5_000
        self.opacity_reset_interval = 3_000
        self.densification_interval = 4_00
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold_coarse = 0.0002
        self.densify_grad_threshold_fine_init = 0.0002
        self.densify_grad_threshold_after = 0.0002
        self.pruning_from_iter = 500
        self.pruning_interval = 100
        self.opacity_threshold_coarse = 0.005
        self.opacity_threshold_fine_init = 0.005
        self.opacity_threshold_fine_after = 0.005
        self.batch_size=1
        self.add_point=False

        self.prevpath = "1"
        self.opthr = 0.005
        self.desicnt = 6
        self.densify_grad_threshold = 0.0002
        self.preprocesspoints = 0  
        self.addsphpointsscale = 0.8  
        self.raystart = 0.7

        self.soft_depth_start = 1000
        self.hard_depth_start = 0
        self.error_tolerance = 0.001

        self.trbfc_lr = 0.0001 # 
        self.trbfs_lr = 0.03
        self.trbfslinit = 0.0 #         
        self.omega_lr = 0.0001
        self.zeta_lr = 0.0001
        self.movelr = 3.5
        self.rgb_lr = 0.0001

        self.stat_npts = 40000
        self.dyn_npts = 20000

        # Scene-geometry knobs needed by sequences where the moving object fills
        # the frame or the camera barely translates (DAVIS).  The defaults are
        # inert, so datasets that do not need them are unaffected.
        #
        # Cap on dynamic initialisation points (0 = uncapped).
        self.max_dyn_init_pts = 0
        # Placeholder static points seeded when a frame has no static pixels at
        # all (0 = seed none and let the static model stay empty).
        self.static_placeholder_total = 1000
        # Absolute scene extent for the densify/prune thresholds
        # (0 = use the extent derived from the camera positions).
        self.densify_extent = 0.0
        # Derive that extent from the initial point cloud radius instead.
        self.densify_extent_from_pcd = False

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
