import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import open3d as o3d
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from random import randint
from utils.sh_utils import RGB2SH, SH2RGB
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud, pts2pixel
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.deformation import deform_network, pose_network
from scene.regulation import compute_plane_smoothness
from helper_model import getcolormodel
from utils.spline_utils import interpolate_cubic_hermite


def build_hermite_A(times_1d: torch.Tensor, N_pts: int):
    """
    times_1d: (T,) in [0,1]
    return: A (T, N_pts)
    """
    device = times_1d.device
    T = times_1d.shape[0]
    N = N_pts

    A = torch.zeros((T, N), device=device, dtype=times_1d.dtype)

    times_scaled = times_1d * (N - 1)
    indices = torch.floor(times_scaled).long().clamp(0, N - 2)

    left  = (indices - 1).clamp(0, N - 1)
    right = (indices + 1).clamp(0, N - 1)
    rr    = (indices + 2).clamp(0, N - 1)

    t = (times_scaled - indices.float())

    h00 = (1 + 2*t) * (1 - t)**2
    h10 = t * (1 - t)**2
    h01 = t**2 * (3 - 2*t)
    h11 = t**2 * (t - 1)

    p1 = h00
    p0 = torch.zeros_like(h00)
    p2 = h01
    p3 = torch.zeros_like(h00)

    h10_add_p0 = torch.where(left == indices, torch.zeros_like(h10), -h10/2)
    h10_add_p1 = torch.where(left == indices, -h10, torch.zeros_like(h10))
    h10_add_p2 = torch.where(left == indices,  h10,  h10/2)

    h11_add_p1 = torch.where(rr == right, -h11, -h11/2)
    h11_add_p2 = torch.where(rr == right,  h11, torch.zeros_like(h11))
    h11_add_p3 = torch.where(rr == right, torch.zeros_like(h11),  h11/2)

    p0 = p0 + h10_add_p0
    p1 = p1 + h10_add_p1 + h11_add_p1
    p2 = p2 + h10_add_p2 + h11_add_p2
    p3 = p3 + h11_add_p3

    # Accumulate into A[row, col] (scatter_add style)
    rows = torch.arange(T, device=device)

    A.index_put_((rows, left),  p0, accumulate=True)
    A.index_put_((rows, indices), p1, accumulate=True)
    A.index_put_((rows, right), p2, accumulate=True)
    A.index_put_((rows, rr),    p3, accumulate=True)

    return A

def inverse_cubic_hermite(curves: torch.Tensor, times_1d: torch.Tensor, N_pts=5, reg=1e-6):
    """
    curves: (B, T, C), with C=3 here
    times_1d: (T,) in [0, 1], shared across the batch
    return: control_pts (B, N_pts, C)
    """
    # A: (T, N)
    A = build_hermite_A(times_1d, N_pts)          # (T, N)
    At = A.transpose(0, 1)                        # (N, T)
    AtA = At @ A                                  # (N, N)
    AtA = AtA + reg * torch.eye(N_pts, device=A.device, dtype=A.dtype)

    # M = (AᵀA)^-1 Aᵀ  -> (N, T)
    M = torch.linalg.solve(AtA, At)               # (N, T)

    # control = M @ curves
    # M: (N,T), curves: (B,T,C) -> (B,N,C)
    control_pts = torch.einsum('nt,btc->bnc', M, curves)
    return control_pts

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree : int, args, rgbfuntion="sandwich"):

        self.active_sh_degree = 0
        temporal_res = args.kplanes_config['resolution'][-1]
        self.control_num = temporal_res // 4 if temporal_res > 12 else temporal_res
        self.error_threshold = 1.0
        self.current_control_num = torch.tensor(self.control_num, device="cuda").repeat(1)
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self.control_xyz = torch.empty(0)
        self._deformation = deform_network(args)
        self._posenet = None
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self._deformation_table = torch.empty(0)
        self.setup_functions()

        self._motion = torch.empty(0)
        self._omega = torch.empty(0)
        self._zeta = torch.empty(0)
        self.initial_control_xyz = torch.empty(0)
        self.parent_node_idx = torch.empty(0)
        self.parent_local_knn_nodes = torch.empty(0)
        self.parent_mlevel_knn_nodes = []
        self.delta_t = None
        self.omegamask = None
        self.maskforems = None
        self.distancetocamera = None
        self.trbfslinit = None
        self.ts = None
        self.trbfoutput = None
        self.preprocesspoints = False
        self.addsphpointsscale = 0.8

        self.maxz, self.minz =  0.0 , 0.0
        self.maxy, self.miny =  0.0 , 0.0
        self.maxx, self.minx =  0.0 , 0.0
        self.raystart = 0.7
        self.computedtrbfscale = None
        self.computedopacity = None
        self.computedscales = None
        self.max_scale = 0.02
        self.min_scale = 0.0

        self.rgbdecoder = getcolormodel(rgbfuntion)

    def create_pose_network(self, args, train_cams):
        self._posenet = pose_network(args, train_cams=train_cams).to("cuda")

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self.control_xyz,
            self.current_control_num,
            self.parent_node_idx,
            self._deformation.state_dict(),
            self._posenet.state_dict(),
            self._deformation_table,
            # self.grid,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
         self._xyz,
         self.control_xyz,
         self.current_control_num,
         self.parent_node_idx,
         deform_state,
         self._deformation_table,

         # self.grid,
         self._features_dc,
         self._features_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         xyz_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale) = model_args
        self._deformation.load_state_dict(deform_state)
        self._posenet.load_state_dict(deform_state)
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self._clamp_physical_scales(self.scaling_activation(self._scaling))

    def _get_scale_bounds(self, reference_tensor, for_log=False):
        min_scale = torch.as_tensor(self.min_scale, dtype=reference_tensor.dtype, device=reference_tensor.device)
        max_scale = torch.as_tensor(self.max_scale, dtype=reference_tensor.dtype, device=reference_tensor.device)
        if for_log:
            eps = max(torch.finfo(reference_tensor.dtype).eps, 1e-12)
            min_scale = torch.clamp(min_scale, min=eps)
        max_scale = torch.maximum(max_scale, min_scale)
        return min_scale, max_scale

    def _clamp_physical_scales(self, scales):
        if scales.numel() == 0:
            return scales
        min_scale, max_scale = self._get_scale_bounds(scales, for_log=False)
        return torch.clamp(scales, min=min_scale, max=max_scale)

    def _physical_scales_to_logits(self, scales):
        if scales.numel() == 0:
            return scales
        min_scale, max_scale = self._get_scale_bounds(scales, for_log=True)
        clamped_scales = torch.clamp(scales, min=min_scale, max=max_scale)
        return self.scaling_inverse_activation(clamped_scales)

    def _clamp_scale_logits(self, scale_logits):
        if scale_logits.numel() == 0:
            return scale_logits
        scales = self.scaling_activation(scale_logits)
        scales = self._clamp_physical_scales(scales)
        return self._physical_scales_to_logits(scales)

    def apply_scale_clamp(self):
        if self._scaling.numel() == 0:
            return
        with torch.no_grad():
            self._scaling.data.copy_(self._clamp_scale_logits(self._scaling.data))

    @property
    def get_rotation_stat(self):
        return self.rotation_activation(self._rotation)

    def get_rotation(self, delta_t=None):
        rotation =  self._rotation + delta_t*self._omega
        self.delta_t = delta_t
        return self.rotation_activation(rotation)

    def get_rotation_dy(self, rotation, delta_t):
        new_rotation =  rotation + delta_t*self._omega
        return new_rotation

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_control_xyz(self):
        return self.control_xyz

    @property
    def get_trbfcenter(self):
        return self._trbf_center
    @property
    def get_trbfscale(self):
        return self._trbf_scale

    def get_features(self, deltat):
        return torch.cat((self._features_dc, deltat * self._features_t), dim=1)

    @property
    def get_features_static(self):
        return torch.cat((self._features_dc, 0.0 * self._features_t), dim=1)

    @property
    def get_blending(self):
        return self._features_rest[:, -1:, 0]

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def flatten_control_point(self):
        flat_control_point = []
        for i in range(self.control_xyz.shape[0]):
            current_control_xyz = self.control_xyz[i][:self.current_control_num.squeeze()[i]]
            flat_control_point.append(current_control_xyz)
        self.flat_control_xyz = torch.cat(flat_control_point, dim=0).contiguous()
        self.index_offset = (torch.cat([torch.zeros(1).cuda(),torch.cumsum(self.current_control_num.squeeze()[:-1], dim=0)], dim=0).long())[:,None]

    def add_dummy_control_point(self):
        self.dummy_control_xyz = torch.cat([self.control_xyz, torch.zeros(self.control_xyz.shape[0], 100, 3).cuda()], dim=1)

    def onedown_control_pts(self, viewpoints):
        dummy_step = torch.arange(0, self.control_num, 1).cuda().float()[None].repeat(self.control_xyz.shape[0], 1)
        time_step = 1 / (self.current_control_num.squeeze(-1) - 1.0)
        t_step = (dummy_step * time_step[...,None])[...,None] # t_step corresponding to the current control points
        new_control_num = self.current_control_num - 1
        new_control_num[new_control_num < 4] = 4
        new_control_pts_value = self.inverse_cubic_hermite_for_prune(self.control_xyz, t_step, N_pts=new_control_num) # reduce 1 control point
        new_control_pts = self.control_xyz.clone()
        new_control_pts[:,:self.control_num-1] = new_control_pts_value


        # update current control number and control points: using 2d projection?
        error = self.compute_prune_error(new_control_pts, new_control_num, viewpoints)
        error_threshold = self.error_threshold
        self.current_control_num[error <= error_threshold] = new_control_num[error <= error_threshold]
        self.control_xyz[error <= error_threshold] = new_control_pts[error <= error_threshold]
        print("One down control points: ", (error <= error_threshold).sum().cpu().numpy())

    def compute_prune_error(self, new_control_pts, new_control_num, viewpoints):
        K = torch.zeros(3, 3).type_as(self.control_xyz)
        K[0, 0] = float(self._posenet.focal_bias.exp())
        K[1, 1] = float(self._posenet.focal_bias.exp())
        K[0, 2] = float(viewpoints[0].image_width / 2)
        K[1, 2] = float(viewpoints[0].image_height / 2)
        K[2, 2] = float(1)
        pix_err_list = []
        for idx, viewpoint in enumerate(viewpoints):
            if idx == 0 or idx == len(viewpoints) - 1:
                continue # skip first and last frame
            deform_means3D = interpolate_cubic_hermite(self.control_xyz.permute(0,2,1), torch.tensor(viewpoint.time).cuda()[None, None].expand(self.control_xyz.shape[0], 3, 1), N=self.current_control_num) * 1e-2
            new_deform_means3D = interpolate_cubic_hermite(new_control_pts.permute(0,2,1), torch.tensor(viewpoint.time).cuda()[None, None].expand(new_control_pts.shape[0], 3, 1), N=new_control_num) * 1e-2
            deform_means2D = pts2pixel(deform_means3D, viewpoint, K)
            new_deform_means2D = pts2pixel(new_deform_means3D, viewpoint, K)
            pix_err_list.append(torch.norm(deform_means2D - new_deform_means2D, dim=-1))
        return torch.stack(pix_err_list, dim=0).mean(0)

    def inverse_cubic_hermite_for_prune(self, curves, times, N_pts):
        transform_matrix = torch.zeros((times.shape[0], self.control_num, self.control_num - 1), device=curves.device) # B, T, N_pts always maximmum entries
        dummy_transform_matrix = torch.diag(torch.ones(self.control_num -1, device=curves.device), -1)[None].repeat(times.shape[0], 1, 1)[:, :, :-1]
        # dummy_eq = torch.zeros(self.control_num, device=curves.device)
        # dummy_eq[-1] = 1
        N = N_pts

        times_scaled = times * (N - 1)[:,None]
        indices = torch.floor(times_scaled).long()
        # Clamping to avoid out-of-bounds indices
        indices = torch.clamp(indices, torch.zeros_like(N)[:,None].expand(-1, self.control_num, -1), (N - 2)[:,None].expand(-1, self.control_num, -1)).long()
        left_indices = torch.clamp(indices - 1, torch.zeros_like(N)[:,None].expand(-1, self.control_num, -1), (N - 1)[:,None].expand(-1, self.control_num, -1)).long()
        right_indices = torch.clamp(indices + 1, torch.zeros_like(N)[:,None].expand(-1, self.control_num, -1), (N - 1)[:,None].expand(-1, self.control_num, -1)).long()
        right_right_indices = torch.clamp(indices + 2, torch.zeros_like(N)[:,None].expand(-1, self.control_num, -1), (N - 1)[:,None].expand(-1, self.control_num, -1)).long()

        t = (times_scaled - indices.float())
        # Hermite basis functions
        h00 = (1 + 2*t) * (1 - t)**2
        h10 = t * (1 - t)**2
        h01 = t**2 * (3 - 2*t)
        h11 = t**2 * (t - 1)

        p1_coef = h00 # B, T, 1
        p0_coef = torch.zeros_like(h00)
        p2_coef = h01
        p3_coef = torch.zeros_like(h00)

        # One-sided derivatives at the boundaries
        h10_add_p0 = torch.where(left_indices == indices, 0, -h10/2)
        h10_add_p1 = torch.where(left_indices == indices, -h10, 0)
        h10_add_p2 = torch.where(left_indices == indices, h10, h10/2)

        h11_add_p1 = torch.where(right_right_indices == right_indices, -h11, -h11/2)
        h11_add_p2 = torch.where(right_right_indices == right_indices, h11, 0)
        h11_add_p3 = torch.where(right_right_indices == right_indices, 0, h11/2)

        p0_coef = p0_coef + h10_add_p0
        p1_coef = p1_coef + h10_add_p1 + h11_add_p1
        p2_coef = p2_coef + h10_add_p2 + h11_add_p2
        p3_coef = p3_coef + h11_add_p3

        # scatter to transform matrix
        transform_matrix = torch.scatter(input=transform_matrix, dim=-1, index=left_indices, src=p0_coef, reduce='add')
        transform_matrix = torch.scatter(input=transform_matrix, dim=-1, index=indices, src=p1_coef, reduce='add')
        transform_matrix = torch.scatter(input=transform_matrix, dim=-1, index=right_indices, src=p2_coef, reduce='add')
        transform_matrix = torch.scatter(input=transform_matrix, dim=-1, index=right_right_indices, src=p3_coef, reduce='add')

        valid_mask = torch.ones((times.shape[0], self.control_num), device=curves.device)
        mask_index = self.current_control_num.squeeze() < self.control_num
        valid_mask[mask_index] = torch.scatter(input=valid_mask[mask_index], dim=-1, index=(self.current_control_num)[mask_index], src=torch.zeros_like(self.current_control_num).float()[mask_index])
        accum_valid_mask = torch.cumprod(valid_mask, dim=-1)

        mask_curves = curves * accum_valid_mask[...,None]
        mask_transform_matrix = transform_matrix * accum_valid_mask[...,None] + dummy_transform_matrix * (1 - accum_valid_mask[...,None])

        control_pts = torch.linalg.lstsq(mask_transform_matrix, mask_curves).solution
        # error = torch.square(control_pts - torch.linalg.pinv(mask_transform_matrix_2) @ curves).sum(-1).mean(-1)
        return control_pts

    def create_from_pcd_dynamic(self, pcd: BasicPointCloud, spatial_lr_scale: float, time_line: int, dyn_tracjectory: torch.tensor):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()

        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())

        times = torch.tensor(np.asarray(pcd.times)).float().cuda()

        features = torch.zeros(
            (fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2 + 1)).float().cuda()  # NOTE: +1 for blending factor
        features[:, :3, 0] = fused_color

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        time_step = 1 / (dyn_tracjectory.shape[1] - 1.0)
        t_step = torch.arange(0, 1 + time_step, time_step).cuda().float()[:dyn_tracjectory.shape[1]]
        t_step = t_step[None, :, None].expand(dyn_tracjectory.shape[0], -1, -1)
        init_control_pts = inverse_cubic_hermite(dyn_tracjectory * 1e2, t_step[0, :, 0], N_pts=self.control_num)
        self.control_xyz = nn.Parameter(init_control_pts.contiguous().requires_grad_(True))
        self.current_control_num = torch.tensor(self.control_num, device="cuda").repeat(fused_point_cloud.shape[0])[...,None]

        self.parent_node_idx = torch.arange(self.control_xyz.shape[0], device="cuda")[:, None]

        self._deformation = self._deformation.to("cuda")
        self.initial_control_xyz = init_control_pts.clone().detach().cuda()
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))

        features9channel = torch.cat((fused_color, fused_color), dim=1)
        self._features_dc = nn.Parameter(features9channel.contiguous().requires_grad_(True))
        N, _ = fused_color.shape

        fomega = torch.zeros((N, 3), dtype=torch.float, device="cuda")
        self._features_t =  nn.Parameter(fomega.contiguous().requires_grad_(True))

        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._deformation_table = torch.gt(torch.ones((self.get_xyz.shape[0]), device="cuda"), 0)

        omega = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        self._omega = nn.Parameter(omega.requires_grad_(True))

        motion = torch.zeros((fused_point_cloud.shape[0], 9), device="cuda")# x1, x2, x3,  y1,y2,y3, z1,z2,z3
        self._motion = nn.Parameter(motion.requires_grad_(True))

        zeta = torch.zeros((fused_point_cloud.shape[0], 1), device="cuda")
        self._zeta = nn.Parameter(zeta.requires_grad_(True))

        self._trbf_center = nn.Parameter(times.contiguous().requires_grad_(True))
        self._trbf_scale = nn.Parameter(torch.ones((self.get_xyz.shape[0], 1), device="cuda").requires_grad_(True))

        ## store gradients
        if self.trbfslinit is not None:
            nn.init.constant_(self._trbf_scale, self.trbfslinit) # too large ?
        else:
            nn.init.constant_(self._trbf_scale, 0) # too large ?

        nn.init.constant_(self._omega, 0)
        nn.init.constant_(self._zeta, 0)

        self.rgb_grd = {}

        self.maxz, self.minz = torch.amax(self._xyz[:,2]), torch.amin(self._xyz[:,2])
        self.maxy, self.miny = torch.amax(self._xyz[:,1]), torch.amin(self._xyz[:,1])
        self.maxx, self.minx = torch.amax(self._xyz[:,0]), torch.amin(self._xyz[:,0])
        self.maxz = min((self.maxz, 200.0)) # some outliers in the n4d datasets..

        for name, W in self.rgbdecoder.named_parameters():
            if 'weight' in name:
                self.rgb_grd[name] = torch.zeros_like(W, requires_grad=False).cuda() #self.rgb_grd[name] + W.grad.clone()
            elif 'bias' in name:
                print('not implemented')
                quit()

    @torch.no_grad()
    def update_parent_topology(self, scaffold):
        """
        Store, for every gaussian (control point), the node indices that its
        parent node is connected to in the scaffold, both by K-NN and across
        the multi-level topology.
        """
        device = self.control_xyz.device
        M = scaffold.M
        parent_idx = self.parent_node_idx.squeeze() # [N]
        
        # 1. Local Topology Node Indices
        self.parent_local_knn_nodes = scaffold.topo_knn_ind[parent_idx] # [N, K_local]
        
        # 2. Multi-level Topology Node Indices
        self.parent_mlevel_knn_nodes = []
        if getattr(scaffold, 'mlevel_arap_flag', False):
            for l_idx, edges in enumerate(scaffold.multilevel_arap_edge_list):
                src_nodes = edges[:, 0]
                dst_nodes = edges[:, 1]
                
                unique_src, counts = torch.unique(src_nodes, return_counts=True)
                k_macro = counts.max().item() if len(counts) > 0 else 0
                
                # Allocate [M, K_macro], defaulting to M
                macro_knn_for_nodes = torch.arange(M, device=device).unsqueeze(1).expand(M, k_macro).clone()
                
                if k_macro > 0:
                    for u_src in unique_src:
                        u_dsts = dst_nodes[src_nodes == u_src]
                        macro_knn_for_nodes[u_src, :len(u_dsts)] = u_dsts
                
                # Expand to the number of gaussians, N
                self.parent_mlevel_knn_nodes.append(macro_knn_for_nodes[parent_idx])
                
    @torch.no_grad()
    def build_topology_from_scaffold(self, scaffold):
        """
        Map the scaffold's local K-NN and multi-level edge structure onto the
        gaussians (control points).
        """
        device = self.control_xyz.device
        N = self.control_xyz.shape[0]
        M = scaffold.M
        parent_idx = self.parent_node_idx.squeeze() # [N]
        
        # (1) Node id -> gaussian id, picking one representative gaussian per node at random
        rand_perm = torch.randperm(N, device=device)
        shuffled_parents = parent_idx[rand_perm]
        
        node_to_gs = torch.full((M,), -1, dtype=torch.long, device=device)
        node_to_gs.scatter_(0, shuffled_parents, rand_perm)
        
        # Empty nodes fall back to gaussian 0, so their contribution to the loss is zero
        node_to_gs = torch.where(node_to_gs == -1, torch.zeros_like(node_to_gs), node_to_gs)
        
        # (2) Local topology: copy the scaffold's K-NN onto the gaussians
        # scaffold.topo_knn_ind is [M, K_local]; map it through each gaussian's parent to [N, K_local]
        self.gs_local_knn = node_to_gs[scaffold.topo_knn_ind[parent_idx]] 
        
        # (3) Multi-level topology, one mapping per level
        self.gs_mlevel_knn_list = []
        if getattr(scaffold, 'mlevel_arap_flag', False):
            for l_idx, edges in enumerate(scaffold.multilevel_arap_edge_list):
                # edges: [E, 2] pairs of (src_node, dst_node)
                src_nodes = edges[:, 0]
                dst_nodes = edges[:, 1]
                
                unique_src, counts = torch.unique(src_nodes, return_counts=True)
                k_macro = counts.max().item() if len(counts) > 0 else 0
                
                # Allocate [M, K_macro], defaulting to the node itself
                macro_knn_for_nodes = torch.arange(M, device=device).unsqueeze(1).expand(M, k_macro).clone()
                
                if k_macro > 0:
                    for u_src in unique_src:
                        u_dsts = dst_nodes[src_nodes == u_src]
                        macro_knn_for_nodes[u_src, :len(u_dsts)] = u_dsts
                
                # Map the [M, K_macro] node indices to [N, K_macro] gaussian indices
                # Nodes without macro edges point at themselves, giving d_rel = 0 and no loss contribution
                gs_macro_knn = node_to_gs[macro_knn_for_nodes[parent_idx]]
                self.gs_mlevel_knn_list.append(gs_macro_knn)

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float, time_line: int):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()

        # mean_pc = torch.mean(fused_point_cloud, dim=1, keepdim=True)
        # std_pc = 0*torch.std(fused_point_cloud, dim=1, keepdim=True) + 10
        # fused_point_cloud = torch.normal(mean=mean_pc, std=std_pc.expand(fused_point_cloud.shape[0], 3)).cuda()

        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        # fused_color = RGB2SH(torch.randn(fused_point_cloud.shape[0], 3).cuda())

        times = torch.tensor(np.asarray(pcd.times)).float().cuda()

        features = torch.zeros(
            (fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2 + 1)).float().cuda()  # NOTE: +1 for blending factor
        features[:, :3, 0] = fused_color

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        mean_x , std_x = torch.mean(fused_point_cloud[...,0], dim=0), torch.std(fused_point_cloud[...,0], dim=0)
        mean_y , std_y = torch.mean(fused_point_cloud[...,1], dim=0), torch.std(fused_point_cloud[...,1], dim=0)
        mean_z , std_z = torch.mean(fused_point_cloud[...,2], dim=0), torch.std(fused_point_cloud[...,2], dim=0)
        std_xyz = torch.tensor([std_x, std_y, std_z], device="cuda")
        mean_xyz = torch.tensor([mean_x, mean_y, mean_z], device="cuda")
        self.control_xyz = nn.Parameter((torch.randn(fused_color.shape[0], self.control_num, 3, device="cuda").requires_grad_(True) * std_xyz[None, None] + mean_xyz[None, None]))
        self.current_control_num = torch.tensor(self.control_num, device="cuda").repeat(fused_color.shape[0], 1)
        
        self.parent_node_idx = torch.arange(self.control_xyz.shape[0], device="cuda")[:, None]
        
        # self.control_xyz = nn.Parameter((torch.randn(self.control_num, 3, device="cuda").requires_grad_(True) * std_xyz[None] + mean_xyz[None])[None].repeat(fused_color.shape[0], 1, 1))
        self._deformation = self._deformation.to("cuda")
        # self.grid = self.grid.to("cuda")

        # self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))

        features9channel = torch.cat((fused_color, fused_color), dim=1)
        self._features_dc = nn.Parameter(features9channel.contiguous().requires_grad_(True))
        N, _ = fused_color.shape

        fomega = torch.zeros((N, 3), dtype=torch.float, device="cuda")
        self._features_t =  nn.Parameter(fomega.contiguous().requires_grad_(True))

        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._deformation_table = torch.gt(torch.ones((self.get_xyz.shape[0]), device="cuda"), 0)

        omega = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        self._omega = nn.Parameter(omega.requires_grad_(True))
        self.initial_control_xyz = self.control_xyz.clone().detach()
        motion = torch.zeros((fused_point_cloud.shape[0], 9), device="cuda")# x1, x2, x3,  y1,y2,y3, z1,z2,z3
        self._motion = nn.Parameter(motion.requires_grad_(True))

        zeta = torch.zeros((fused_point_cloud.shape[0], 1), device="cuda")
        self._zeta = nn.Parameter(zeta.requires_grad_(True))

        self._trbf_center = nn.Parameter(times.contiguous().requires_grad_(True))
        self._trbf_scale = nn.Parameter(torch.ones((self.get_xyz.shape[0], 1), device="cuda").requires_grad_(True))

        ## store gradients
        if self.trbfslinit is not None:
            nn.init.constant_(self._trbf_scale, self.trbfslinit) # too large ?
        else:
            nn.init.constant_(self._trbf_scale, 0) # too large ?

        nn.init.constant_(self._omega, 0)
        nn.init.constant_(self._zeta, 0)

        self.rgb_grd = {}

        self.maxz, self.minz = torch.amax(self._xyz[:,2]), torch.amin(self._xyz[:,2])
        self.maxy, self.miny = torch.amax(self._xyz[:,1]), torch.amin(self._xyz[:,1])
        self.maxx, self.minx = torch.amax(self._xyz[:,0]), torch.amin(self._xyz[:,0])
        self.maxz = min((self.maxz, 200.0)) # some outliers in the n4d datasets..

        for name, W in self.rgbdecoder.named_parameters():
            if 'weight' in name:
                self.rgb_grd[name] = torch.zeros_like(W, requires_grad=False).cuda() #self.rgb_grd[name] + W.grad.clone()
            elif 'bias' in name:
                print('not implemented')
                quit()


    def get_params(self):
        gs_params = [self._xyz, self.control_xyz, self.current_control_num, self._features_dc, self._features_rest, self._opacity, self._scaling, self._rotation]
        return gs_params + list(self._deformation.get_mlp_parameters()) + list(self._deformation.get_grid_parameters()) \
               + list(self._posenet.get_mlp_parameters())

    def training_setup(self, training_args, stage="coarse", is_static=False):
        self.max_scale = float(getattr(training_args, "max_scale", self.max_scale))
        self.min_scale = float(getattr(training_args, "min_scale", self.min_scale))
        if self.max_scale < self.min_scale:
            self.max_scale, self.min_scale = self.min_scale, self.max_scale
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        self.preprocesspoints
        self.rgbdecoder.cuda()

        # Adjust learning rates based on is_static and stage
        if is_static and stage == "fine":
            static_lr_factor = 0.1  # Reduce the learning rate by 1/10 for static and fine stage
        else:
            static_lr_factor = 1.0  # Default learning rate factor is 1

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale * static_lr_factor, "name": "xyz"},
            {'params': [self.control_xyz], 'lr': 30 * training_args.position_lr_init * self.spatial_lr_scale * static_lr_factor, "name": "control_xyz"},
            {'params': [self.current_control_num], 'lr': 0.0, "name": "current_control_num"},
            {'params': list(self._deformation.get_mlp_parameters()),
            'lr': training_args.deformation_lr_init * self.spatial_lr_scale * static_lr_factor, "name": "deformation"},
            {'params': list(self._deformation.get_grid_parameters()),
            'lr': training_args.grid_lr_init * self.spatial_lr_scale * static_lr_factor, "name": "grid"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr * static_lr_factor, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0 * static_lr_factor, "name": "f_rest"},
            {'params': [self._features_t], 'lr': training_args.featuret_lr * static_lr_factor, "name": "f_t"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr * static_lr_factor, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr * static_lr_factor, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr * static_lr_factor, "name": "rotation"},
            {'params': [self._omega], 'lr': training_args.omega_lr * static_lr_factor, "name": "omega"},
            {'params': [self._zeta], 'lr': training_args.zeta_lr * static_lr_factor, "name": "zeta"},
            {'params': [self._trbf_center], 'lr': training_args.trbfc_lr * static_lr_factor, "name": "trbf_center"},
            {'params': [self._trbf_scale], 'lr': training_args.trbfs_lr * static_lr_factor, "name": "trbf_scale"},
            {'params': [self._motion], 'lr': training_args.position_lr_init * self.spatial_lr_scale * 0.5 * training_args.movelr * static_lr_factor, "name": "motion"},
            {'params': list(self.rgbdecoder.parameters()), 'lr': training_args.rgb_lr * static_lr_factor, "name": "decoder"},
        ]

        # Pose is run during warm up, we want a lower starting LR for fine
        if stage == "fine":
            if self._posenet is not None:
                l.append({'params': list(self._posenet.get_mlp_parameters()),
                        'lr': training_args.pose_lr_init/10, "name": "posenet"})
                self.pose_scheduler_args = get_expon_lr_func(
                    lr_init=training_args.pose_lr_init/10,
                    lr_final=training_args.pose_lr_final,
                    max_steps=training_args.position_lr_max_steps)
        else:
            if self._posenet is not None:
                l.append({'params': list(self._posenet.get_focal_parameters()), 'lr': 0.005, "name": "focal"})
                l.append({'params': list(self._posenet.get_mlp_parameters()),
                        'lr': training_args.pose_lr_init, "name": "posenet"})
                l.append({'params': list(self._posenet.get_scale_parameters()),
                        'lr': training_args.pose_lr_init, "name": "posenet_cvdscale"})
                self.pose_scheduler_args = get_expon_lr_func(
                    lr_init=training_args.pose_lr_init,
                    lr_final=training_args.pose_lr_final,
                    max_steps=training_args.position_lr_max_steps)

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale * static_lr_factor,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale * static_lr_factor,
            max_steps=training_args.position_lr_max_steps)

        self.deformation_scheduler_args = get_expon_lr_func(
            lr_init=training_args.deformation_lr_init * self.spatial_lr_scale * static_lr_factor,
            lr_final=training_args.deformation_lr_final * self.spatial_lr_scale * static_lr_factor,
            max_steps=training_args.position_lr_max_steps)

        self.grid_scheduler_args = get_expon_lr_func(lr_init=training_args.grid_lr_init * self.spatial_lr_scale * static_lr_factor,
                                                    lr_final=training_args.grid_lr_final * self.spatial_lr_scale * static_lr_factor,
                                                    max_steps=training_args.position_lr_max_steps)
        self.apply_scale_clamp()


    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr
            elif "grid" in param_group["name"]:
                lr = self.grid_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr
            elif param_group["name"] == "deformation":
                lr = self.deformation_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr
            elif param_group["name"] == "posenet":
                lr = self.pose_scheduler_args(iteration)
                param_group['lr'] = lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'trbf_center', 'trbf_scale' ,'nx', 'ny', 'nz'] # 'trbf_center', 'trbf_scale'
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        for i in range(self._features_t.shape[1]):
            l.append('f_t_{}'.format(i))
        for i in range(self._motion.shape[1]):
            l.append('motion_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self._omega.shape[1]):
            l.append('omega_{}'.format(i))
        for i in range(self._zeta.shape[1]):
            l.append('zeta_{}'.format(i))
        for i in range(self.control_xyz.shape[1]):
            for j in range(self.control_xyz.shape[2]):
                if j == 0:
                    l.append('control_x_{}'.format(i))
                elif j == 1:
                    l.append('control_y_{}'.format(i))
                elif j == 2:
                    l.append('control_z_{}'.format(i))
        l.append('current_control_num')
        for i in range(self.parent_node_idx.shape[1]):
            l.append('parent_node_idx_{}'.format(i))
        return l

    def compute_deformation(self, time):
        deform = self._deformation[:, :, :time].sum(dim=-1)
        xyz = self._xyz + deform
        return xyz

    def load_model(self, path):
        print("loading model from exists{}".format(path))
        weight_dict = torch.load(os.path.join(path, "deformation.pth"), map_location="cuda")
        self._deformation.load_state_dict(weight_dict)
        self._deformation = self._deformation.to("cuda")
        self._deformation_table = torch.gt(torch.ones((self.get_xyz.shape[0]), device="cuda"), 0)
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        if os.path.exists(os.path.join(path, "deformation_table.pth")):
            self._deformation_table = torch.load(os.path.join(path, "deformation_table.pth"), map_location="cuda")
        if os.path.exists(os.path.join(path, "deformation_accum.pth")):
            self._deformation_accum = torch.load(os.path.join(path, "deformation_accum.pth"), map_location="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        # print(self._deformation.deformation_net.grid.)

        weight_dict = torch.load(os.path.join(path, "posenet.pth"), map_location="cuda")
        if self._posenet is not None:
            self._posenet.load_state_dict(weight_dict)
            self._posenet = self._posenet.to("cuda")

    def save_deformation(self, path):
        torch.save(self._deformation.state_dict(), os.path.join(path, "deformation.pth"))
        torch.save(self._deformation_table, os.path.join(path, "deformation_table.pth"))
        torch.save(self._deformation_accum, os.path.join(path, "deformation_accum.pth"))
        torch.save(self._posenet.state_dict(), os.path.join(path, "posenet.pth"))

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        control_xyz = self.control_xyz.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        current_control_num = self.current_control_num.detach().cpu().numpy()
        parent_node_idx = self.parent_node_idx.detach().cpu().numpy()
        
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().cpu().numpy()
        # f_rest = self._features_rest.detach().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

        f_t =  self._features_t.detach().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        trbf_center= self._trbf_center.detach().cpu().numpy()

        trbf_scale = self._trbf_scale.detach().cpu().numpy()
        motion = self._motion.detach().cpu().numpy()

        omega = self._omega.detach().cpu().numpy()
        zeta = self._zeta.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        # attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        attributes = np.concatenate((xyz, trbf_center, trbf_scale, normals, f_dc, f_rest, f_t, motion, opacities, scale, rotation, omega, zeta, control_xyz, current_control_num, parent_node_idx), axis=1)
        elements[:] = list(map(tuple, attributes))

        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        model_fname = path.replace(".ply", ".pt")
        print(f'Saving model checkpoint to: {model_fname}')
        torch.save(self.rgbdecoder.state_dict(), model_fname)
        init_ctrl_fname = path.replace(".ply", "_init_control.pt")
        final_ctrl_fname = path.replace(".ply", "_final_control.pt")
        torch.save(self.initial_control_xyz.detach().cpu(), init_ctrl_fname)
        torch.save(self.control_xyz.detach().cpu(), final_ctrl_fname)
        
    def save_ply_compact(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        f_dc = self._features_dc.detach().cpu().numpy()

        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes_compact()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, f_dc, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))

        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        model_fname = path.replace(".ply", ".pt")
        print(f'Saving model checkpoint to: {model_fname}')
        torch.save(self.rgbdecoder.state_dict(), model_fname)

    def save_ply_compact_dy(self, path):
        mkdir_p(os.path.dirname(path))
        xyz = self._xyz.detach().cpu().numpy()
        # control_xyz = self.control_xyz.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        current_control_num = self.current_control_num.detach().cpu().numpy()

        f_dc = self._features_dc.detach().cpu().numpy()


        f_t =  self._features_t.detach().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        omega = self._omega.detach().cpu().numpy()
        zeta = self._zeta.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes_compact_dy()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        # attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        attributes = np.concatenate((f_dc, f_t, opacities, scale, rotation, omega, zeta,  current_control_num), axis=1)
        elements[:] = list(map(tuple, attributes))


        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        model_fname = path.replace(".ply", ".pt")
        print(f'Saving model checkpoint to: {model_fname}')
        torch.save(self.rgbdecoder.state_dict(), model_fname)

        control_fname = path.replace(".ply", ".npy")
        print(f'Saving control points to: {control_fname}')
        # np.save(control_fname, self.flat_control_xyz.detach().cpu().numpy())
        np.savez_compressed(control_fname.replace('npy', 'npz'), flat_control_xyz=self.flat_control_xyz.detach().cpu().numpy())

    def construct_list_of_attributes_compact(self):
        l = ['x', 'y', 'z']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]):
            l.append('f_dc_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def construct_list_of_attributes_compact_dy(self):
        l = [] # 'trbf_center', 'trbf_scale'
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_t.shape[1]):
            l.append('f_t_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self._omega.shape[1]):
            l.append('omega_{}'.format(i))
        for i in range(self._zeta.shape[1]):
            l.append('zeta_{}'.format(i))
        l.append('current_control_num')
        return l

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        if torch.isnan(opacities_new).any():
            print("opacities_new is nan,end training, ending program now.")
            exit()
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]


    def zero_omega(self, threhold=0.15):
        scales = self.get_scaling
        omegamask = torch.sum(torch.abs(self._omega), dim=1) > threhold # default
        scalemask = torch.max(scales, dim=1).values.unsqueeze(1) > 0.2
        scalemaskb = torch.max(scales, dim=1).values.unsqueeze(1) < 0.6
        pointopacity = self.get_opacity
        opacitymask = pointopacity > 0.7

        mask = torch.logical_and(torch.logical_and(omegamask.unsqueeze(1), scalemask), torch.logical_and(scalemaskb, opacitymask))
        omeganew = mask.float() * self._omega
        optimizable_tensors = self.replace_tensor_to_optimizer(omeganew, "omega")
        self._omega = optimizable_tensors["omega"]
        return mask

    def zero_omegabymotion(self, threhold=0.15):
        scales = self.get_scaling
        omegamask = torch.sum(torch.abs(self._motion[:, 0:3]), dim=1) > 0.3 #  #torch.sum(torch.abs(self._omega), dim=1) > threhold # default
        scalemask = torch.max(scales, dim=1).values.unsqueeze(1) > 0.2
        scalemaskb = torch.max(scales, dim=1).values.unsqueeze(1) < 0.6
        pointopacity = self.get_opacity
        opacitymask = pointopacity > 0.7
        mask = torch.logical_and(torch.logical_and(omegamask.unsqueeze(1), scalemask), torch.logical_and(scalemaskb, opacitymask))

        omeganew = mask.float() * self._omega
        optimizable_tensors = self.replace_tensor_to_optimizer(omeganew, "omega")
        self._omega = optimizable_tensors["omega"]
        return mask

    def load_ply(self, path):
        plydata = PlyData.read(path)

        ckpt = torch.load(path.replace(".ply", ".pt"))
        self.rgbdecoder.cuda()
        self.rgbdecoder.load_state_dict(ckpt)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        control_xyz_list = []
        for i in range(self.control_num):
            control_xyz_list.append(np.stack((np.asarray(plydata.elements[0][f"control_x_{i}"]),
                            np.asarray(plydata.elements[0][f"control_y_{i}"]),
                            np.asarray(plydata.elements[0][f"control_z_{i}"])), axis=1))
        control_xyz = np.stack(control_xyz_list, axis=1)
        current_control_num = np.asarray(plydata.elements[0]["current_control_num"])[..., np.newaxis]

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        trbf_center= np.asarray(plydata.elements[0]["trbf_center"])[..., np.newaxis]
        trbf_scale = np.asarray(plydata.elements[0]["trbf_scale"])[..., np.newaxis]

        motion_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("motion")]
        nummotion = 9
        motion = np.zeros((xyz.shape[0], nummotion))
        for i in range(nummotion):
            motion[:, i] = np.asarray(plydata.elements[0]["motion_"+str(i)])

        fdc_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_dc")]
        features_dc = np.zeros((xyz.shape[0], len(fdc_names)))
        for idx, attr_name in enumerate(fdc_names):
            features_dc[:, idx] = np.asarray(plydata.elements[0][attr_name])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        # assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2))

        ft_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_t")]
        ftomegas = np.zeros((xyz.shape[0], len(ft_names)))
        for idx, attr_name in enumerate(ft_names):
            ftomegas[:, idx] = np.asarray(plydata.elements[0][attr_name])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        omega_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("omega_")]
        omegas = np.zeros((xyz.shape[0], len(omega_names)))
        for idx, attr_name in enumerate(omega_names):
            omegas[:, idx] = np.asarray(plydata.elements[0][attr_name])

        zeta_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("zeta_")]
        zetas = np.zeros((xyz.shape[0], len(zeta_names)))
        for idx, attr_name in enumerate(zeta_names):
            zetas[:, idx] = np.asarray(plydata.elements[0][attr_name])

        parent_node_idx_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("parent_node_idx_")]
        parent_node_idxs = np.zeros((xyz.shape[0], len(parent_node_idx_names)))
        for idx, attr_name in enumerate(parent_node_idx_names):
            parent_node_idxs[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self.control_xyz = nn.Parameter(torch.tensor(control_xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self.current_control_num = nn.Parameter(torch.tensor(current_control_num, dtype=torch.int64, device="cuda"), requires_grad=False)
        self.parent_node_idx = torch.tensor(parent_node_idxs, dtype=torch.int64, device="cuda")
        
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").squeeze().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_t = nn.Parameter(torch.tensor(ftomegas, dtype=torch.float, device="cuda").requires_grad_(True))

        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self._trbf_center = nn.Parameter(torch.tensor(trbf_center, dtype=torch.float, device="cuda").requires_grad_(True))
        self._trbf_scale = nn.Parameter(torch.tensor(trbf_scale, dtype=torch.float, device="cuda").requires_grad_(True))
        self._motion = nn.Parameter(torch.tensor(motion, dtype=torch.float, device="cuda").requires_grad_(True))
        self._omega = nn.Parameter(torch.tensor(omegas, dtype=torch.float, device="cuda").requires_grad_(True))
        self._zeta = nn.Parameter(torch.tensor(zetas, dtype=torch.float, device="cuda").requires_grad_(True))
        
        init_ctrl_fname = path.replace(".ply", "_init_control.pt")
        if os.path.exists(init_ctrl_fname):
            loaded_init_ctrl = torch.load(init_ctrl_fname).cuda()
            self.initial_control_xyz = loaded_init_ctrl
        else:
            self.initial_control_xyz = self.control_xyz.clone().detach()

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) > 1 or group["name"] == "focal":
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                if group["name"] == "current_control_num":
                    group["params"][0] = nn.Parameter(group["params"][0][mask], requires_grad=False)
                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self.control_xyz = optimizable_tensors["control_xyz"]
        self.current_control_num = optimizable_tensors["current_control_num"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._features_t = optimizable_tensors["f_t"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self._deformation_accum = self._deformation_accum[valid_points_mask]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self._deformation_table = self._deformation_table[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.initial_control_xyz = self.initial_control_xyz[valid_points_mask]
        self.parent_node_idx = self.parent_node_idx[valid_points_mask]
        
        if hasattr(self, 'parent_local_knn_nodes') and self.parent_local_knn_nodes.numel() > 0:
            self.parent_local_knn_nodes = self.parent_local_knn_nodes[valid_points_mask]
        if hasattr(self, 'parent_mlevel_knn_nodes') and len(self.parent_mlevel_knn_nodes) > 0:
            self.parent_mlevel_knn_nodes = [m_knn[valid_points_mask] for m_knn in self.parent_mlevel_knn_nodes]
        
        self._trbf_center = optimizable_tensors["trbf_center"]
        self._trbf_scale = optimizable_tensors["trbf_scale"]
        self._motion = optimizable_tensors["motion"]
        self._omega = optimizable_tensors["omega"]
        self._zeta = optimizable_tensors["zeta"]
        
    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) > 1 or group["name"] == "focal": continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                if group["name"] == "current_control_num":
                    group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0), requires_grad=False)
                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(
                        torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    @property
    def get_aabb(self):
        return self._deformation.get_aabb

    def get_displayment(self, selected_point, point, perturb):
        xyz_max, xyz_min = self.get_aabb
        displacements = torch.randn(selected_point.shape[0], 3).to(selected_point) * perturb
        final_point = selected_point + displacements

        mask_a = final_point < xyz_max
        mask_b = final_point > xyz_min
        mask_c = mask_a & mask_b
        mask_d = mask_c.all(dim=1)
        final_point = final_point[mask_d]

        return final_point, mask_d

    @torch.no_grad()
    def update_deformation_table(self, threshold):
        # print("origin deformation point nums:",self._deformation_table.sum())
        self._deformation_table = torch.gt(self._deformation_accum.max(dim=-1).values / 100, threshold)

    def print_deformation_weight_grad(self):
        for name, weight in self._deformation.named_parameters():
            if weight.requires_grad:
                if weight.grad is None:

                    print(name, " :", weight.grad)
                else:
                    if weight.grad.mean() != 0:
                        print(name, " :", weight.grad.mean(), weight.grad.min(), weight.grad.max())
        print("-" * 50)

    def _plane_regulation(self):
        multi_res_grids = self._deformation.deformation_net.grid.grids
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids = [0, 1, 3]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return total

    def _time_regulation(self):
        multi_res_grids = self._deformation.deformation_net.grid.grids
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids = [2, 4, 5]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return total

    def _l1_regulation(self):
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        multi_res_grids = self._deformation.deformation_net.grid.grids

        total = 0.0
        for grids in multi_res_grids:
            if len(grids) == 3:
                continue
            else:
                # These are the spatiotemporal grids
                spatiotemporal_grids = [2, 4, 5]
            for grid_id in spatiotemporal_grids:
                total += torch.abs(1 - grids[grid_id]).mean()
        return total

    def compute_regulation(self, time_smoothness_weight, l1_time_planes_weight, plane_tv_weight):
        return plane_tv_weight * self._plane_regulation() + time_smoothness_weight * self._time_regulation() + l1_time_planes_weight * self._l1_regulation()


    def densification_postfix(self, new_xyz, new_control_xyz, new_current_control_num, new_features_dc, new_features_rest, new_features_t,
                              new_opacities, new_scaling, new_rotation, new_deformation_table, new_trbf_center, new_trbfscale, new_motion,
                              new_omega, new_zeta, new_initial_control_xyz, new_parent_node_idx,
                              new_parent_local_knn_nodes=None, new_parent_mlevel_knn_nodes=None):
        new_scaling = self._clamp_scale_logits(new_scaling)
        d = {"xyz": new_xyz,
             "control_xyz": new_control_xyz,
             "current_control_num": new_current_control_num,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "f_t": new_features_t,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation,
             "trbf_center" : new_trbf_center,
             "trbf_scale" : new_trbfscale,
             "motion": new_motion,
             "omega": new_omega,
             "zeta" : new_zeta
             }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self.control_xyz = optimizable_tensors["control_xyz"]
        self.current_control_num  = optimizable_tensors["current_control_num"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._features_t = optimizable_tensors["f_t"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self._deformation_table = torch.cat([self._deformation_table, new_deformation_table], -1)
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self._trbf_center = optimizable_tensors["trbf_center"]
        self._trbf_scale = optimizable_tensors["trbf_scale"]
        self._motion = optimizable_tensors["motion"]
        self._omega = optimizable_tensors["omega"]
        self._zeta = optimizable_tensors["zeta"]
        self.initial_control_xyz = torch.cat([self.initial_control_xyz, new_initial_control_xyz], dim=0)
        self.parent_node_idx = torch.cat([self.parent_node_idx, new_parent_node_idx], dim=0)
        
        if new_parent_local_knn_nodes is not None:
            self.parent_local_knn_nodes = torch.cat([self.parent_local_knn_nodes, new_parent_local_knn_nodes], dim=0)
        if new_parent_mlevel_knn_nodes is not None:
            self.parent_mlevel_knn_nodes = [torch.cat([self.parent_mlevel_knn_nodes[i], new_parent_mlevel_knn_nodes[i]], dim=0) for i in range(len(self.parent_mlevel_knn_nodes))]
        
    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)

        if not selected_pts_mask.any():
            return

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling_physical = self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        new_scaling = self._physical_scales_to_logits(new_scaling_physical)
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)

        new_control_xyz = self.control_xyz[selected_pts_mask].repeat(N, 1, 1)
        new_current_control_num = self.current_control_num[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_features_t = self._features_t[selected_pts_mask].repeat(N, 1)

        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_deformation_table = self._deformation_table[selected_pts_mask].repeat(N)

        new_trbf_center = self._trbf_center[selected_pts_mask].repeat(N,1)
        new_trbf_scale = self._trbf_scale[selected_pts_mask].repeat(N,1)
        new_motion = self._motion[selected_pts_mask].repeat(N,1)
        new_omega = self._omega[selected_pts_mask].repeat(N,1)
        new_zeta = self._zeta[selected_pts_mask].repeat(N,1)

        new_initial_control_xyz = self.initial_control_xyz[selected_pts_mask].repeat(N, 1, 1)
        new_parent_node_idx = self.parent_node_idx[selected_pts_mask].repeat(N, 1)
        
        new_parent_local_knn_nodes = self.parent_local_knn_nodes[selected_pts_mask].repeat(N, 1) if hasattr(self, 'parent_local_knn_nodes') and self.parent_local_knn_nodes.numel() > 0 else None
        if hasattr(self, 'parent_mlevel_knn_nodes') and len(self.parent_mlevel_knn_nodes) > 0:
            new_parent_mlevel_knn_nodes = [m_knn[selected_pts_mask].repeat(N, 1) for m_knn in self.parent_mlevel_knn_nodes]
        else:
            new_parent_mlevel_knn_nodes = None
        
        self.densification_postfix(new_xyz, new_control_xyz, new_current_control_num, new_features_dc, new_features_rest, new_features_t,
                                   new_opacity, new_scaling, new_rotation, new_deformation_table, new_trbf_center, new_trbf_scale,
                                   new_motion, new_omega, new_zeta, new_initial_control_xyz, new_parent_node_idx,
                                   new_parent_local_knn_nodes, new_parent_mlevel_knn_nodes)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent)
        new_xyz = self._xyz[selected_pts_mask]
        new_control_xyz = self.control_xyz[selected_pts_mask]
        new_current_control_num = self.current_control_num[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_features_t = self._features_t[selected_pts_mask]

        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_deformation_table = self._deformation_table[selected_pts_mask]

        new_trbf_center = self._trbf_center[selected_pts_mask]
        new_trbf_scale = self._trbf_scale[selected_pts_mask]
        new_motion = self._motion[selected_pts_mask]
        new_omega = self._omega[selected_pts_mask]
        new_zeta = self._zeta[selected_pts_mask]
        new_initial_control_xyz = self.initial_control_xyz[selected_pts_mask]
        new_parent_node_idx = self.parent_node_idx[selected_pts_mask]
        
        new_parent_local_knn_nodes = self.parent_local_knn_nodes[selected_pts_mask] if hasattr(self, 'parent_local_knn_nodes') and self.parent_local_knn_nodes.numel() > 0 else None
        if hasattr(self, 'parent_mlevel_knn_nodes') and len(self.parent_mlevel_knn_nodes) > 0:
            new_parent_mlevel_knn_nodes = [m_knn[selected_pts_mask] for m_knn in self.parent_mlevel_knn_nodes]
        else:
            new_parent_mlevel_knn_nodes = None
        
        self.densification_postfix(new_xyz, new_control_xyz, new_current_control_num, new_features_dc, new_features_rest, new_features_t,
                                   new_opacities, new_scaling, new_rotation, new_deformation_table, new_trbf_center, new_trbf_scale,
                                   new_motion, new_omega, new_zeta, new_initial_control_xyz, new_parent_node_idx,
                                   new_parent_local_knn_nodes, new_parent_mlevel_knn_nodes)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
