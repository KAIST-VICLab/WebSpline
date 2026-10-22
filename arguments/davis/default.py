ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [64, 64, 64, 64]
    },
    multires = [1,2,4],
    defor_depth = 1,
    net_width = 128,
    plane_tv_weight = 0.0002,
    time_smoothness_weight = 0.001,
    l1_time_planes =  0.0001,
)

OptimizationParams = dict(
    iterations = 30000,
    batch_size=1,
    coarse_iterations = 1000,
    fine_static_iterations = 15000,
    densify_until_iter = 10000,
    opacity_reset_interval = 3_000,
    stat_npts = 40000,
    dyn_npts = 20000,
)
