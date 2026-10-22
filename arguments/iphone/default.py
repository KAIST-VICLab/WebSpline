ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [64, 64, 64, 150]
    },
    multires = [1,2,4],
    defor_depth = 1,
    net_width = 128,
    plane_tv_weight = 0.0002,
    time_smoothness_weight = 0.001,
    l1_time_planes =  0.0001,
)

OptimizationParams = dict(
    # 30000 fine iterations. save_iterations defaults to every 20000 plus this
    # value, so the run leaves checkpoints at 20000 and 30000; 20000 is the one
    # that evaluates best, and is what the run scripts point eval.py at.
    iterations = 30000,
    batch_size=1,
    coarse_iterations = 1000,
    fine_static_iterations = 15000,
    densify_until_iter = 10000,
    opacity_reset_interval = 3_000,
    stat_npts = 40000,
    dyn_npts = 20000,
)