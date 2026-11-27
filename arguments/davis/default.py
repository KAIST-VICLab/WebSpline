# Shared DAVIS settings; arguments/davis/<scene>.py only sets the frame count.
# The DAVIS-specific method settings (bundle adjustment, track-curve SPG scaffold,
# median depth normalisation, SplineGS-style colour decoder, one control point per
# frame, motion-adaptive control lr, no size cap) are applied automatically when the
# source directory is detected as DAVIS: see DATASET_DEFAULTS in arguments/__init__.py
# and DATASET_PRESETS["davis"] in train.py.  Anything set here or in a scene file
# overrides them.
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
