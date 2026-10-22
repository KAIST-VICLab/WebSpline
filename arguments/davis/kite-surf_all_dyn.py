_base_ = './default.py'

# 20 frames, every pixel masked dynamic.
ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 32,
     'resolution': [64, 64, 64, 20]
    }
)

# The static photometric loss is masked by (1 - motion_mask), which is identically
# zero here, so the fine_static stage has nothing to learn. Keep it short and give
# the budget to the fine stage instead.
OptimizationParams = dict(
    fine_static_iterations = 1000,
    # Every pixel is dynamic, so the dynamic init would otherwise take the whole
    # image and the scaffold would be unusably large.
    max_dyn_init_pts = 2000,
    # The camera barely translates here, so the camera-derived extent (~0.03) is
    # ~20x too small and densify_and_prune would drop everything.
    densify_extent_from_pcd = True,
)
