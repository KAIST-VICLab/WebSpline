_base_ = './default.py'

# 25 frames: keep the temporal plane resolution matched to the sequence length
# (this also sets the number of spline control points: one per frame).
ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 32,
     'resolution': [64, 64, 64, 25]
    }
)
