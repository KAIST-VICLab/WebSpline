_base_ = './default.py'

# 50 frames: keep the temporal plane resolution matched to the sequence length.
ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 32,
     'resolution': [64, 64, 64, 50]
    }
)
