# MIT License

# Copyright (c) 2023 OPPO

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch 
import numpy as np
from simple_knn._C import distCUDA2
# from mmcv.ops import knn
import torch.nn as nn



def get_plucker_rays(rays):
    """
    rays: [B, 6, H, W], channels 0:3 origin and 3:6 direction
    """
    o = rays[:, :3, :, :]
    d = rays[:, 3:, :, :]
    
    # Normalising the direction helps training
    d_norm = torch.nn.functional.normalize(d, dim=1)
    
    # Moment, o x d
    m = torch.cross(o, d_norm, dim=1)
    
    # Plücker coordinates, [d, m]
    plucker = torch.cat([d_norm, m], dim=1) # [B, 6, H, W]
    return plucker


class PositionalEncoding(nn.Module):
    def __init__(self, in_channels, num_frequencies=4):
        super().__init__()
        self.num_frequencies = num_frequencies
        # Output dim = input dim + input dim * 2 * num_freqs
        # e.g. 6-d Plücker -> 6 + 6 * 2 * 4 = 54
        self.out_dim = in_channels + (in_channels * 2 * num_frequencies)

    def forward(self, x):
        out = [x]
        for i in range(self.num_frequencies):
            for func in [torch.sin, torch.cos]:
                out.append(func(x * (2.0 ** i)))
        return torch.cat(out, dim=1)

class Sandwich(nn.Module):
    def __init__(self, spec_in_dim=3, time_in_dim=3, ray_in_dim=6, hidden_dim=64, bias=False):
        super(Sandwich, self).__init__()
        
        # Ray encoding; 4-6 frequencies works well
        self.ray_encoder = PositionalEncoding(in_channels=ray_in_dim, num_frequencies=4)
        
        # Account for the encoded width: ray_in_dim goes from 6 to 54
        total_in_dim = spec_in_dim + time_in_dim + self.ray_encoder.out_dim
        
        # The wider input warrants a larger hidden_dim (32 -> 64)
        self.mlp1 = nn.Conv2d(total_in_dim, hidden_dim, kernel_size=1, bias=bias)
        self.mlp2 = nn.Conv2d(hidden_dim, 3, kernel_size=1, bias=bias)
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, rays):
        albedo, spec, timefeature = input.chunk(3, dim=1)
        
        # To Plücker coordinates
        plucker_rays = get_plucker_rays(rays) # [1, 6, H, W]
        
        # Positional encoding on the Plücker coordinates
        encoded_rays = self.ray_encoder(plucker_rays) # [1, 54, H, W]
        
        # Concatenate the encoded ray features
        specular_in = torch.cat([spec, timefeature, encoded_rays], dim=1)
        
        x = self.relu(self.mlp1(specular_in))
        specular_out = self.mlp2(x)

        albedo_out = self.sigmoid(albedo)
        specular_out = self.relu(specular_out)
        
        result = torch.clamp(albedo_out + specular_out, 0.0, 1.0)
        return result


class Sandwichnoact(nn.Module):
    def __init__(self, dim, outdim=3, bias=False):
        super(Sandwichnoact, self).__init__()
        
        self.mlp1 = nn.Conv2d(12, 6, kernel_size=1, bias=bias) 
        self.mlp2 = nn.Conv2d(6, 3, kernel_size=1, bias=bias)
        self.relu = nn.ReLU()



    def forward(self, input, rays, time=None):
        albedo, spec, timefeature = input.chunk(3,dim=1)
        specular = torch.cat([spec, timefeature, rays], dim=1) # 3+3 + 5
        specular = self.mlp1(specular)
        specular = self.relu(specular)
        specular = self.mlp2(specular)

        result = albedo + specular
        result = torch.clamp(result, min=0.0, max=1.0)
        return result

class Sandwichnoactss(nn.Module):
    def __init__(self, dim, outdim=3, bias=False):
        super(Sandwichnoactss, self).__init__()
        
        self.mlp1 = nn.Conv2d(12, 6, kernel_size=1, bias=bias)  
        self.mlp2 = nn.Conv2d(6, 3, kernel_size=1, bias=bias)


        self.relu = nn.ReLU()



    def forward(self, input, rays, time=None):
        albedo, spec, timefeature = input.chunk(3,dim=1)
        specular = torch.cat([spec, timefeature, rays], dim=1) # 3+3 + 5
        specular = self.mlp1(specular)
        specular = self.relu(specular)
        specular = self.mlp2(specular)

        result = albedo + specular
        return result
    
    
####### following are also good rgb model but not used in the paper, slower than sandwich, inspired by color shift in hyperreel
# remove sigmoid for immersive dataset
class RGBDecoderVRayShift(nn.Module):
    def __init__(self, dim, outdim=3, bias=False):
        super(RGBDecoderVRayShift, self).__init__()
        
        self.mlp1 = nn.Conv2d(dim, outdim, kernel_size=1, bias=bias)
        self.mlp2 = nn.Conv2d(15, outdim, kernel_size=1, bias=bias)
        self.mlp3 = nn.Conv2d(6, outdim, kernel_size=1, bias=bias)
        self.sigmoid = torch.nn.Sigmoid()

        self.dwconv1 = nn.Conv2d(9, 9, kernel_size=1, bias=bias)

    def forward(self, input, rays, t=None):
        x = self.dwconv1(input) + input 
        albeado = self.mlp1(x)
        specualr = torch.cat([x, rays], dim=1)
        specualr = self.mlp2(specualr)

        finalfeature = torch.cat([albeado, specualr], dim=1)
        result = self.mlp3(finalfeature)
        result = self.sigmoid(result)   
        return result 
    


def getcolormodel(rgbfuntion):
    if rgbfuntion == "sandwich":
        rgbdecoder = Sandwich()
    
    elif rgbfuntion == "sandwichnoact":
        rgbdecoder = Sandwichnoact(9,3)
    elif rgbfuntion == "sandwichnoactss":
        rgbdecoder = Sandwichnoactss(9,3)
    else :
        return None 
    return rgbdecoder

def pix2ndc(v, S):
    return (v * 2.0 + 1.0) / S - 1.0


def ndc2pix(v, S):
    return ((v + 1.0) * S - 1.0) * 0.5



