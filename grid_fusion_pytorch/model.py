import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def conv3d3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv3d:
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv3d1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv3d:
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# +
def leaky_relu_init(m, negative_slope=0.2):
    gain = np.sqrt(2.0 / (1.0 + negative_slope ** 2))
    if isinstance(m, torch.nn.Conv1d):
        ksize = m.kernel_size[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, torch.nn.Conv2d):
        ksize = m.kernel_size[0] * m.kernel_size[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, torch.nn.Conv3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, torch.nn.ConvTranspose1d):
        ksize = m.kernel_size[0] // 2
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, torch.nn.ConvTranspose2d):
        ksize = m.kernel_size[0] * m.kernel_size[1] // 4
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, torch.nn.ConvTranspose3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] // 8
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, torch.nn.Linear):
        n1 = m.in_features
        n2 = m.out_features

        std = gain * np.sqrt(2.0 / (n1 + n2))
    else:
        return

    # is_wnw = is_weight_norm_wrapped(m)
    # if is_wnw:
    #     m.fuse()

    m.weight.data.uniform_(-std * np.sqrt(3.0), std * np.sqrt(3.0))
    if m.bias is not None:
        m.bias.data.zero_()


    # blockwise initialization for transposed convs
    if isinstance(m, torch.nn.ConvTranspose2d):
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]

    if isinstance(m, torch.nn.ConvTranspose3d):
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 0::2, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 0::2, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2, 0::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]

    # if is_wnw:
        # m.unfuse()

def apply_weight_init_fn(m, fn, negative_slope=1.0):
    should_initialize_weight=True
    if not hasattr(m, "weights_initialized"): #if we don't have this then we need to intiialzie
        # fn(m, is_linear, scale)
        should_initialize_weight=True
    elif m.weights_initialized==False: #if we have it but it's set to false
        # fn(m, is_linear, scale)
        should_initialize_weight=True
    else:
        print("skipping weight init on ", m)
        should_initialize_weight=False

    if should_initialize_weight:
        # fn(m, is_linear, scale)
        fn(m,negative_slope)
        # m.weights_initialized=True
        for module in m.children():
            apply_weight_init_fn(module, fn, negative_slope)


# -

# basic resnet block  
class BasicBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, norm_layer = None, non_lin=None):
        super(BasicBlock3d, self).__init__()
        if norm_layer is None or norm_layer == 'BatchNorm3d':
            norm_layer = nn.BatchNorm3d
            norm_kwargs = {'num_features': out_channels}
        elif norm_layer == 'GroupNorm':
            norm_layer = nn.GroupNorm
            num_groups = out_channels // 16 if out_channels % 16 == 0 else out_channels // 2
            norm_kwargs = {'num_groups': num_groups, 'num_channels': out_channels}
        elif norm_layer == "InstanceNorm3d":
            norm_layer = nn.InstanceNorm3d
            norm_kwargs = {'num_features': out_channels, 'track_running_stats': False, 'affine': True}
        else:
            raise NotImplementedError
        if non_lin is None or non_lin == 'ReLU':
            non_lin = nn.ReLU
        elif non_lin == 'LeakyReLU':
            non_lin = nn.LeakyReLU
        else:
            raise NotImplementedError
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.convnet = nn.Sequential(conv3d3x3(in_channels, out_channels, stride),
                                     norm_layer(**norm_kwargs),
                                     non_lin(inplace=True),
                                     conv3d3x3(out_channels, out_channels),
                                     norm_layer(**norm_kwargs))
        #apply_weight_init_fn(self.convnet, leaky_relu_init, negative_slope=0.0)
        #leaky_relu_init(self.convnet[-2], negative_slope=1.0)
        self.identity = nn.Sequential(conv3d1x1(in_channels, out_channels, stride), norm_layer(**norm_kwargs))
        self.non_lin = non_lin(inplace=False)
        
    def forward(self, x):
        # type: (Tensor) -> Tensor
        out = self.convnet(x) + self.identity(x)
        return self.non_lin(out)

# basic hourglass block  
class HourGlassBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer = None, non_lin=None):
        super(HourGlassBlock, self).__init__()
        if norm_layer is None or norm_layer == 'BatchNorm3d':
            norm_layer = nn.BatchNorm3d
            norm_kwargs = {'num_features': out_channels}
        elif norm_layer == 'GroupNorm':
            norm_layer = nn.GroupNorm
            num_groups = out_channels // 16 if out_channels % 16 == 0 else out_channels // 2
            norm_kwargs = {'num_groups': num_groups, 'num_channels': out_channels}
        elif norm_layer == "InstanceNorm3d":
            norm_layer = nn.InstanceNorm3d
            norm_kwargs = {'num_features': out_channels, 'track_running_stats': False, 'affine': True}
        else:
            raise NotImplementedError
        if non_lin is None or non_lin == 'ReLU':
            non_lin = nn.ReLU
        elif non_lin == 'LeakyReLU':
            non_lin = nn.LeakyReLU
        else:
            raise NotImplementedError
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Sequential(conv3d3x3(in_channels, out_channels, stride=2),
                                   norm_layer(**norm_kwargs),
                                   non_lin(inplace=True),
                                   conv3d3x3(out_channels, out_channels),
                                   norm_layer(**norm_kwargs),
                                   non_lin(inplace=True))
        self.conv2 = nn.Sequential(conv3d3x3(out_channels, out_channels, stride=2),
                                   norm_layer(**norm_kwargs),
                                   non_lin(inplace=True),
                                   conv3d3x3(out_channels, out_channels),
                                   norm_layer(**norm_kwargs),
                                   non_lin(inplace=True),
                                   nn.ConvTranspose3d(out_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                                   norm_layer(**norm_kwargs))
        self.conv3 = nn.Sequential(nn.ConvTranspose3d(out_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                                   norm_layer(**norm_kwargs))  
        self.identity = nn.Sequential(conv3d1x1(in_channels, out_channels), norm_layer(**norm_kwargs))
        self.non_lin = non_lin(inplace=False)
        
    def forward(self, x):
        # type: (Tensor) -> Tensor
        x1 = self.conv1(x)
        x2 = self.non_lin(self.conv2(x1) + x1)
        out = self.conv3(x2) + self.identity(x)
        return self.non_lin(out)

### positional encoding taken from https://colab.research.google.com/drive/1TppdSsLz8uKoNwqJqDGg8se8BHQcvg_K?usp=sharing#scrollTo=rrbs7YoMHAbF
class PositionalEncoder(nn.Module):
  """
  Sine-cosine positional encoder for input points.
  """
  def __init__(
    self,
    d_input: int,
    n_freqs: int,
    log_space: bool = False
  ):
    super().__init__()
    self.d_input = d_input
    self.n_freqs = n_freqs
    self.log_space = log_space
    self.d_output = d_input * (1 + 2 * self.n_freqs)
    self.embed_fns = [lambda x: x]

    # Define frequencies in either linear or log scale
    if self.log_space:
        freq_bands = 2.**torch.linspace(0., self.n_freqs - 1, self.n_freqs)
    else:
        freq_bands = torch.linspace(2.**0., 2.**(self.n_freqs - 1), self.n_freqs)

    # Alternate sin and cos
    for freq in freq_bands:
        self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
        self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))
  
  def forward(self,x) -> torch.Tensor:
    """
    Apply positional encoding to input.
    """
    return torch.concat([fn(x) for fn in self.embed_fns], dim=-1)

class RefineModel(nn.Module):
    def __init__(self, in_channels, out_channels, num_ignore_channels=1, mode='resnet', norm_layer=None, non_lin=None):
        super().__init__()
        self.in_channels = in_channels if isinstance(in_channels, list) else [in_channels]
        self.out_channels = out_channels if isinstance(out_channels, list) else [out_channels]
        assert len(self.in_channels) == len(self.out_channels)
        assert self.in_channels[0] == self.out_channels[-1] + num_ignore_channels
        self.num_ignore_channels = num_ignore_channels
        self.mode = mode
        if self.mode == 'resnet':
            block = BasicBlock3d
        elif self.mode == 'hourglass':
            block = HourGlassBlock
        self.blocks = [block(self.in_channels[i], self.out_channels[i], norm_layer=norm_layer, non_lin=non_lin) for i in range(len(self.in_channels))]
        self.blocks = nn.Sequential(*self.blocks)
        apply_weight_init_fn(self, leaky_relu_init, negative_slope=0.0)
        
    def forward(self, x):
        # type: (Tensor) -> Tensor
        return torch.cat([self.blocks(x), x[:,-self.num_ignore_channels:]], dim=1)

def get_model(config, rank):
    if 'parameter filepath' in config['model'].keys():
        model_path = config['model']['parameter filepath']
        model_config_path = ''.join(['/' + item for item in model_path.split('/')[1:-1]]) + '/model_config.npy' if model_path.split('/')[0] == "" else ''.join(['/' + item for item in model_path.split('/')[:-1]]) + '/model_config.npy'
        model_kwargs = np.load(model_config_path, allow_pickle=True).item()
        model = RefineModel(**model_kwargs)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model_dict = checkpoint['model_state_dict']
        model_dict_converted = {key[7:] : model_dict[key] for key in model_dict}
        model.load_state_dict(model_dict_converted)
    else:
        num_ignore_channels = 1 if not config['voxel grid']['use_pos_enc'] else 1 + config['voxel grid']['pos_enc'].shape[0]
        config['model']['in_channels'] = [1+config['voxel grid']['channels']+num_ignore_channels] + config['model']['channels']
        config['model']['out_channels'] = config['model']['channels'] + [1+config['voxel grid']['channels']]
        config['model']['num_ignore_channels'] = num_ignore_channels
        model_kwargs = config['model'].copy()
        model_kwargs.pop('channels', None)
        model = RefineModel(**model_kwargs)
    if rank == 0:
        if(not os.path.exists('data/'+config['title'])):
            os.makedirs('data/'+config['title'])
        np.save('data/'+config['title']+'/model_config.npy', model_kwargs)
    return model