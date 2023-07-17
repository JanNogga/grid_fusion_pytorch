import os
import torch
import torch.nn as nn
import numpy as np
import io
from PIL import Image
from colorama import Fore
import seaborn as sns
from model import PositionalEncoder
import itertools


def getCMap(num_colors):
    # Get jet color map from Matlab
    cmap = sns.hls_palette(num_colors)
    cmap = np.array(cmap)
    
    return list([list(item) for item in (255*cmap).astype(np.uint8)])

# applies colors to a semseg map
# adapted from https://poutyne.org/examples/semantic_segmentation.html
def target_to_color(array, palette):
    pil_out = Image.fromarray(array.cpu().numpy().astype(np.uint8), mode='P')
    pil_out.putpalette(palette)
    pil_out = pil_out.convert('RGB')
    return torch.from_numpy(np.asarray(pil_out))

# prints the loaded configuration, highlighting changes and additions
def print_config(d,ref, indent=1):
    for key, value in d.items():
        print('  ' * indent + str(key).ljust(25)[:25],end="\r\n" if isinstance(value, dict) else "")
        rk = ref.get(key)
        if isinstance(value, dict):
            if rk!=None:
                print_config(value, rk, indent + 1)
            else:
                print_config(value,{}, indent + 1)
        else:
            if rk!=None:
                if rk ==value:
                    print('  ' * (indent + 1) + Fore.GREEN+str(value)+Fore.RESET)
                else:
                    print('  ' * (indent + 1) + Fore.RED+str(value)+" (Diff)"+Fore.RESET)
            else:
                print('  ' * (indent + 1) + Fore.CYAN+str(value)+" (New)"+Fore.RESET)

def is_better(a, best, delta, patience, mode='min'):
    if patience == 0:
        return True
    if mode == 'min':
        return a < best - delta
    else:
        return a > best + delta

# can be used to trigger early stopping in a training process
class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=7):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0

    def step(self, metrics):
        if self.patience == 0:
            return False
        if self.best is None:
            self.best = metrics
            return False

        if is_better(metrics, self.best, self.min_delta, self.patience, self.mode):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True
        return False

def metric_to_gpu(criterion, gpu):
    metric_name, metric = criterion
    return (metric_name, metric.cuda(gpu))

def epoch_average(losses, counts):
    losses_np = np.array(losses)
    counts_np = np.array(counts)
    weighted_losses = losses_np * counts_np
    return weighted_losses.sum()/counts_np.sum()

def save_history(history, description, title, suffix='_best'):
    if(not os.path.exists(f"data/{title}/trainer_logs")):
        os.makedirs(f"data/{title}/trainer_logs")
    savepath = f"data/{title}/trainer_logs/{description}{suffix}.npy"
    np.save(savepath, history)
    return savepath

def save_model(model, optimizer, description, title, suffix='_best'):
    if(not os.path.exists(f"data/{title}/models")):
        os.makedirs(f"data/{title}/models")
    savepath = f"data/{title}/models/{description}{suffix}.pt"
    torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    }, savepath)
    return savepath

def format_wandb_config(config_dict):
    ret = { key : value for key, value in config_dict.items() if key not in ('_wandb')}
    ret = { key : ret[key]["value"] for key, _ in ret.items()}
    return ret

def set_encoder_requires_grad(model, requires_grad=False):
    for param in model.encoder.parameters():
        param.requires_grad = requires_grad
    return None

def add_dropout(model):
    for i in range(len(model.decoder.blocks)):
        tmp = [model.decoder.blocks[i].conv2]
        tmp = [nn.Dropout2d(p=0.5)] + tmp
        model.decoder.blocks[i].conv2 = nn.Sequential(*tmp)
    return None

def set_dropout_p(model, p=0.5):
    for i in range(len(model.decoder.blocks)):
        model.decoder.blocks[i].conv2[0].p = p
    return None

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    return img


# +
def generate_pos_enc(voxel_grid_config):
    grid_encoder = PositionalEncoder(3, voxel_grid_config['num_freqs'])
    with torch.no_grad():
        linspaces = [torch.linspace(voxel_grid_config['range_min'][i], voxel_grid_config['range_max'][i], voxel_grid_config['world_size'][i]) for i in range(3)]
        coords = torch.cartesian_prod(*linspaces).view(*voxel_grid_config['world_size'],3)
        voxel_grid_config['pos_enc'] = grid_encoder(coords).permute(3,0,1,2)
        return voxel_grid_config
        

def complete_config(config, dataset, rank):
    config['dataset']['num_classes'] = dataset.num_classes
    config['voxel grid']['range_min'] = dataset.range_min
    config['voxel grid']['range_max'] = dataset.range_max
    axis_range = config['voxel grid']['range_max'][:3] - config['voxel grid']['range_min'][:3]
    axis_range /= torch.min(axis_range)
    config['voxel grid']['world_size'] = (torch.ones(3)*config['voxel grid']['voxel_base_num']*axis_range).long()
    config['voxel grid']['world_size'] -= torch.remainder(config['voxel grid']['world_size'], 4)
    config['voxel grid']['channels'] = dataset.num_classes
    config['voxel grid']['voxel_size'] = (axis_range/config['voxel grid']['world_size'].float()).mean()
    config['voxel grid']['anchor'] = (config['voxel grid']['range_max'][:3] + config['voxel grid']['range_min'][:3]) / 2
    config['voxel grid']['pos_enc'] = None
    if rank == 0:
        if(not os.path.exists('data/'+config['title'])):
            os.makedirs('data/'+config['title'])
        np.save('data/'+config['title']+'/voxel_grid_config.npy', config['voxel grid'])
    if config['voxel grid']['use_pos_enc']:
        config['voxel grid'] = generate_pos_enc(config['voxel grid'])
    return config


# -

def get_sample(dataloader, sample_idx, batch_size):
    batch_idx = int(np.floor(sample_idx/batch_size))
    return sample_idx % batch_size, next(itertools.islice(dataloader, batch_idx, None))


def unbatch_minibatch(idx, mini_batch):
    pcd, semseg, cam_pose, depth, cam_k, gt = mini_batch
    ret_pcd = [item[idx:idx+1] for item in pcd]
    return ret_pcd, semseg[idx:idx+1], cam_pose[idx:idx+1], depth[idx:idx+1], cam_k[idx:idx+1], gt[idx:idx+1]