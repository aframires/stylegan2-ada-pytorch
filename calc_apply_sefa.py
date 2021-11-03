import argparse
import torch
import dnnlib
import legacy
import pickle
import numpy as np
import os
from tqdm import tqdm
from training.training_loop import spec_to_audio
import soundfile as sf


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        description='Discover semantics from the pre-trained weight.')
    parser.add_argument('ckpt', type=str,
                        help='Name to the pre-trained model.')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory to save the visualization pages. '
                             '(default: %(default)s)')
    parser.add_argument('-L', '--layer_idx', type=str, default='all',
                        help='Indices of layers to interpret. '
                             '(default: %(default)s)')
    parser.add_argument('-N', '--num_samples', type=int, default=5,
                        help='Number of samples used for visualization. '
                             '(default: %(default)s)')
    parser.add_argument('-K', '--num_semantics', type=int, default=5,
                        help='Number of semantic boundaries corresponding to '
                             'the top-k eigen values. (default: %(default)s)')
    parser.add_argument('--start_distance', type=float, default=-3.0,
                        help='Start point for manipulation on each semantic. '
                             '(default: %(default)s)')
    parser.add_argument('--end_distance', type=float, default=3.0,
                        help='Ending point for manipulation on each semantic. '
                             '(default: %(default)s)')
    parser.add_argument('--step', type=int, default=11,
                        help='Manipulation step on each semantic. '
                             '(default: %(default)s)')
    parser.add_argument('--truncation', type=float, default=0.7,
                        help='Psi factor used for truncation. This is '
                             'particularly applicable to StyleGAN (v1/v2). '
                             '(default: %(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for sampling. (default: %(default)s)')

    return parser.parse_args()

def parse_indices(obj, min_val=None, max_val=None):
    """Parses indices.

    The input can be a list or a tuple or a string, which is either a comma
    separated list of numbers 'a, b, c', or a dash separated range 'a - c'.
    Space in the string will be ignored.

    Args:
        obj: The input object to parse indices from.
        min_val: If not `None`, this function will check that all indices are
            equal to or larger than this value. (default: None)
        max_val: If not `None`, this function will check that all indices are
            equal to or smaller than this value. (default: None)

    Returns:
        A list of integers.

    Raises:
        If the input is invalid, i.e., neither a list or tuple, nor a string.
    """
    if obj is None or obj == '':
        indices = []
    elif isinstance(obj, int):
        indices = [obj]
    elif isinstance(obj, (list, tuple, np.ndarray)):
        indices = list(obj)
    elif isinstance(obj, str):
        indices = []
        splits = obj.replace(' ', '').split(',')
        for split in splits:
            numbers = list(map(int, split.split('-')))
            if len(numbers) == 1:
                indices.append(numbers[0])
            elif len(numbers) == 2:
                indices.extend(list(range(numbers[0], numbers[1] + 1)))
            else:
                raise ValueError(f'Unable to parse the input!')

    else:
        raise ValueError(f'Invalid type of input: `{type(obj)}`!')

    assert isinstance(indices, list)
    indices = sorted(list(set(indices)))
    for idx in indices:
        assert isinstance(idx, int)
        if min_val is not None:
            assert idx >= min_val, f'{idx} is smaller than min val `{min_val}`!'
        if max_val is not None:
            assert idx <= max_val, f'{idx} is larger than max val `{max_val}`!'

    return indices

def zs_to_ws(G,device,label,truncation_psi,zs):
    ws = []
    for z_idx, z in enumerate(zs):
        # z = torch.from_numpy(z).to(device) ###### VERIFY THIS WITH JORDAN
        w = G.mapping(z, label, truncation_psi=truncation_psi, truncation_cutoff=8)
        ws.append(w)
    return ws

def factorize_weights(model,layer_idx):

    modulate = {
        k[0]: k[1]
        for k in model.named_parameters()
        if "affine" in k[0] and "torgb" not in k[0] and "weight" in k[0] or ("torgb" in k[0] and "b4" in k[0] and "weight" in k[0] and "affine" in k[0])
    }

    if layer_idx == 'all':
        layers = list(range(len(modulate)))
    else:
        layers = parse_indices(layer_idx,
                                min_val=0,
                                max_val=len(modulate) - 1)

    idx = 0


    weight_mat_T = []
    for k, v in modulate.items():
        if idx in layers:
            tmp_w = v.T
            weight_mat_T.append(tmp_w.cpu().detach().numpy())
        idx=idx+1
    weight = np.concatenate(weight_mat_T, axis=1).astype(np.float32)
    weight = weight / np.linalg.norm(weight, axis=0, keepdims=True)
    eigen_values, eigen_vectors = np.linalg.eig(weight.dot(weight.T))

    torch.save({"ckpt": args.ckpt, "eigvec": eigen_vectors, "eigval": eigen_values}, "factorT.pt")


    return layers, eigen_vectors.T, eigen_values


if __name__ == "__main__":

    args = parse_args()
    G_kwargs = dnnlib.EasyDict()
    device = torch.device('cuda')
    
    save_dir = args.save_dir + '_trunc' + str(args.truncation) + '_layers' + args.layer_idx
    if not os.path.exists(args.save_dir):
      os.makedirs(args.save_dir)

    print('Loading networks from "%s"...' % args.ckpt)
    device = torch.device('cuda')
    with dnnlib.util.open_url(args.ckpt) as f:
        G = legacy.load_network_pkl(f, **G_kwargs)['G_ema'].to(device)

    label = torch.zeros([1, G.c_dim], device=device) # assume no class label
    noise_mode = "const" # default
    truncation_psi = args.truncation
    
    # Factorize weights.
    layers, boundaries, values = factorize_weights(G, args.layer_idx)


    # Set random seed.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Prepare codes.
    codes = torch.randn(args.num_samples, 1, G.z_dim).cuda()
    codes = zs_to_ws(G,device,label,truncation_psi,codes)

    distances = np.linspace(args.start_distance,args.end_distance, args.step)
    num_sam = args.num_samples
    num_sem = args.num_semantics

    for sam_id in tqdm(range(num_sam), desc='Sample ', leave=False):
        code = codes[sam_id]
        sample_folder = os.path.join(save_dir,'sample_'+str(sam_id))
        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder)
        for sem_id in tqdm(range(num_sem), desc='Semantic ', leave=False):
            boundary = boundaries[sem_id:sem_id + 1]
            semantic_folder = os.path.join(sample_folder,'semantic_'+str(sem_id))
            if not os.path.exists(semantic_folder):
                os.makedirs(semantic_folder)           
            for i,d in enumerate(distances, start=1):
                temp_code = code.cpu()
                temp_code[:, layers, :] += boundary * d
                audio = G.synthesis(temp_code.cuda(), noise_mode=noise_mode, force_fp32=True)
                audio = spec_to_audio(audio[0].cpu().numpy())
                filename = os.path.join(semantic_folder, str(i) + '_' + str(d) + '.wav')
                sf.write(filename, audio, 16000)

