# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import functools
import io
import json
import os
import pickle
import sys
import tarfile
import gzip
import zipfile
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import click
import numpy as np
import PIL.Image
from tqdm import tqdm
import librosa

#----------------------------------------------------------------------------

audio_length = 16000
sample_rate = 16000
hop_size = 512
win_size = 2048

#----------------------------------------------------------------------------

def error(msg):
    print('Error: ' + msg)
    sys.exit(1)

#----------------------------------------------------------------------------

def maybe_min(a: int, b: Optional[int]) -> int:
    if b is not None:
        return min(a, b)
    return a

#----------------------------------------------------------------------------

def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]

#----------------------------------------------------------------------------

def is_audio_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f'{ext}' in ['aac', 'au', 'flac', 'm4a', 'mp3', 'ogg', 'wav']

#----------------------------------------------------------------------------

def zeropad(signal):
    if len(signal) < audio_length:
        return np.append(
            signal, 
            np.zeros(audio_length - len(signal))
        )
    else:
        return signal

#----------------------------------------------------------------------------

def fade_out(x, percent=30.):
    """
        Applies fade out at the end of an audio vector
        x 
    """
    assert type(x) == np.ndarray, f"Fade_out: data type {type(x)} not {np.ndarray}"
    assert len(x.shape) == 1, f"Data has incompatible shape {x.shape}"

    fade_idx = int(x.shape[-1] * percent /100.)
    fade_curve = np.logspace(1, 0, fade_idx)
    fade_curve -= min(fade_curve)
    fade_curve /= max(fade_curve)
    x[-fade_idx:] *= fade_curve   
    return x

#----------------------------------------------------------------------------

def norm_audio(x):
    if max(x) != 0:
        return x/max(x)
    else:
        return x

#----------------------------------------------------------------------------

def stft(x):
    return librosa.core.stft(
            x,
            hop_length=hop_size,
            win_length=win_size,
            n_fft=win_size)

#----------------------------------------------------------------------------

def complex_to_lin(x):
    return np.stack((np.real(x), np.imag(x)))

#----------------------------------------------------------------------------

def remove_dc(spectrum):
        return spectrum[:, 1:, :]

#----------------------------------------------------------------------------

def process_audio(fname):
    #audio = np.array(PIL.Image.open(fname))
    audio = librosa.core.load(
            fname,
            sr=sample_rate,
            mono=True,
            offset=0.0,
            duration=audio_length / sample_rate,
            dtype=np.float32,
            res_type='kaiser_best')[0]
    
    audio = zeropad(audio)
    audio = fade_out(audio)
    audio = norm_audio(audio)
    spec = stft(audio)
    spec = complex_to_lin(spec)
    spec = remove_dc(spec)
    return spec

#----------------------------------------------------------------------------

def open_audio_folder(source_dir, *, max_images: Optional[int]):
    input_audios = [str(f) for f in sorted(Path(source_dir).rglob('*')) if is_audio_ext(f) and os.path.isfile(f)]

    # Load labels.
    labels = {}
    meta_fname = os.path.join(source_dir, 'dataset.json')
    if os.path.isfile(meta_fname):
        with open(meta_fname, 'r') as file:
            labels = json.load(file)['labels']
            if labels is not None:
                labels = { x[0]: x[1] for x in labels }
            else:
                labels = {}

    max_idx = maybe_min(len(input_audios), max_images)

    def iterate_audios():
        for idx, fname in enumerate(input_audios):
            arch_fname = os.path.relpath(fname, source_dir)
            arch_fname = arch_fname.replace('\\', '/')
            audio = process_audio(fname)
            yield dict(img=audio, label=labels.get(arch_fname))
            if idx >= max_idx-1:
                break
    return max_idx, iterate_audios()


#----------------------------------------------------------------------------

def open_dataset(source, *, max_images: Optional[int]):
    if os.path.isdir(source):
        return open_audio_folder(source, max_images=max_images)
    else:
        error(f'Missing input directory: {source}')

#----------------------------------------------------------------------------

def open_dest(dest: str) -> Tuple[str, Callable[[str, Union[bytes, str]], None], Callable[[], None]]:
    dest_ext = file_ext(dest)

    if dest_ext == 'zip':
        if os.path.dirname(dest) != '':
            os.makedirs(os.path.dirname(dest), exist_ok=True)
        zf = zipfile.ZipFile(file=dest, mode='w', compression=zipfile.ZIP_STORED)
        def zip_write_bytes(fname: str, data: Union[bytes, str]):
            zf.writestr(fname, data)
        return '', zip_write_bytes, zf.close
    else:
        # If the output folder already exists, check that is is
        # empty.
        #
        # Note: creating the output directory is not strictly
        # necessary as folder_write_bytes() also mkdirs, but it's better
        # to give an error message earlier in case the dest folder
        # somehow cannot be created.
        if os.path.isdir(dest) and len(os.listdir(dest)) != 0:
            error('--dest folder must be empty')
        os.makedirs(dest, exist_ok=True)

        def folder_write_bytes(fname: str, data: Union[bytes, str]):
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            with open(fname, 'wb') as fout:
                if isinstance(data, str):
                    data = data.encode('utf8')
                fout.write(data)
        return dest, folder_write_bytes, lambda: None

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--source', help='Directory or archive name for input dataset', required=True, metavar='PATH')
@click.option('--dest', help='Output directory or archive name for output dataset', required=True, metavar='PATH')
@click.option('--max-images', help='Output only up to `max-images` images', type=int, default=None)
@click.option('--resize-filter', help='Filter to use when resizing images for output resolution', type=click.Choice(['box', 'lanczos']), default='lanczos', show_default=True)
@click.option('--transform', help='Input crop/resize mode', type=click.Choice(['center-crop', 'center-crop-wide']))
@click.option('--width', help='Output width', type=int)
@click.option('--height', help='Output height', type=int)
def convert_dataset(
    ctx: click.Context,
    source: str,
    dest: str,
    max_images: Optional[int],
    transform: Optional[str],
    resize_filter: str,
    width: Optional[int],
    height: Optional[int]
):
    """Convert an image dataset into a dataset archive usable with StyleGAN2 ADA PyTorch.

    The input dataset format is guessed from the --source argument:

    \b
    --source *_lmdb/                    Load LSUN dataset
    --source cifar-10-python.tar.gz     Load CIFAR-10 dataset
    --source train-images-idx3-ubyte.gz Load MNIST dataset
    --source path/                      Recursively load all images from path/
    --source dataset.zip                Recursively load all images from dataset.zip

    Specifying the output format and path:

    \b
    --dest /path/to/dir                 Save output files under /path/to/dir
    --dest /path/to/dataset.zip         Save output files into /path/to/dataset.zip

    The output dataset format can be either an image folder or an uncompressed zip archive.
    Zip archives makes it easier to move datasets around file servers and clusters, and may
    offer better training performance on network file systems.

    Images within the dataset archive will be stored as uncompressed PNG.
    Uncompresed PNGs can be efficiently decoded in the training loop.

    Class labels are stored in a file called 'dataset.json' that is stored at the
    dataset root folder.  This file has the following structure:

    \b
    {
        "labels": [
            ["00000/img00000000.png",6],
            ["00000/img00000001.png",9],
            ... repeated for every image in the datase
            ["00049/img00049999.png",1]
        ]
    }

    If the 'dataset.json' file cannot be found, the dataset is interpreted as
    not containing class labels.

    Image scale/crop and resolution requirements:

    Output images must be square-shaped and they must all have the same power-of-two
    dimensions.

    To scale arbitrary input image size to a specific width and height, use the
    --width and --height options.  Output resolution will be either the original
    input resolution (if --width/--height was not specified) or the one specified with
    --width/height.

    Use the --transform=center-crop or --transform=center-crop-wide options to apply a
    center crop transform on the input image.  These options should be used with the
    --width and --height options.  For example:

    \b
    python dataset_tool.py --source LSUN/raw/cat_lmdb --dest /tmp/lsun_cat \\
        --transform=center-crop-wide --width 512 --height=384
    """

    if dest == '':
        ctx.fail('--dest output filename or directory must not be an empty string')

    num_files, input_iter = open_dataset(source, max_images=max_images)
    archive_root_dir, save_bytes, close_dest = open_dest(dest)

    dataset_attrs = None

    labels = []
    for idx, audio in tqdm(enumerate(input_iter), total=num_files):
        idx_str = f'{idx:08d}'
        archive_fname = f'{idx_str[:5]}/audio{idx_str}.npy'
        ad = audio['img']
        # Transform may drop images.
        if ad is None:
            continue

        # Error check to require uniform image attributes across
        # the whole dataset.
        channels = ad.shape[0]
        cur_image_attrs = {
            'width': ad.shape[2],
            'height': ad.shape[1],
            'channels': channels
        }
        if dataset_attrs is None:
            dataset_attrs = cur_image_attrs
            width = dataset_attrs['width']
            height = dataset_attrs['height']
            #if width != height:
            #    error(f'Image dimensions after scale and crop are required to be square.  Got {width}x{height}')
            #if dataset_attrs['channels'] not in [1, 3]:
            #    error('Input images must be stored as RGB or grayscale')
            #if width != 2 ** int(np.floor(np.log2(width))):
            #    error('Image width/height after scale and crop are required to be power-of-two')
        elif dataset_attrs != cur_image_attrs:
            err = [f'  dataset {k}/cur image {k}: {dataset_attrs[k]}/{cur_image_attrs[k]}' for k in dataset_attrs.keys()]
            error(f'Image {archive_fname} attributes must be equal across all images of the dataset.  Got:\n' + '\n'.join(err))

        # Save the image as an uncompressed .npy
        f_path = os.path.join(archive_root_dir, archive_fname)
        os.makedirs(os.path.dirname(f_path), exist_ok=True)
        np.save(f_path,ad)
        labels.append([archive_fname, audio['label']] if audio['label'] is not None else None)

    metadata = {
        'labels': labels if all(x is not None for x in labels) else None
    }
    save_bytes(os.path.join(archive_root_dir, 'dataset.json'), json.dumps(metadata))
    close_dest()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    convert_dataset() # pylint: disable=no-value-for-parameter
