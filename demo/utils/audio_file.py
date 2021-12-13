import numpy as np
import collections
import librosa.core as audio_reader
import soundfile    as audio_writer

from pathlib import Path

from utils.path_utils import native_path_string

AudioFile = collections.namedtuple('AudioFile', 'audio_data sample_rate num_channels num_frames')

def get_dimensions(data):
    assert data.ndim <= 2 

    if (len(data.shape) == 1):
        return (1, int(data.shape[0]))

    return (int(data.shape[0]), int(data.shape[1]))


def reshape_mono_audio_data(data):
    return np.reshape(data,(1,-1))


def read_audio_file(audio_file_path: Path, target_sample_rate=22050):
    
    try:
        data, sr = audio_reader.load(audio_file_path, target_sample_rate)
    except ValueError as err:
        return AudioFile(audio_data=None, sample_rate=0, num_channels=0, num_frames=0)
    
    nchan, nframes = get_dimensions(data)
    if nchan == 1:
        data = reshape_mono_audio_data(data)

    return AudioFile(audio_data=data, sample_rate=sr, num_channels=nchan, num_frames=nframes)


def write_audio_file(file_path: Path, audio_file: AudioFile, sample_rate=None):
    if sample_rate is None:
        sample_rate = audio_file.sample_rate

    if audio_file.num_channels == 1 and len(audio_file.audio_data.shape) == 2:
        data = audio_file.audio_data.reshape((audio_file.num_frames,))
        audio_writer.write(native_path_string(file_path), data, int(sample_rate))
        return

    audio_writer.write(native_path_string(file_path), audio_file.audio_data, int(sample_rate))