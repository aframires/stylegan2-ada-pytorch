import os
from dataset_tool_audio import zeropad, fade_out, norm_audio

import librosa
import soundfile as sf
import numpy as np

sr_in = 44100
sr_out = 16000
audio_length = 16000

filetype = '.wav'


def process_file(fname, sample_rate):
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

    return audio

def save_audio(fn,data):
    sf.write(fn, data, 16000, subtype='PCM_16')

def run(dir_in,dir_out,sr_in,sr_out):
    
    flns = [x for x in os.listdir(dir_in) if x.endswith(filetype)]

    for file in flns:
        audio = process_file(os.path.join(dir_in,file), sr_out)
        save_audio(os.path.join(dir_out, file), audio)





dir_in = "/home/aframires/code/NI/ni-samples-drums"
dir_out = "/home/aframires/code/NI/ni-samples-drums-16k"

run(dir_in, dir_out, sr_in, sr_out)
