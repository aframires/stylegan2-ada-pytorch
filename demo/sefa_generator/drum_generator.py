import torch
from typing import List

from training.training_loop import spec_to_audio

from PySide2.QtCore import QRunnable, QObject, Signal, Slot

from demo.utils.audio_file import AudioFile

class DGSignals(QObject):
    generation_finished = Signal(AudioFile)
    status_log = Signal(str)

k_model_name                = 'StyleGAN2'
k_sample_rate               = 44100 # TODO what sample rate is the model running at ?


def format_dim_for_channels(audio_data):
    if len(audio_data.shape) == 1:
        from numpy import expand_dims
        return expand_dims(audio_data, 0)

def apply_s_curve(input, amount = 1.0):
    from numpy import exp
    def sigmoid(x):
        return 1 / (1 + exp(-x))
    a = 6
    output = sigmoid( 2 * a * (input - 0.5) )
    offset = sigmoid(-a)
    output = (output - offset) / (1 - 2*offset)
    return output * amount + input * (1-amount)

def compute_fade_in(fade_in_samples: int, total_samples: int):
    from numpy import linspace, expand_dims, pad
    fade_in = linspace(start=0.0, stop=1.0, num=max([1, fade_in_samples]))
    fade_in = pad(fade_in, (0, total_samples-fade_in_samples), 'constant', constant_values=(0, 1))
    fade_in = expand_dims(fade_in, 0)
    return fade_in

def compute_fade_out(fade_out_samples, total_samples):
    from numpy import linspace, expand_dims, pad
    fade_out = linspace(start=1.0, stop=0.0, num=max([1, fade_out_samples]))
    fade_out = pad(fade_out, (total_samples-fade_out_samples, 0), 'constant', constant_values=(1, 0))
    fade_out = expand_dims(fade_out, 0)
    return fade_out

def get_model_name():
    return "StyleGAN2"

class DGWorker(QRunnable):
    def __init__(self, saved_model: dict, latent_vector: torch.Tensor, fade_in_ms: float = None, fade_out_ms: float = None, offset_ms: float = None):
        super(DGWorker, self).__init__()

        self.drum_generator = saved_model.eval()
        
        self.model_name = get_model_name()

        self.sample_rate = k_sample_rate 
        self.latent_dimension = self.drum_generator.z_dim

        self.latent_vector = latent_vector

        self.signals = DGSignals()
        self.fade_in_ms = fade_in_ms if fade_in_ms > 0 else None
        self.fade_out_ms = fade_out_ms if fade_out_ms > 0 else None
        self.offset_ms = offset_ms if offset_ms > 0 else None


    @Slot()
    def run(self):
        self.signals.status_log.emit('Generating Kick Sample')        

        output_audio_data = format_dim_for_channels(self.generate_audio())

        # apply fade-in
        if self.fade_in_ms is not None:
            fade_in_samples = round(self.fade_in_ms * 1e-3 * self.sample_rate)
            total_samples = output_audio_data.shape[1]
            output_audio_data *= compute_fade_in(fade_in_samples, total_samples)

        # apply fade-out
        if self.fade_out_ms is not None:
            fade_out_samples = round(self.fade_out_ms * 1e-3 * self.sample_rate)
            total_samples = output_audio_data.shape[1]
            output_audio_data *= apply_s_curve(compute_fade_out(fade_out_samples, total_samples))

        if self.offset_ms is not None:
            from numpy import roll
            offset_smp = round(self.offset_ms * 1e-3 * self.sample_rate)
            output_audio_data = roll(output_audio_data, offset_smp)

        output_audio_file = AudioFile(audio_data=output_audio_data,  sample_rate=self.sample_rate, num_channels=output_audio_data.shape[0], num_frames=output_audio_data.shape[1])

        self.signals.generation_finished.emit(output_audio_file)


    def generate_audio(self, truncation_psi=1):
        class_idx = None
        noise_mode = 'const'

        # Labels.
        label = torch.zeros([1, self.drum_generator.c_dim], device='cpu')
        if self.drum_generator.c_dim != 0:
            if class_idx is None:
                print('Must specify class label with --class when using a conditional network')
            label[:, class_idx] = 1
        else:
            if class_idx is not None:
                print ('warn: --class=lbl ignored when running on an unconditional network')

        spectrogram = self.drum_generator(self.latent_vector, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        return spec_to_audio(spectrogram[0].numpy())


class DGBatchWorker(QRunnable):
    def __init__(self, saved_model: dict, latent_vectors: List[torch.Tensor], fade_in_ms: float = None, fade_out_ms: float = None, offset_ms: float = None):
        super(DGBatchWorker, self).__init__()

        self.drum_generator = saved_model.eval()
        self.model_name = get_model_name()

        self.sample_rate = k_sample_rate 
        self.latent_dimension = self.drum_generator.z_dim

        self.latent_vectors = latent_vectors

        self.signals = DGSignals()
        self.fade_in_ms = fade_in_ms if fade_in_ms > 0 else None
        self.fade_out_ms = fade_out_ms if fade_out_ms > 0 else None
        self.offset_ms = offset_ms if offset_ms > 0 else None


    @Slot()
    def run(self):
        output_audio_files = []
        for idx, latent_vector in enumerate(self.latent_vectors):

            self.signals.status_log.emit(f'Generating Kick Sample {idx + 1}')

            output_audio_data = format_dim_for_channels(self.generate_audio(latent_vector))

            # apply fade-in
            if self.fade_in_ms is not None:
                fade_in_samples = round(self.fade_in_ms * 1e-3 * self.sample_rate)
                total_samples = output_audio_data.shape[1]
                output_audio_data *= compute_fade_in(fade_in_samples, total_samples)

            # apply fade-out
            if self.fade_out_ms is not None:
                fade_out_samples = round(self.fade_out_ms * 1e-3 * self.sample_rate)
                total_samples = output_audio_data.shape[1]
                output_audio_data *= apply_s_curve(compute_fade_out(fade_out_samples, total_samples))

            if self.offset_ms is not None:
                from numpy import roll
                offset_smp = round(self.offset_ms * 1e-3 * self.sample_rate)
                output_audio_data = roll(output_audio_data, offset_smp)

            output_audio_files.append(AudioFile(audio_data=output_audio_data,  sample_rate=self.sample_rate, num_channels=output_audio_data.shape[0], num_frames=output_audio_data.shape[1]))


        self.signals.generation_finished.emit(output_audio_files)


    def generate_audio(self, latent_vector, truncation_psi=1):
        class_idx = None
        noise_mode = 'const'

        # Labels.
        label = torch.zeros([1, self.drum_generator.c_dim], device='cpu')
        if self.drum_generator.c_dim != 0:
            if class_idx is None:
                print('Must specify class label with --class when using a conditional network')
            label[:, class_idx] = 1
        else:
            if class_idx is not None:
                print ('warn: --class=lbl ignored when running on an unconditional network')

        spectrogram = self.drum_generator(latent_vector, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        return spec_to_audio(spectrogram[0].numpy())
