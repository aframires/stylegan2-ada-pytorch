import torch
from typing import List

from PySide2.QtCore import QRunnable, QObject, Signal, Slot

from utils.audio_file import AudioFile

class KGSignals(QObject):
    generation_finished = Signal(AudioFile)
    status_log = Signal(str)

k_model_generator_key       = 'generator_smoothed'
k_model_input_mapping_key   = 'input_mapping_smoothed'
k_sample_rate_key           = 'sample_rate'
k_latent_dim_key            = 'latent_dim'
k_styleALAE_z_dim_key       = 'z_dim'
k_model_name_key            = 'name'
k_model_name_styleALAE      = 'StyleALAE'
k_model_name_proGAN         = 'ProgressiveDCGAN'

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

def get_model_name(model: dict):
    # to do: save model name explicitly in state dict
    checkpoint_name = model[k_model_name_key].stem 
    if k_model_name_styleALAE in checkpoint_name:
        return k_model_name_styleALAE
    elif k_model_name_proGAN in checkpoint_name:
        return k_model_name_proGAN
    else:
        raise NotImplementedError()

def get_model_latent_dim(model: dict):
    if get_model_name(model) == k_model_name_styleALAE:
        return model[k_styleALAE_z_dim_key]
    else:
        return model[k_latent_dim_key]

class KGWorker(QRunnable):
    def __init__(self, saved_model: dict, latent_vector: torch.Tensor, fade_in_ms: float = None, fade_out_ms: float = None, offset_ms: float = None):
        super(KGWorker, self).__init__()
        self.kick_generator = saved_model[k_model_generator_key].eval()
        
        self.model_name = get_model_name(saved_model)

        if self.model_name == k_model_name_styleALAE:
            self.kick_input_mapping = saved_model[k_model_input_mapping_key].eval()

        self.sample_rate = int(saved_model[k_sample_rate_key].numpy())
        self.latent_vector = latent_vector
        self.signals = KGSignals()
        self.fade_in_ms = fade_in_ms if fade_in_ms > 0 else None
        self.fade_out_ms = fade_out_ms if fade_out_ms > 0 else None
        self.offset_ms = offset_ms if offset_ms > 0 else None


    @Slot()
    def run(self):
        self.signals.status_log.emit('Generating Kick Sample')        

        if self.model_name == k_model_name_styleALAE:
            # hack, for now: InputMappingNetwork.forward() breaks with a batch size of 1 so we repeat the latent vector to get a batch size of 2
            # TO DO: fix there instead
            latent_vector = self.latent_vector.repeat(2,1,1)
            output_audio_data = self.kick_generator(self.kick_input_mapping(latent_vector)).detach().numpy()
        else:
            output_audio_data = self.kick_generator(self.latent_vector).detach().numpy()

        # drop batch dim
        output_audio_data = output_audio_data[0,:,:]

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


class KGBatchWorker(QRunnable):
    def __init__(self, saved_model: dict, latent_vectors: List[torch.Tensor], fade_in_ms: float = None, fade_out_ms: float = None, offset_ms: float = None):
        super(KGBatchWorker, self).__init__()

        self.model_name = get_model_name(saved_model)
        if self.model_name == k_model_name_styleALAE:
            self.kick_input_mapping = saved_model[k_model_input_mapping_key].eval()
        self.kick_generator = saved_model[k_model_generator_key].eval()
        self.sample_rate = int(saved_model[k_sample_rate_key].numpy())
        self.latent_vectors = latent_vectors
        self.signals = KGSignals()
        self.fade_in_ms = fade_in_ms if fade_in_ms > 0 else None
        self.fade_out_ms = fade_out_ms if fade_out_ms > 0 else None
        self.offset_ms = offset_ms if offset_ms > 0 else None


    @Slot()
    def run(self):
        output_audio_files = []
        for idx, latent_vector in enumerate(self.latent_vectors):

            self.signals.status_log.emit(f'Generating Kick Sample {idx + 1}')

            if self.model_name == k_model_name_styleALAE:
                output_audio_data = self.kick_generator(self.kick_input_mapping(latent_vector)).detach().numpy()
            else:
                output_audio_data = self.kick_generator(latent_vector).detach().numpy()

            # drop batch dim
            output_audio_data = output_audio_data[0,:,:]

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
