from pathlib import Path
import torch
from typing import List

from PySide2.QtCore import QRunnable, QObject, Signal, Slot

from demo.utils.audio_file import AudioFile, write_audio_file
from demo.utils.path_utils import native_path_string

from demo.sefa_generator.drum_generator import KGBatchWorker

k_variation_eps = 1e-2

class KGSignals(QObject):
    status_log = Signal(str)


class RoundRobin(QRunnable):
    def __init__(self,  saved_model: dict, 
                        reference_latent_vector: torch.Tensor, 
                        file_path_reference: Path, 
                        num_round_robin: int, 
                        variation_factor: float=2,
                        fade_in_ms: float = 0,
                        fade_out_ms: float = 0):
        super(RoundRobin, self).__init__()

        self.saved_model                = saved_model
        self.reference_latent_vector    = reference_latent_vector
        self.file_path_reference        = file_path_reference
        self.num_round_robin            = num_round_robin
        self.variation_eps              = variation_factor * k_variation_eps
        self.fade_in_ms                 = fade_in_ms
        self.fade_out_ms                = fade_out_ms

        self.signals = KGSignals()


    @Slot()
    def run(self):

        latent_vectors = self.__synthesize_round_robin()

        self.kick_generator = KGBatchWorker(self.saved_model, latent_vectors, self.fade_in_ms, self.fade_out_ms)
        self.kick_generator.signals.generation_finished.connect(self.__on_generation_finished)
        self.kick_generator.signals.status_log.connect(self.__log_status)

        self.kick_generator.run()


    def __synthesize_round_robin(self):
        latent_vectors = []
        [latent_vectors.append(self.reference_latent_vector.clone() + 2 * (torch.rand_like(self.reference_latent_vector) - torch.tensor(0.5)) * self.variation_eps ) for _ in range(self.num_round_robin)]
        return latent_vectors


    def __on_generation_finished(self, audio_files: List[AudioFile]):
        [self.__write_file_to_disk(audio_file, idx) for idx, audio_file in enumerate(audio_files)]
        self.__log_status(f'Round Robin Generation Finished, files saved to {self.file_path_reference.parent.absolute()}')


    def __log_status(self, msg: str):
        self.signals.status_log.emit(f'round_robin: {msg}')


    def __write_file_to_disk(self, audio_file: AudioFile, idx: int):
        file_path_no_ext = native_path_string(self.file_path_reference.with_suffix(''))

        write_path = Path(file_path_no_ext + f'_{idx}').with_suffix(self.file_path_reference.suffix)
        write_audio_file(write_path, audio_file)
