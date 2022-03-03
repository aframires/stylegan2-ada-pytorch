from pathlib import Path
import torch
from typing import List

from PySide2.QtCore import QRunnable, QObject, Signal, Slot

from demo.utils.audio_file import AudioFile, write_audio_file
from demo.utils.path_utils import native_path_string

from demo.sefa_generator.drum_generator import DGBatchWorker

class KGSignals(QObject):
    status_log = Signal(str)

class Interpolator(QRunnable):
    def __init__(self,  saved_model: dict, 
                        file_path_reference: Path, 
                        latent_vector_1: torch.Tensor, 
                        latent_vector_2: torch.Tensor, 
                        num_steps: int=10, 
                        fade_in_ms: float = 0,
                        fade_out_ms: float = 0):
        super(Interpolator, self).__init__()

        self.saved_model            = saved_model
        self.file_path_reference    = file_path_reference
        self.latent_vector_1        = latent_vector_1
        self.latent_vector_2        = latent_vector_2
        self.num_steps              = num_steps
        self.fade_in_ms             = fade_in_ms
        self.fade_out_ms            = fade_out_ms

        self.signals = KGSignals()


    @Slot()
    def run(self):

        latent_vectors = self.__synthesize_interpolation()
        
        self.batch_generator = DGBatchWorker(self.saved_model, latent_vectors, self.fade_in_ms, self.fade_out_ms)
        self.batch_generator.signals.generation_finished.connect(self.__on_generation_finished)
        self.batch_generator.signals.status_log.connect(self.__log_status)

        self.batch_generator.run()


    def __synthesize_interpolation(self):
        diff_vec = self.latent_vector_2 - self.latent_vector_1
        step_vec = diff_vec / self.num_steps
        
        latent_vectors = [self.latent_vector_1.clone() + idx * step_vec.clone() for idx in range(self.num_steps)]

        return latent_vectors


    def __on_generation_finished(self, audio_files: List[AudioFile]):
        [self.__write_file_to_disk(audio_file, idx) for idx, audio_file in enumerate(audio_files)]
        self.__log_status(f'Interpolation Finished, files saved to {self.file_path_reference.parent.absolute()}')


    def __log_status(self, msg: str):
        self.signals.status_log.emit(f'interpolator: {msg}')


    def __write_file_to_disk(self, audio_file: AudioFile, idx: int):
        file_path_no_ext = native_path_string(self.file_path_reference.with_suffix(''))

        write_path = Path(file_path_no_ext + f'_{idx}').with_suffix(self.file_path_reference.suffix)
        write_audio_file(write_path, audio_file)