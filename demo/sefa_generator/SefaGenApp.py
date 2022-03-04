import sys
import torch
import tempfile
from demo.sefa_generator.latent_transformer import LatentTransformer
import dnnlib
import legacy
import functools

from PySide2.QtWidgets import QWidget, QPushButton, QVBoxLayout, QTextBrowser, QApplication
from PySide2.QtCore import QThreadPool
from PySide2.QtMultimedia import QSound

from pathlib import Path

from demo.utils.audio_file import AudioFile, write_audio_file
from demo.utils.path_utils import native_path_string

from demo.gui_elements.file_dialog_widget import FileDialogWidget
from demo.gui_elements.line_widget import HorLineWidget
from demo.gui_elements.slider_widget import SliderWidget
from demo.gui_elements.canvas_widget import CanvasWidget

from demo.sefa_generator.round_robin import RoundRobin
from demo.sefa_generator.interpolator import Interpolator

from demo.sefa_generator.drum_generator import DGWorker

k_app_title     = 'Sefa Drum Generator'
k_window_height = 400
k_window_width  = 150

k_tmp_file_path = 'tmp.wav'

k_num_transform_weights = 4

class SGApp(QWidget):
    def __init__(self):
        super(SGApp, self).__init__()

        self.choose_model_file_dialog   = FileDialogWidget  (title="Choose Model File",   
                                                            file_type_filter="Pickel Files (*.pkl)", 
                                                            default_filter = "Pickel Files (*.pkl)")

        self.choose_factorization_file_dialog = FileDialogWidget  (title="Choose Model File",   
                                                            file_type_filter="PyTorch File Format (*.pt)", 
                                                            default_filter = "PyTorch File Format (*.pt)")

        self.save_file_dialog           = FileDialogWidget( title="Save Output File",    
                                                            file_type_filter="Wav Files (*.wav)")

        self.round_robin_path_dialog    = FileDialogWidget( title="Save Round Robin Samples",    
                                                            file_type_filter="Wav Files (*.wav)")

        self.interpolate_path_dialog    = FileDialogWidget( title="Save Interpolated Samples",    
                                                            file_type_filter="Wav Files (*.wav)")

        self.model_file_path = self.choose_model_file_dialog.choose_open_file_path()
        with dnnlib.util.open_url(str(self.model_file_path)) as f:
            self.drum_model = legacy.load_network_pkl(f)['G_ema'].to('cpu')
        self.drum_model = self.drum_model.float()
        self.drum_model.forward = functools.partial(self.drum_model.forward, force_fp32=True)

        self.factorisation_file_path = self.choose_factorization_file_dialog.choose_open_file_path()
        self.latent_transformer = LatentTransformer(self.factorisation_file_path)
        self.latent_transformation_weights = torch.zeros(k_num_transform_weights, device="cpu", requires_grad=False)

        self.latent_dimension = self.drum_model.z_dim

        self.latent_vector = torch.zeros(1, self.latent_dimension, device="cpu", requires_grad=False)
        self.latent_vector_trans = torch.zeros(1, self.latent_dimension, device="cpu", requires_grad=False)

        self.pinned_latent_vector_1 = None
        self.pinned_latent_vector_2 = None

        self.save_file_path = None
        self.tmp_file_path = Path(tempfile.mkdtemp()) / k_tmp_file_path

        self.generated_audio_file = None

        self.threadpool = QThreadPool()


    def draw_window(self):
        self.window = QWidget()
        self.layout = QVBoxLayout()

        self.latent_space_canvas = CanvasWidget(width=self.latent_dimension, 
                                                height=100, 
                                                data_update_fctr=self.update_latent_vector,
                                                sync_fctr=self.sync_canvas)
        self.layout.addWidget(self.latent_space_canvas)
        self.latent_space_canvas.paint_data(self.latent_vector)

        self.button_generate_sample = QPushButton('Generate Drum Sample')
        self.button_generate_sample.clicked.connect(self.__on_generate_drum)
        self.layout.addWidget(self.button_generate_sample)

        self.layout.addWidget(HorLineWidget())

        self.button_generate_random_sample = QPushButton('Generate Random Drum Sample')
        self.button_generate_random_sample.clicked.connect(self.__on_generate_random_drum)
        self.layout.addWidget(self.button_generate_random_sample)

        self.layout.addWidget(HorLineWidget())

        self.button_playback = QPushButton('Play Sample!')
        self.button_playback.clicked.connect(self.__on_play_generated_drum)
        self.layout.addWidget(self.button_playback)

        # requires a generated audio file
        self.button_playback.setDisabled(True)

        self.button_save_to_disk = QPushButton('Save Generated Sample')
        self.button_save_to_disk.clicked.connect(self.__on_save_generated_sample)
        self.layout.addWidget(self.button_save_to_disk)
        # requires a generated audio file to be enabled
        self.button_save_to_disk.setDisabled(True)

        self.layout.addWidget(HorLineWidget())

        self.fade_in_slider = SliderWidget('Fade-in (ms)', 10, 0, 0.1, 1)
        self.layout.addWidget(self.fade_in_slider.get_slider_label_widget())
        self.layout.addWidget(self.fade_in_slider.get_slider_widget())        
        
        self.fade_out_slider = SliderWidget('Fade-out (ms)', 200, 0, 1, 100)
        self.layout.addWidget(self.fade_out_slider.get_slider_label_widget())
        self.layout.addWidget(self.fade_out_slider.get_slider_widget())

        self.offset_slider = SliderWidget('Offset (ms)', 20, 0, 0.1, 0)
        self.layout.addWidget(self.offset_slider.get_slider_label_widget())
        self.layout.addWidget(self.offset_slider.get_slider_widget())
        
        self.layout.addWidget(HorLineWidget())

        self.latent_transform_val_sliders = []
        for transform_weight_idx in range(k_num_transform_weights):
            self.latent_transform_val_sliders.append(SliderWidget(f'{transform_weight_idx}', 1, -1, 0.01, 0, vertical=False, position_idx=transform_weight_idx))
            self.layout.addWidget(self.latent_transform_val_sliders[transform_weight_idx].get_slider_label_widget())
            self.layout.addWidget(self.latent_transform_val_sliders[transform_weight_idx].get_slider_widget())

        # self.num_round_robin_slider = SliderWidget('Num Round Robin', 15, 2, 1, 10)
        # self.layout.addWidget(self.num_round_robin_slider.get_slider_label_widget())
        # self.layout.addWidget(self.num_round_robin_slider.get_slider_widget())

        # self.variation_round_robin_slider = SliderWidget('Round Robin Spread', 10, 1, 1, 2)
        # self.layout.addWidget(self.variation_round_robin_slider.get_slider_label_widget())
        # self.layout.addWidget(self.variation_round_robin_slider.get_slider_widget())

        # self.button_round_robin = QPushButton('Generate Round Robin')
        # self.button_round_robin.clicked.connect(self.__on_generate_round_robin)
        # self.layout.addWidget(self.button_round_robin)

        # self.layout.addWidget(HorLineWidget())

        # self.button_pin_latent_1 = QPushButton('Pin Sample 1')
        # self.button_pin_latent_1.clicked.connect(self.__on_pin_latent_1)
        # self.layout.addWidget(self.button_pin_latent_1)
        # self.button_pin_latent_2 = QPushButton('Pin Sample 2')
        # self.button_pin_latent_2.clicked.connect(self.__on_pin_latent_2)
        # self.layout.addWidget(self.button_pin_latent_2)

        # self.num_interpolation_steps_slider = SliderWidget('Num Interpolation Steps', 15, 2, 1, 10)
        # self.layout.addWidget(self.num_interpolation_steps_slider.get_slider_label_widget())
        # self.layout.addWidget(self.num_interpolation_steps_slider.get_slider_widget())

        # self.button_interpolation = QPushButton('Generate Sample Interpolation')
        # self.button_interpolation.clicked.connect(self.__on_generate_interpolated_samples)
        # self.layout.addWidget(self.button_interpolation)
        # self.button_interpolation.setDisabled(True)

        self.text_box = QTextBrowser()
        self.layout.addWidget(self.text_box)

        self.window.resize(k_window_height, k_window_width)
        self.window.setWindowTitle(k_app_title)
        self.window.setLayout(self.layout)
        self.window.show()


    def update_latent_transformation_weights(self):
        for weight_idx, slider in enumerate(self.latent_transform_val_sliders):
            self.latent_transformation_weights[weight_idx] = slider.get_slider_value()


    def update_latent_vector(self, idx, value):
        self.latent_vector[0, idx] = value


    def apply_latent_transformation(self):
        self.update_latent_transformation_weights()
        self.latent_vector_trans = self.latent_transformer.transform(self.latent_vector, self.latent_transformation_weights)


    def sync_canvas(self):
        self.latent_space_canvas.clear()
        self.latent_space_canvas.paint_data(self.latent_vector)


    def __on_pin_latent_1(self):
        self.pinned_latent_vector_1 = self.latent_vector
        self.__enable_interpolation()


    def __on_pin_latent_2(self):
        self.pinned_latent_vector_2 = self.latent_vector
        self.__enable_interpolation()


    def __enable_interpolation(self):
        if self.pinned_latent_vector_1 is not None and  self.pinned_latent_vector_2 is not None:
            self.button_interpolation.setDisabled(False)


    def __on_generate_interpolated_samples(self):
        self.interpolate_path_reference = self.interpolate_path_dialog.choose_save_file_path()
        interpolation_generator = Interpolator(self.drum_model, 
                                                self.interpolate_path_reference, 
                                                self.pinned_latent_vector_1, 
                                                self.pinned_latent_vector_2,
                                                self.num_interpolation_steps_slider.get_slider_value(),
                                                self.fade_in_slider.get_slider_value(),
                                                self.fade_out_slider.get_slider_value(),
                                                offset_ms=self.offset_slider.get_slider_value())

        interpolation_generator.signals.status_log.connect(self.__log_status)
        self.threadpool.start(interpolation_generator)


    def __on_generate_round_robin(self):
        self.round_robin_reference_path = self.round_robin_path_dialog.choose_save_file_path()
        round_robin_generator = RoundRobin(self.drum_model,
                                            self.latent_vector, 
                                            self.round_robin_reference_path, 
                                            num_round_robin=self.num_round_robin_slider.get_slider_value(),
                                            variation_factor=self.variation_round_robin_slider.get_slider_value(),
                                            fade_in_ms=self.fade_in_slider.get_slider_value(),
                                            fade_out_ms=self.fade_out_slider.get_slider_value(),
                                            offset_ms=self.offset_slider.get_slider_value())

        round_robin_generator.signals.status_log.connect(self.__log_status)
        self.threadpool.start(round_robin_generator)


    def __generate_rand_latent_vector(self):
        self.latent_vector = torch.randn(1, self.latent_dimension, device="cpu")
        self.latent_space_canvas.clear()
        self.latent_space_canvas.paint_data(self.latent_vector)


    def __on_generate_random_drum(self):
        self.__generate_rand_latent_vector()
        self.__on_generate_drum()


    def __on_generate_drum(self):
        self.text_box.clear()
        self.button_generate_random_sample.setDisabled(True)
        self.button_generate_sample.setDisabled(True)

        fade_in_ms = self.fade_in_slider.get_slider_value()
        fade_out_ms = self.fade_out_slider.get_slider_value()
        offset_ms = self.offset_slider.get_slider_value()

        self.apply_latent_transformation()

        drum_generator = DGWorker(  saved_model=self.drum_model, 
                                    latent_vector=self.latent_vector_trans, 
                                    fade_in_ms=fade_in_ms, 
                                    fade_out_ms=fade_out_ms,
                                    offset_ms=offset_ms)
        drum_generator.signals.generation_finished.connect(self.__on_generation_finished)
        drum_generator.signals.status_log.connect(self.__log_status)

        self.threadpool.start(drum_generator)


    def __on_play_generated_drum(self):
        write_audio_file(self.tmp_file_path, self.generated_audio_file)
        QSound.play(native_path_string(self.tmp_file_path))


    def __on_generation_finished(self, audio_file:AudioFile):
        self.__log_status('Drum Generated')

        self.generated_audio_file = audio_file

        self.button_generate_random_sample.setDisabled(False)
        self.button_generate_sample.setDisabled(False)
        self.button_save_to_disk.setDisabled(False)
        self.button_playback.setDisabled(False)

        self.__on_play_generated_drum()


    def __on_save_generated_sample(self):
        self.save_file_path = self.save_file_dialog.choose_save_file_path()
        write_audio_file(self.save_file_path, self.generated_audio_file)
        self.__log_status(f'Sample saved to : {self.save_file_path}')


    def __log_status(self, status):
        self.text_box.append(status)


def run():
    app = QApplication([])
    kgApp = SGApp()
    kgApp.draw_window()
    sys.exit(app.exec_())
