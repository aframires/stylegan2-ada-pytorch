import os
import subprocess
import shutil
import json
from pathlib import Path
from sys import platform

k_frechet_audio_distance_dir = 'google_research/frechet_audio_distance'
input_dir = 'generated_data'
out_dir = 'analysis_data'
dataset_embeddings = 'stats/dataset_embeddings'
k_frechet_bg_stats_filename = 'analysis_data/NI_background.cvs'
vggish_model_path = k_frechet_audio_distance_dir / Path('data/vggish_model.ckpt')


# ------------------------------------------------------------------------------------------------------------------------

def download_pretrained_model():
    # Download model files into a data directory
    if not vggish_model_path.is_file():
        subprocess.call(
            f'curl -o {vggish_model_path.parent.absolute()}/vggish_model.ckpt https://storage.googleapis.com/audioset/vggish_model.ckpt', shell=True)

# ------------------------------------------------------------------------------------------------------------------------

def create_file_list(input_files: list, output_file: Path):
    input_path_strings = [str(path.absolute()) for path in input_files]

    with open(str(output_file.absolute()), 'w') as csvfile:
        [csvfile.write(string + '\n') for string in input_path_strings]

# ------------------------------------------------------------------------------------------------------------------------

def create_embeddings(input_file_list: Path, stats_output: Path, input_sr: int, import_from_wav=True):
    import collections
    from google_research.frechet_audio_distance import create_embeddings_beam
    ModelConfig = collections.namedtuple(
        'ModelConfig', 'model_ckpt embedding_dim step_size')
    embedding_model = ModelConfig(model_ckpt=str((vggish_model_path).absolute()),
                                  embedding_dim=128,
                                  step_size=8000)

    pipeline = create_embeddings_beam.create_pipeline(files_input_list=str(input_file_list.absolute()),
                                                      embedding_model=embedding_model,
                                                      stats_output=str(
        stats_output.absolute()),
        import_and_process_hpy=not import_from_wav,
        input_sr=input_sr)

    result = pipeline.run()
    result.wait_until_finish()

# ------------------------------------------------------------------------------------------------------------------------

def compute_fad(background_stats: Path, test_stats: Path):
    from google_research.frechet_audio_distance import fad_utils
    mu_bg, sigma_bg = fad_utils.read_mean_and_covariances(
        str(background_stats.absolute()))
    mu_test, sigma_test = fad_utils.read_mean_and_covariances(
        str(test_stats.absolute()))
    fad = fad_utils.frechet_distance(mu_bg, sigma_bg, mu_test, sigma_test)
    return fad

# ------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    download_pretrained_model()

    os.makedirs(out_dir, exist_ok=True)
    runs = [f for f in os.scandir(input_dir) if f.is_dir()]

    fad_dict = {}

    for run in runs:
        run_analysis_path = os.path.join(out_dir, run.name)
        os.makedirs(run_analysis_path, exist_ok=True)
        epochs = [f for f in os.scandir(run.path) if f.is_dir()]
        
        for epoch in epochs:
            epoch_file_list = os.path.join(run_analysis_path, epoch.name + '.csv')
            create_file_list([Path(f) for f in Path(epoch.path).iterdir()], Path(epoch_file_list))
            stats_output = os.path.join(out_dir,'emb_' + run.name +'_' + epoch.name + '.csv')
            create_embeddings(Path(epoch_file_list),
                              stats_output=Path(stats_output),
                              input_sr=16000)

            fad = compute_fad(k_frechet_bg_stats_filename, stats_output)

            fad_dict[run.name + run.epoch] = fad
    
    with open('fad_dict.json', 'w') as fp:
        json.dump(dict, fp)

