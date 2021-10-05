import os
import subprocess
import shutil
import json
from pathlib import Path
from sys import platform

k_frechet_audio_distance_dir = 'google_research/frechet_audio_distance'
bg_data_path = '../ni-samples-drums-16k/'
input_dir = 'generated_frechet_audio'
out_dir = 'analysis_data'
bg_emb = os.path.join(out_dir, 'emb_bg.csv')
vggish_model_path = k_frechet_audio_distance_dir / \
    Path('data/vggish_model.ckpt')


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
                                                      stats_output=str(stats_output.absolute()))

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


def compute_bg_fad():

    bg_file_list = os.path.join(out_dir, 'bg_file_list.csv')
    create_file_list([Path(f)
                     for f in Path(bg_data_path).iterdir()], Path(bg_file_list))
    create_embeddings(Path(bg_file_list),
                      stats_output=Path(bg_emb),
                      input_sr=16000)

# ------------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':

    download_pretrained_model()

    os.makedirs(out_dir, exist_ok=True)
    fad_out_dir = os.path.join(out_dir, 'fad_results')
    os.makedirs(fad_out_dir, exist_ok=True)
    runs = [f for f in os.scandir(input_dir) if f.is_dir()]

    #compute_bg_fad()

    fad_dict = {}

    for run in runs:
        run_analysis_path = os.path.join(out_dir, run.name)
        os.makedirs(run_analysis_path, exist_ok=True)
        epochs = [f for f in os.scandir(run.path) if f.is_dir()]

        for epoch in epochs:
            epoch_file_list = os.path.join(
                run_analysis_path, epoch.name + '.csv')
            create_file_list([Path(f) for f in Path(
                epoch.path).iterdir()], Path(epoch_file_list))
            stats_output = os.path.join(
                out_dir, 'emb_' + run.name + '_' + epoch.name + '.csv')
            create_embeddings(Path(epoch_file_list),
                              stats_output=Path(stats_output),
                              input_sr=16000)

            fad = compute_fad(Path(bg_emb), Path(stats_output))

            fad_dict[run.name + epoch.name] = fad
            with open(os.path.join(fad_out_dir, run.name + '_' + epoch.name + '.txt'), 'w') as output:
                output.write(str(fad))

    with open('fad_dict.json', 'w') as fp:
        json.dump(dict, fp)
