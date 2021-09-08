from argparse import ArgumentParser
from pathlib import Path
from subprocess import call

def get_args():
    parser = ArgumentParser(description='Configure training environment for StyleGAN2-ADA')

    parser.add_argument('--bucket_audio_data_dir'
                        , type=str
                        , default=''
                        , help='')

    parser.add_argument('--local_audio_data_dir'
                        , type=str
                        , default=''
                        , help='')

    parser.add_argument('--local_dataset_dir'
                        , type=str
                        , default=''
                        , help='')

    parser.add_argument('--train_config'
                        , type=str
                        , default='drumgan'
                        , help='')

    return parser.parse_args()


def install_dependencies():
    call("sh bootstrap/bootstrap.sh".split(" "))


def copy_audio_data_from_bucket(source_dir: str, dest_dir: str):
    if not Path(dest_dir).exists():
        Path.mkdir(dest_dir)
        
    call(f"gsutil -m cp -r gs://{source_dir}/* {dest_dir}".split(" "))


def render_audio_data_to_dataset(audio_data_dir: str, dataset_dir: str):
    call(f"dataset_tool_audio.py --source {audio_data_dir} --dest {dataset_dir}".split(" "))


def train_model(dataset_dir: str, training_data_out_dir: str, train_config: str):
    call(f"python3.8 train.py --outdir {training_data_out_dir} --data {dataset_dir} --gpus 1 --cfg {train_config}".split(" "))


if __name__ == "__main__":

    setup_args = get_args()

    install_dependencies()
    copy_audio_data_from_bucket(setup_args.bucket_audio_data_dir, setup_args.local_audio_data_dir)
    render_audio_data_to_dataset(setup_args.local_audio_data_dir, setup_args.local_dataset_dir)
    train_model(setup_args.local_dataset_dir, "saved_training_data/", setup_args.train_config)