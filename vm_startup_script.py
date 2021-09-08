from argparse import ArgumentParser
from pathlib import Path
from subprocess import call


def get_args():
    parser = ArgumentParser(description='Configure training environment for StyleGAN2-ADA')

    parser.add_argument('--bucket_audio_data_dir'
                        , type=str
                        , required=True
                        , help='')

    parser.add_argument('--train_config'
                        , type=str
                        , required=True
                        , help='')

    # optional arguments
    parser.add_argument('--data_prep'
                    , type=int
                    , default=1
                    , required=True
                    , help='')

    parser.add_argument('--train'
                , type=int
                , default=1
                , required=True
                , help='')

    parser.add_argument('--local_audio_data_dir'
                        , type=str
                        , default='audio/'
                        , help='')

    parser.add_argument('--local_dataset_dir'
                        , type=str
                        , default='dataset/'
                        , help='')

    parser.add_argument('--local_saved_training_data_dir'
                        , type=str
                        , default='saved_training_data/'
                        , help='')


    return parser.parse_args()


def install_dependencies():
    call('sh bootstrap/bootstrap.sh'.split(' '))


def copy_audio_data_from_bucket(source_dir: str, dest_dir: str):
    if not Path(dest_dir).exists():
        Path(dest_dir).mkdir()
        
    call(f'gsutil -m cp -r gs://{source_dir}/* {dest_dir}'.split(' '))


def render_audio_data_to_dataset(audio_data_dir: str, dataset_dir: str):
    call(f'python3.8 dataset_tool_audio.py --source {audio_data_dir} --dest {dataset_dir}'.split(' '))


def train_model(dataset_dir: str, training_data_out_dir: str, train_config: str):
    Path('setup_log.log').touch()
    training_log = open('training_log.log', 'w')

    training_cmd = f'python3.8 train.py --outdir {training_data_out_dir} --data {dataset_dir} --gpus 1 --cfg {train_config}'

    call(f'tmux new -d -s {train_config}_training'.split(' '), stdout=training_log, stderr=training_log)
    call(['tmux', 'send-keys', '-t', f'{train_config}_training', f'{training_cmd}', 'Enter'], stdout=training_log, stderr=training_log)


if __name__ == "__main__":

    setup_args = get_args()

    install_dependencies()

    if setup_args.data_prep:
        copy_audio_data_from_bucket(setup_args.bucket_audio_data_dir, setup_args.local_audio_data_dir)
        render_audio_data_to_dataset(setup_args.local_audio_data_dir, setup_args.local_dataset_dir)

    if setup_args.train:
        train_model(setup_args.local_dataset_dir, setup_args.local_saved_training_data_dir, setup_args.train_config)