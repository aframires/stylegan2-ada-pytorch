from argparse import ArgumentParser
from pathlib import Path
from subprocess import call


def get_args():
    parser = ArgumentParser(description='Configure training environment for StyleGAN2-ADA')

    parser.add_argument('--train_config'
                        , type=str
                        , required=True
                        , help='')

    # optional arguments
    parser.add_argument('--data_prep'
                        , type=int
                        , default=0
                        , help='')

    parser.add_argument('--bucket_audio_data_dir'
                        , type=str
                        , default='sound-similarity/ni-samples-drums'
                        , help='')

    parser.add_argument('--bucket_dataset_dir'
                        , type=str
                        , default='sound-similarity/stylegan2-dataset-antonio'
                        , help='')

    parser.add_argument('--train'
                        , type=int
                        , default=1
                        , help='')

    parser.add_argument('--gpus'
                        , type=int
                        , default=1
                        , help='')

    parser.add_argument('--augmentation'
                        , type=int
                        , default=0
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

    parser.add_argument('--resume_path'
                        , type=str
                        , default=''
                        , help='gs://...')

    return parser.parse_args()


def install_dependencies():
    call('sh bootstrap/bootstrap.sh'.split(' '))


def copy_audio_data_from_bucket(source_dir: str, dest_dir: str):
    if not Path(dest_dir).exists():
        Path(dest_dir).mkdir()

    call(f'gsutil -m cp -r gs://{source_dir}/* {dest_dir}'.split(' '))


def render_audio_data_to_dataset(audio_data_dir: str, dataset_dir: str):
    call(f'python3.8 dataset_tool_audio.py --source {audio_data_dir} --dest {dataset_dir}'.split(' '))


def copy_dataset_from_bucket(source_dir: str, dest_dir: str):
    if not Path(dest_dir).exists():
        Path(dest_dir).mkdir()

    call(f'gsutil -m cp -r gs://{source_dir}/* {dest_dir}'.split(' '))


def train_model(dataset_dir: str, training_data_out_dir: str, num_gpus: int, train_config: str, resume_path: str, aug: bool = False):
    Path('setup_log.log').touch()
    training_log = open('training_log.log', 'w')

    training_cmd = f'python3.8 train.py --outdir {training_data_out_dir} --data {dataset_dir} --gpus {num_gpus} --cfg {train_config}'
    if (aug):
        training_cmd += ' --aug ada --augpipe bg'

    if not resume_path == '':
        Path.mkdir('./model_archive')
        model_name = Path(f'{resume_path}').parts[-1]
        call(f'gsutil cp -r {resume_path} ./model_archive')
        training_cmd += f' --resume ./model_archive/{model_name}'

    call(f'tmux new -d -s {train_config}_training'.split(' '), stdout=training_log, stderr=training_log)
    call(['tmux', 'send-keys', '-t', f'{train_config}_training', f'{training_cmd}', 'Enter'], stdout=training_log, stderr=training_log)


if __name__ == "__main__":

    setup_args = get_args()

    install_dependencies()

    call('sleep 20'.split(' '))

    if setup_args.data_prep:
        copy_audio_data_from_bucket(setup_args.bucket_audio_data_dir, setup_args.local_audio_data_dir)
        render_audio_data_to_dataset(setup_args.local_audio_data_dir, setup_args.local_dataset_dir)

    else:
        copy_dataset_from_bucket(setup_args.bucket_dataset_dir, setup_args.local_dataset_dir)

    if setup_args.train:        

        config = None
        try:
            config = setup_args.train_config        
        except:
            print('Need to specify a traing config! Please run script again.')
            exit()

        train_model(setup_args.local_dataset_dir, setup_args.local_saved_training_data_dir, setup_args.gpus, config, setup_args.resume_path, setup_args.augmentation == 1)
