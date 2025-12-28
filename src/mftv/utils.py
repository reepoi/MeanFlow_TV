import logging
from pathlib import Path


DIR_ROOT = (Path(__file__).parent/'..'/'..').resolve()
HYDRA_INIT = dict(version_base=None, config_path='../../conf', config_name='training_and_evaluation')

# generated using f'0x{secrets.randbits(128):x}'
RNG_RANDBITS = dict(
    DATASET={
        # train
        2376999025: 0xbb2631e48c09b45de657855db13b873b,
        # test
    },
)


def filename_relative_to_dir_root(filename):
    return Path(filename).relative_to(DIR_ROOT)


def getLoggerByFilename(filename):
    return logging.getLogger(str(filename_relative_to_dir_root(filename)))
