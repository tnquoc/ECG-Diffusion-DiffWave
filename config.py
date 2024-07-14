FREQUENCY = 250  # Hz
TIME_DURATION = 10  # seconds
SAMPLES = FREQUENCY * TIME_DURATION

MODEL_CONFIG = {
    'beta_1': 1e-4,   # do not change if not necessary
    'beta_2': 0.02,   # do not change if not necessary
    'T': 200
}

DATA_CONFIG = {
    'waveform_dir': 'data/mitdb',
    'info_path': 'data/mitdb_dataset.csv',
}

HYPER_PARAMETERS_CONFIG = {
    "learning_rate": 0.0001,
    "batch_size": 16,
    "epochs": 600,
    "type_loss": 'l2',
    "save_from_epoch": 200
}

TEST_CKPT_PATH = "/path_to_checkpoint"
