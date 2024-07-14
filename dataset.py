import os

import torch
import numpy as np
import pandas as pd
import json
import wfdb

from config import *


class ECGDataset:
    def __init__(self, length, dataset):
        self.length = length
        self.dataset = dataset
        self.waveform_dir = DATA_CONFIG['waveform_dir']

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        waveform, _ = wfdb.rdsamp(os.path.join(self.waveform_dir, str(self.dataset['FileName'][idx])))
        waveform = waveform[self.dataset['SampleMark1'][idx]:self.dataset['SampleMark2'][idx],
                   self.dataset['Channel'][idx]][None, :]
        label = self.dataset['Labels'][idx]
        sample = {
            'waveform': torch.from_numpy(waveform).type(torch.FloatTensor),
            'label': torch.tensor(int(label)).type(torch.LongTensor)
        }

        return sample


def get_dataset():
    df = pd.read_csv(DATA_CONFIG['info_path'])
    dataset = ECGDataset(length=len(df), dataset=df)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=HYPER_PARAMETERS_CONFIG['batch_size'],
                                              shuffle=True)

    return data_loader
