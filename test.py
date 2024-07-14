import os
import time
import argparse
from datetime import datetime

import matplotlib.pyplot as plt
import torch

from models.diffusion_model import DiffusionModel
from config import *


@torch.no_grad()
def sample_signal(model, label):
    # Sample noise
    model.eval()
    img = torch.randn((1, 1, length_samples), device=device)
    plt.figure(figsize=(16, 3), constrained_layout=True)
    plt.axis('off')

    for i in range(0, MODEL_CONFIG['T'])[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = model.sample_timestep(img.type(torch.FloatTensor).to(device), t, label)

    return img[0][0].cpu().detach().numpy()


if __name__ == '__main__':
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_samples', type=int, default=16, help='Number of samples to generate')
    parser.add_argument('--time_duration', type=int, default=10, help='Length of sample in seconds')
    args = parser.parse_args()
    test_samples = args.test_samples
    length_samples = FREQUENCY * args.time_duration

    # load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DiffusionModel(**MODEL_CONFIG).to(device)
    checkpoint = torch.load(TEST_CKPT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # random type generate
    label = torch.randint(0, 4, (1,)).to(device)
    # label = torch.tensor(0).to(device)  # choose type in [0, 1, 2, 3]

    # init directories
    base_test_results = 'test_results'
    current_time = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    save_test_results_path = f'{base_test_results}/test_{current_time}'
    if not os.path.exists(base_test_results):
        os.mkdir(base_test_results)
    if not os.path.exists(save_test_results_path):
        os.mkdir(save_test_results_path)

    # generate and save results
    print(f'Save results to {save_test_results_path}')
    for i in range(test_samples):
        t0 = time.time()
        test_id = i + 1
        img_path = os.path.join(save_test_results_path, f'{test_id}.png')
        sig = sample_signal(model, label)
        fig = plt.gcf()
        fig.set_size_inches(25, 2.56)
        plt.plot(sig)
        plt.axis('off')
        plt.savefig(img_path, bbox_inches='tight', pad_inches=0.0, dpi=100)
        print('Generated sample {:d} in {:.2f}s'.format(test_id, time.time() - t0))
