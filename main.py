import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import tqdm

from dataset import get_dataset
from models.diffusion_model import DiffusionModel
from config import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def init_directory():
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    if not os.path.exists('./logs/training'):
        os.mkdir('./logs/training')
    if not os.path.exists('./logs/training/results'):
        os.mkdir('./logs/training/results')
    if not os.path.exists('./logs/training/checkpoints'):
        os.mkdir('./logs/training/checkpoints')


@torch.no_grad()
def sample_plot_signal(model, save_path, T, epoch):
    # Sample noise
    model.eval()
    img = torch.randn((1, 1, SAMPLES), device=device)
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    num_images = 10
    stepsize = int(T / num_images)

    for i in range(0, T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        label = torch.randint(0, 4, (1,)).to(device)
        img = model.sample_timestep(img.type(torch.FloatTensor).to(device), t, label)
        if i % stepsize == 0:
            plt.subplot(num_images, 1, int(i / stepsize + 1))
            plt.plot(img[0][0].cpu().detach().numpy())
    plt.savefig(os.path.join(save_path, f'epoch_{epoch+1}'), bbox_inches='tight', pad_inches=0.0)
    plt.close()


def train():
    data_loader = get_dataset()

    T = MODEL_CONFIG['T']
    model = DiffusionModel(**MODEL_CONFIG).to(device)
    print("Num params:", sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(), lr=HYPER_PARAMETERS_CONFIG['learning_rate'])
    epochs = HYPER_PARAMETERS_CONFIG['epochs']
    loss_fn = F.l1_loss if HYPER_PARAMETERS_CONFIG['type_loss'] == 'l1' else F.mse_loss

    for epoch in range(epochs):
        print(f'---------------- Epoch {epoch + 1}/{epochs} ----------------')
        running_loss = 0
        for step, data in enumerate(tqdm.tqdm(data_loader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')):
            # clear gradient
            optimizer.zero_grad()

            # prepare data
            img = data['waveform']
            label = data['label']
            t = torch.randint(0, T, (img.shape[0],), device=device).long()

            # Forward pass
            noise, noise_pred = model(img.to(device), t, label.to(device))

            # calculate loss
            loss = loss_fn(noise, noise_pred)
            running_loss += loss.item()

            # Backprop and optimize
            loss.backward()
            optimizer.step()

        print(f'[Epoch {epoch + 1}] Loss: {running_loss / len(data_loader)}')
        sample_plot_signal(model, save_logs_path, T, epoch)
        if epoch >= HYPER_PARAMETERS_CONFIG['save_from_epoch']:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(save_ckpt_path, f'epoch_{epoch + 1}.pth'))


if __name__ == '__main__':
    init_directory()

    PREFIX = 'MITDB-SINUS'
    save_logs_path = f'logs/training/results/{PREFIX}'
    save_ckpt_path = f'logs/training/checkpoints/{PREFIX}'
    if not os.path.exists(save_logs_path):
        os.mkdir(save_logs_path)
    if not os.path.exists(save_ckpt_path):
        os.mkdir(save_ckpt_path)

    train()
