import os

import cv2
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from model import *


import PIL
from PIL import Image
import numpy as np
import einops
from tqdm import tqdm

BATCH_SIZE = 512
N_EPOCHS = 500


def show_forward_sample(img_or, t):
    ddpm = DDPM('cpu',n_steps=100)
    img_or.show()

    image_list = []

    concatenated_image = np.array(img_or)

    trans = transforms.ToPILImage()
    trans_tensor = transforms.ToTensor()
    for i in range(1, t+1):
        img = img_or
        img = ddpm.sample_forward(trans_tensor(img), i).squeeze()
        img = np.array(img)
        concatenated_image = np.concatenate([concatenated_image, img], axis=1)

    concatenated_image = Image.fromarray(concatenated_image)
    concatenated_image.show()

def download_dataset():
    mnist = torchvision.datasets.MNIST(root='./data/mnist', download=True)
    print("Length of Mnist: ", len(mnist))   # 1x28x28 [0, 1]

    # 用于观察逐渐加噪的过程
    # id = 4
    # img, label = mnist[id]
    #
    # show_forward_sample(img, 5)


def get_dataloader(batch_size: int):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x - 0.5) * 2)
    ])
    dataset = torchvision.datasets.MNIST(root='./data/mnist', transform=transform)
    # dataset = Faces(transform=transform)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)


class Faces(Dataset):
    def __init__(self, root='./faces', transform=None):
        self.images_list = [os.path.join(root, path) for path in os.listdir(root)]
        self.transform = transform

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image = Image.open(self.images_list[idx])
        image = self.transform(image)

        return image, idx



class DDPM():

    def __init__(self,
                 device,
                 n_steps: int,
                 min_beta: float=1e-4,
                 max_beta: float=0.02):
        betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        alphas = 1 - betas
        alphas_bar = torch.empty_like(alphas)
        product = 1

        # 生成每一步的 α_hat
        for i, alpha in enumerate(alphas):
            product *= alpha
            alphas_bar[i] = product

        self.betas = betas
        self.n_steps = n_steps
        self.alphas = alphas
        self.alphas_bar = alphas_bar

    def sample_forward(self, x, t, eps=None):
        alpha_bar = self.alphas_bar[t].reshape(-1, 1, 1, 1)

        # 从正态分布中采样
        if eps is None:
            eps = torch.randn_like(x)

        res = eps * torch.sqrt(1 - alpha_bar) + torch.sqrt(alpha_bar) * x

        return res

    def sample_backward(self, img_shape, net, device, simple_var=True):
        x = torch.randn(img_shape).to(device)  # img_shape (N, C, H ,W)
        net = net.to(device)

        for t in range(self.n_steps - 1, -1, -1):
            x = self.sample_backward_step(x, t, net, simple_var)
        return x

    def sample_backward_step(self, x_t, t, net, simple_var=True):
        n = x_t.shape[0]
        t_tensor = torch.tensor([t] * n, dtype=torch.long).to(x_t.device).unsqueeze(1)
        eps = net(x_t, t_tensor)

        if t == 0:
            noise = 0
        else:
            if simple_var:
                var = self.betas[t]
            else:
                var = (1 - self.alphas_bar[t - 1]) / (1 - self.alphas_bar[t]) * self.betas[t]

            noise = torch.randn_like(x_t)
            noise *= torch.sqrt(var)

        x_mean = (x_t - (1 - self.alphas[t]) / torch.sqrt(1 - self.alphas_bar[t]) * eps) / torch.sqrt(self.alphas[t])

        x_t = x_mean + noise

        return x_t  # 上一步的预测图


def train(ddpm: DDPM, net, device, ckpt_path):
    n_steps = ddpm.n_steps
    dataloader = get_dataloader(BATCH_SIZE)
    net = net.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    for epoch in range(N_EPOCHS):

        net.train()
        bar = tqdm(dataloader)

        for step, (x, _) in enumerate(bar, 1):
            current_batch_szie = x.shape[0]
            x = x.to(device)

            t = torch.randint(0, n_steps, (current_batch_szie, )).to(device)
            eps = torch.randn_like(x).to(device)
            x_t = ddpm.sample_forward(x, t, eps)
            eps_theta = net(x_t, t.reshape(current_batch_szie, 1))

            loss = criterion(eps_theta, eps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                bar.set_postfix_str(f'loss={loss / step}')

    torch.save(net.state_dict(), 'model.ckpt')


def sample_imgs(ddpm,
                model,
                output_path,
                n_sample=81,
                device='cuda',
                simple_var=True):

    model.load_state_dict(torch.load('model.ckpt'))

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        shape = (n_sample, 1, 28, 28)
        imgs = ddpm.sample_backward(shape,
                                    model,
                                    device=device,
                                    simple_var=simple_var).detach().cpu()

        imgs = (imgs + 1) / 2 * 255
        imgs = imgs.clamp(0, 255)
        imgs = einops.rearrange(
            imgs,
            '(b1 b2) c h w -> (b1 h) (b2 w) c',
            b1 = int(n_sample ** 0.5)
        )

        imgs = imgs.numpy().astype(np.uint8)

        cv2.imwrite(output_path, imgs)



if __name__ == '__main__':
    # download_dataset()
    n_steps = 1000
    config_id = 4
    device = 'cuda'
    model_path = 'model.ckpt'

    config = unet_res_cfg
    net = build_network(config, n_steps)
    ddpm = DDPM(device, n_steps)

    train(ddpm, net, device, ckpt_path=model_path)

    sample_imgs(ddpm, net, 'images.jpg')













