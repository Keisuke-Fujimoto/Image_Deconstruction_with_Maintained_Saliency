import os

import cv2
import numpy as np
import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm
from tensorboardX import SummaryWriter

from data_loader import DataLoader
from utils import *
from models import *

# Device
device = 'cuda'

# Hyper Parameter Settings
batch_size = 80
lr = 0.0003
num_epoch = 1200
scale = 4
z_S_ch = 8

lambda_Adv = 0.1
lambda_features = 1


H = int(48/scale)
W = int(64/scale)

# Discriminator for MapGenerator
D_map = Discriminator(in_channels=4, scale=scale)
D_map.apply(weights_init_normal)
D_map.to(device)
for param in D_map.parameters():
    param.requires_grad_()

# Generator for Saliency Maps
G_map = MapGenerator(z_S_ch=z_S_ch)
G_map.apply(weights_init_normal)
G_map.to(device)
for param in G_map.parameters():
    param.requires_grad_()

# Loss Criterion
MSE = nn.MSELoss()
BCE = nn.BCELoss()
L1 = nn.L1Loss()

# Optimizer
optimizer_D = torch.optim.Adam(D_map.parameters(), lr=lr)
optimizer_G = torch.optim.Adam(G_map.parameters(), lr=lr)

# DataLoader
dataloader = DataLoader(batch_size)
num_batch = dataloader.num_batches  # length of data / batch_size

# Directory Setting
NAME = os.path.basename(__file__).split(os.sep)[-1].split('.')[0]
DIR_TO_SAVE = './runs/{}'.format(NAME)
if not os.path.exists(DIR_TO_SAVE):
    os.makedirs(DIR_TO_SAVE)

# For Log
writer = SummaryWriter(logdir='./runs/__{}'.format(NAME))
counter = 0

for current_epoch in range(1,num_epoch+1):
    for idx in tqdm(range(num_batch)):
        # Get Batch
        (batch_img, batch_map) = dataloader.get_batch()
        batch_img = downsample(batch_img.to(device), scale)
        batch_map = downsample(batch_map.to(device), scale)

        # Labels for Adversarial Loss
        real_labels = (torch.ones(batch_size, H, W) - (torch.rand(batch_size, H, W) * 0.30)).to(device)
        fake_labels = (torch.zeros(batch_size, H, W) + (torch.rand(batch_size, H, W) * 0.30)).to(device)

        # Estimate Saliency Maps
        fake_map = G_map(batch_img)


        # Train Discriminator
        optimizer_D.zero_grad()

        # Real Loss
        inp = torch.cat((batch_img, batch_map), 1)
        real_outputs = D_map(inp).squeeze()

        loss_D_real = BCE(real_outputs, real_labels)
        real_score = real_outputs.data.mean()   # For Log

        # Fake Loss
        inp = torch.cat((batch_img, fake_map.detach()), 1)
        fake_outputs = D_map(inp).squeeze()

        loss_D_fake = BCE(fake_outputs, fake_labels)

        # Total Loss
        loss_D = lambda_Adv * 0.5 * (loss_D_real + loss_D_fake)

        # Backward and Optimization
        loss_D.backward()
        optimizer_D.step()


        # Train Generator for Saliency Maps
        optimizer_G.zero_grad()

        # Adversarial Loss
        inp = torch.cat((batch_img, fake_map), 1)
        fake_outputs = D_map(inp).squeeze()
        fake_features = D_map.features

        loss_Adv = BCE(fake_outputs, real_labels)
        fake_score = fake_outputs.data.mean()   # For Log

        # Features Loss
        inp = torch.cat((batch_img, batch_map), 1)
        real_outputs = D_map(inp).squeeze()
        real_features = D_map.features

        loss_features = sum([MSE(real_feature, fake_features[i]) for i, real_feature in enumerate(real_features)])

        # Total Loss
        loss_G = lambda_Adv * loss_Adv + lambda_features * loss_features
        loss_G.backward()
        optimizer_G.step()


        # Log
        with torch.no_grad():
            # Scalars
            writer.add_scalars('data/score_mean', {'real_score':real_score.item()}, counter)
            writer.add_scalars('data/score_mean', {'fake_score':fake_score.item()}, counter)
            writer.add_scalars('data/loss', {'loss_features':loss_features}, counter)

            loss_BCE_train = BCE(fake_map, batch_map)
            writer.add_scalars('data/loss', {'loss_BCE_train':loss_BCE_train}, counter)

            # Validation
            (val_img, val_map) = dataloader.get_val()
            val_img = downsample(val_img.to(device), scale)
            val_map = downsample(val_map.to(device), scale)

            val_fake_map = G_map(val_img)
            loss_BCE_val = BCE(val_fake_map, val_map)
            writer.add_scalars('data/loss', {'loss_BCE_val':loss_BCE_val}, counter)

            # Display
            grid_batch_img = torchvision.utils.make_grid(batch_img[:5], nrow=1, padding=1, normalize=True)
            grid_batch_map = torchvision.utils.make_grid(batch_map[:5], nrow=1, padding=1, normalize=True)
            grid_fake_map = torchvision.utils.make_grid(fake_map[:5], nrow=1, padding=1, normalize=True)

            grid_val_img = torchvision.utils.make_grid(val_img[:5], nrow=1, padding=1, normalize=True)
            grid_val_map = torchvision.utils.make_grid(val_map[:5], nrow=1, padding=1, normalize=True)
            grid_val_fake_map = torchvision.utils.make_grid(val_fake_map[:5], nrow=1, padding=1, normalize=True)

            grid_batch_map = convert_3ch(grid_batch_map)
            grid_fake_map = convert_3ch(grid_fake_map)
            grid_val_map = convert_3ch(grid_val_map)
            grid_val_fake_map = convert_3ch(grid_val_fake_map)

            results = torch.cat((grid_batch_img.unsqueeze(0), grid_batch_map, grid_fake_map, grid_val_img.unsqueeze(0), grid_val_map, grid_val_fake_map), 0)
            results = torchvision.utils.make_grid(results, nrow=6, padding=0, normalize=True)

            writer.add_image('Results', BGR2RGB(upsample(results.unsqueeze(0), 2).squeeze(0)), counter)
        
        counter += 1
        
    # Save weights every 3 epoch
    if (current_epoch + 1) % 3 == 0:
        print('pkl saved')
        print('Epoch [{}/{}]'.format(current_epoch, num_epoch))
        torch.save(G_map.state_dict(), ''.join([DIR_TO_SAVE, '/G_map.pkl']))
torch.save(G_map.state_dict(), ''.join([DIR_TO_SAVE, '/G_map.pkl']))
writer.close()
print('Done')