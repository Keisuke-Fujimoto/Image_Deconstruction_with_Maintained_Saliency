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
batch_size = 40
lr = 0.0003
num_epoch = 120

scale = 4
latent_dim = 8
z_S_ch = 8

lambda_Adv_recon = 0.1
lambda_Adv_ss = 0.01
lambda_features = 1

lambda_kl = 10
lambda_KL_reg = 0.2
lambda_latent = 0.05
lambda_map = 0.1


H_imgs = int(48/scale)
W_imgs = int(64/scale)

H_kl = int(H_imgs/4)
W_kl = int(W_imgs/4)

# Discriminator for Reconstruction
D_R = Discriminator(in_channels=3, scale=scale)
D_R.apply(weights_init_normal)
D_R.to(device)
for param in D_R.parameters():
    param.requires_grad_()

# Discriminator for Stochastic Sampling
D_SS = Discriminator(in_channels=3, scale=scale)
D_SS.apply(weights_init_normal)
D_SS.to(device)
for param in D_SS.parameters():
    param.requires_grad_()

# Generator for Saliency Maps
G_map = MapGenerator(z_S_ch=z_S_ch)
pretrained_dict = torch.load('./Weights/G_map/G_map.pkl')
G_map.load_state_dict(pretrained_dict)
G_map.to(device)
for param in G_map.parameters():
    param.requires_grad = False
    

# Generator for Images
G = Decoder(3, latent_dim, z_S_ch, scale)
G.apply(weights_init_normal)
G.to(device)
for param in G.parameters():
    param.requires_grad_()

# Encoder for Images
E = Encoder(3, latent_dim, scale)
E.apply(weights_init_normal)
E.to(device)
for param in E.parameters():
    param.requires_grad_()

# Loss Crierion
MSE = nn.MSELoss()
BCE = nn.BCELoss()
L1 = nn.L1Loss()

# Optimizer
optimizer_D_R = torch.optim.Adam(D_R.parameters(), lr=lr)
optimizer_D_SS = torch.optim.Adam(D_SS.parameters(), lr=lr)
optimizer_G = torch.optim.Adam(G.parameters(), lr=lr)
optimizer_E = torch.optim.Adam(E.parameters(), lr=lr)

# DataLoader
dataloader = DataLoader(batch_size)
num_batch = dataloader.num_batches  # length of data / batch_size

# Directory Setting
NAME = os.path.basename(__file__).split(os.sep)[-1].split('.')[0]
DIR_TO_SAVE = './runs/{}'.format(NAME)
if not os.path.exists(DIR_TO_SAVE):
    os.makedirs(DIR_TO_SAVE)

# Example Settings
example = cv2.imread("COCO_val2014_000000143859.png")
example_map = cv2.imread("COCOValidationSampleMap.png")

to_tensor = transforms.ToTensor() # Transforms 0-255 numbers to 0 - 1.0.
pilTrans = torchvision.transforms.ToPILImage()

example_img = to_tensor(example)
example_img = example_img.unsqueeze(0).to(device)
example_img = downsample(example_img, scale)

example_map = pilTrans(example_map).convert('L')
example_map = to_tensor(example_map).unsqueeze(0).to(device)
example_map = downsample(example_map, scale)

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
        real_labels = (torch.ones(batch_size, H_imgs, W_imgs) - (torch.rand(batch_size, H_imgs, W_imgs) * 0.30)).to(device)
        fake_labels = (torch.zeros(batch_size, H_imgs, W_imgs) + (torch.rand(batch_size, H_imgs, W_imgs) * 0.30)).to(device)

        # Estimate Saliency Maps and Get the middle layer of G_map
        fake_map = G_map(batch_img)
        z_S = G_map.z_S

        # Reconstruct the Original Images
        mu, logvar = E(batch_img)
        encoded_z = reparameterize(mu, logvar)
        img_recon = G(encoded_z, z_S.detach())

        # Stochastic Sampling
        sampled_z = torch.randn(mu.size(0), latent_dim, H_imgs, W_imgs).to(device)
        img_ss = G(sampled_z, z_S.detach())


        # Train Discriminator for Reconstruction
        optimizer_D_R.zero_grad()

        # Real Loss
        real_outputs = D_R(batch_img)
        loss_D_R_real = BCE(real_outputs.squeeze(), real_labels)
        real_score_recon = real_outputs.data.mean()   # For Log

        # Fake Loss
        fake_outputs = D_R(img_recon.detach())
        loss_D_R_fake = BCE(fake_outputs.squeeze(), fake_labels)

        # Total Loss
        loss_D_R = lambda_Adv_recon * 0.5 * (loss_D_R_real + loss_D_R_fake)

        # Backward and Optimization
        loss_D_R.backward()
        optimizer_D_R.step()


        # Train Discriminator for Stochastic Sampling
        optimizer_D_SS.zero_grad()

        # Real Loss
        real_outputs = D_SS(batch_img)
        loss_D_SS_real = BCE(real_outputs.squeeze(), real_labels)
        real_score_ss = real_outputs.data.mean()   # For Log

        # Fake Loss
        fake_outputs = D_SS(img_ss.detach())
        loss_D_SS_fake = BCE(fake_outputs.squeeze(), fake_labels)

        # Total Loss
        loss_D_SS = lambda_Adv_ss * 0.5 * (loss_D_SS_real + loss_D_SS_fake)

        # Backward and Optimization
        loss_D_SS.backward()
        optimizer_D_SS.step()


        # Train Encoder and Generator for Images
        optimizer_E.zero_grad()
        optimizer_G.zero_grad()

        # Adversarial Loss for Reconstruction
        outputs = D_R(img_recon)
        fake_features = D_R.features

        loss_Adv_recon = BCE(outputs.squeeze(), real_labels)
        fake_score_recon = outputs.data.mean()    # For Log
        
        # Features Loss
        D_R(batch_img)
        real_features = D_R.features

        loss_features = sum([MSE(real_feature, fake_features[i]) for i, real_feature in enumerate(real_features)])

        # Adversarial Loss for Stochastic Sampling
        outputs = D_SS(img_ss)
        loss_ss_GAN = BCE(outputs.squeeze(), real_labels)
        fake_score_ss = outputs.data.mean()    # For Log

        # KL Loss for batch
        batch_mean = mu.mean(0)
        batch_var = mu.var(0) + logvar.exp().mean(0)
        loss_KL_batch = KLD(batch_mean, batch_var)

        # Additional Regularization Term for KL Divergence
        mu_reg = logvar.exp().mean(0)
        var_reg = mu.var(0)
        loss_KL_reg = KLD(mu_reg, var_reg, 1-lambda_KL_reg, lambda_KL_reg)

        # KL Loss
        loss_KL = loss_KL_batch.mean() + loss_KL_reg.mean()

        # Total Loss for Encoder and Generator
        loss_GE = lambda_kl * loss_KL
        loss_GE += lambda_features * loss_features + lambda_Adv_recon * loss_Adv_recon
        loss_GE.backward(retain_graph=True)
        optimizer_E.step()

        # Latent loss
        mu_ss, _ = E(img_ss)
        loss_latent = MSE(sampled_z, mu_ss)

        # Saliency Map Loss
        map_ss = G_map(img_ss)
        loss_map = BCE(map_ss, fake_map)

        # Loss for only Generator
        loss_G = lambda_latent * loss_latent + lambda_map * loss_map
        loss_G += lambda_Adv_ss * loss_ss_GAN
        loss_G.backward()
        optimizer_G.step()


        # Log
        with torch.no_grad():
            # Scalars
            writer.add_scalars('data/loss', {'loss_KL':loss_KL}, counter)
            writer.add_scalars('data/loss', {'loss_latent':loss_latent}, counter)
            writer.add_scalars('data/loss', {'loss_map':loss_map}, counter)

            writer.add_scalars('data/score_mean', {'real_score_recon':real_score_recon.item()}, counter)
            writer.add_scalars('data/score_mean', {'fake_score_recon':fake_score_recon}, counter)
            writer.add_scalars('data/loss', {'loss_features':loss_features}, counter)

            writer.add_scalars('data/score_mean', {'real_score_ss':real_score_ss.item()}, counter)
            writer.add_scalars('data/score_mean', {'fake_score_ss':fake_score_ss}, counter)

            # Generate Maps by Training Data
            map_recon = G_map(img_recon)
            map_ss = G_map(img_ss)

            # Generate Images and Maps by Validation Data
            (val_img, _) = dataloader.get_val()
            val_img = downsample(val_img.to(device), scale)
            val_dic = generate_results(G_map, E, G, val_img, latent_dim=8, H_imgs=H_imgs, W_imgs=W_imgs)

            # Order Results
            train_results = show_results(batch_img, img_recon, img_ss, fake_map, map_ss)
            val_results = show_results(val_img, val_dic['img_recon'], val_dic['img_ss'], val_dic['fake_maps'], val_dic['map_ss'])

            # Generate Example
            example_dic = generate_results(G_map, E, G, example_img, latent_dim=8, H_imgs=H_imgs, W_imgs=W_imgs)
            example_fake_map = convert_3ch(example_dic['fake_maps'])
            example_map_ss = convert_3ch(example_dic['map_ss'])

            example = torch.cat((example_img, example_dic['img_ss'], example_fake_map, example_map_ss), 0)
            example = torchvision.utils.make_grid(example, nrow=2, padding=1, normalize=True)

            # Generate Examples by Diverse Random Numbers
            chs = 8
            val_sampled_z = torch.randn(chs, latent_dim, H_imgs, W_imgs).to(device)
            G_map(example_img.expand(chs, 3, int(192/scale), int(256/scale)))
            val_z_S = G_map.z_S
            sampled_diversity = G(val_sampled_z, val_z_S)

            sampled_diversity = torch.cat((example_img, sampled_diversity), 0)
            sampled_diversity = torchvision.utils.make_grid(sampled_diversity, nrow=3, normalize=True)


            writer.add_image('Results/0.train_results', BGR2RGB(train_results), counter)
            writer.add_image('Results/1.val_results', BGR2RGB(val_results), counter)
            writer.add_image('Results/2.example', BGR2RGB(example), counter)
            writer.add_image('Results/3.sampled_diversity', BGR2RGB(sampled_diversity), counter)

        counter += 1
        
    # Save weights every 3 epoch
    if (current_epoch + 1) % 3 == 0:
        print('pkl saved')
        print('Epoch [{}/{}]'.format(current_epoch, num_epoch))
        torch.save(G.state_dict(), ''.join([DIR_TO_SAVE, '/G.pkl']))
        torch.save(E.state_dict(), ''.join([DIR_TO_SAVE, '/E.pkl']))
torch.save(G.state_dict(), ''.join([DIR_TO_SAVE, '/G.pkl']))
torch.save(E.state_dict(), ''.join([DIR_TO_SAVE, '/E.pkl']))
writer.close()
print('Done')