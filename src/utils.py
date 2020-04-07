import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image

device = 'cuda'

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

def reparameterize(mu, logvar):
    sampled_z = torch.randn_like(mu)
    z = mu + (sampled_z * (torch.exp(0.5 * logvar)))
    return z

def KLD(mu_1, var_1, mu_2=0, var_2=1):
    kld = 0.5 * ((mu_1-mu_2)**2/var_2 + var_1/var_2 - (var_1/var_2).log() - 1)
    return kld

def BGR2RGB(x):
    y = x.to('cpu').numpy()
    y = y[(2,1,0),:,:]
    return torch.as_tensor(y)
    
def convert_3ch(gray_img):
    to_tensor = transforms.ToTensor()
    pilTrans = transforms.ToPILImage()
    img = pilTrans(gray_img.squeeze(0).to('cpu')).convert("RGB")
    img = to_tensor(img).unsqueeze(0).to(device)
    return img

def save_img(img, path, gray=False):
    pilTrans = transforms.ToPILImage()
    if gray:
        img = img.to('cpu').data
    else:
        img = BGR2RGB(img).to('cpu').data
    pilImg = pilTrans(img)
    pilImg.save(path)

def generate_results(G_map, E, G, imgs, latent_dim, H_imgs, W_imgs):
    results_dic = {}

    mu, logvar = E(imgs)
    results_dic['fake_maps'] = G_map(imgs)
    z_S = G_map.z_S

    encoded_z = reparameterize(mu, logvar)
    results_dic['img_recon'] = G(encoded_z, z_S)
    results_dic['map_recon'] = G_map(results_dic['img_recon'])

    sampled_z = torch.randn(mu.size(0), latent_dim, H_imgs, W_imgs).to(device)
    results_dic['img_ss'] = G(sampled_z, z_S)
    results_dic['map_ss'] = G_map(results_dic['img_ss'])

    return results_dic

def show_results(imgs, imgs_recon, imgs_ss, fake_maps, maps_ss):
    grid_imgs = torchvision.utils.make_grid(imgs[:8], nrow=1, padding=1, normalize=True)
    grid_imgs_recon = torchvision.utils.make_grid(imgs_recon[:8], nrow=1, padding=1, normalize=True)
    grid_imgs_ss = torchvision.utils.make_grid(imgs_ss[:8], nrow=1, padding=1, normalize=True)
    grid_fake_maps = torchvision.utils.make_grid(fake_maps[:8], nrow=1, padding=1, normalize=True)
    grid_maps_ss = torchvision.utils.make_grid(maps_ss[:8], nrow=1, padding=1, normalize=True)

    grid_fake_maps = convert_3ch(grid_fake_maps)
    grid_maps_ss = convert_3ch(grid_maps_ss)

    results = torch.cat((grid_imgs.unsqueeze(0), grid_imgs_recon.unsqueeze(0), grid_imgs_ss.unsqueeze(0), grid_fake_maps, grid_maps_ss), 0)
    results = torchvision.utils.make_grid(results, nrow=5, padding=0, normalize=True)
    return results