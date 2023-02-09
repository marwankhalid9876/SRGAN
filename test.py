import math
import os

import torch

from torchvision.utils import save_image
import pickle
from skimage.metrics import structural_similarity as ssim
from torch import nn
from torch import optim
from vggLoss import VGGLoss
from torch.utils.data import DataLoader
from model import GeneratorResNet, Discriminator
from datasets2 import train_loader, val_loader, set14_loader,set5_loader
import matplotlib.pyplot as plt
import numpy as np

def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer





genresnet = GeneratorResNet()
opt_gen = optim.Adam(genresnet.parameters(), lr=1e-4, betas=(0.9, 0.999))
genresnet, opt_gen = load_ckp('D:\GUC S8\Models\Experiment6\G.ckptsrresnet100Flickr DS lr=6.25e-06', genresnet, opt_gen)
genresnet.eval()
gen50 = GeneratorResNet()
opt_gen = optim.Adam(gen50.parameters(), lr=1e-4, betas=(0.9, 0.999))
gen50, opt_gen = load_ckp('G.ckpt70FlickrDS', gen50, opt_gen)
# genresnet.load_state_dict(torch.load('G.ckptsrresnet'))
gen50.eval()

# gen50 = GeneratorResNet()
# gen50.load_state_dict(torch.load('G.ckpt50'))
# gen50.eval()
# gen100 = GeneratorResNet()#.to(config.DEVICE)
# gen100.load_state_dict(torch.load('Models/G.ckptNewVGGLOSS100'))
# gen100.eval()

def scale(img):

    img = torch.div(
        torch.subtract(
            img,
            torch.min(img)
        ) ,
        torch.subtract(
            torch.max(img),
            torch.min(img)
        )
    )
    img = img * 255
    img = img.int()
    return img

# for idx, (low_res, high_res) in enumerate(set14_loader):
#     # print("Low", low_res[0], "High", high_res[0])
#     low_res = low_res.permute((0, 1, 2, 3)).float().to('cpu')
#     high_res = high_res.permute((0, 1, 2, 3)).float().to('cpu')
#     high_res = scale(high_res)
#     fig = plt.figure()
#     # for i in range(1):
#     img = high_res[0].permute([1,2,0])
#         # fig.add_subplot(2, 4, i+1)
#     plt.imshow(img)
#     plt.show()
#
#     fakeresnet = genresnet(low_res).detach()
#     fakeresnet = scale(fakeresnet)
#
#     fake50 = gen50(low_res).detach()
#     fake50 = scale(fake50)
#     # fake100 = gen100(low_res).detach()
#     # print("Fake", fake100[0])
#     # fake100 = scale(fake100)
#     fig1 = plt.figure()
#     # for i in range(1):
#     img = fakeresnet[0].permute([1, 2, 0])
#         # fig1.add_subplot(2, 4, i + 1)
#     plt.imshow(img)
#     plt.show()
#     fig2 = plt.figure()
#     # for i in range(1):
#     img = fake50[0].permute([1, 2, 0])
#         # fig2.add_subplot(2, 4, i + 1)
#     plt.imshow(img)
#     plt.show()
#
#     # fig2 = plt.figure()
#     # for i in range(8):
#     #     img = fake50[i].permute([1, 2, 0])
#     #     fig2.add_subplot(2, 4, i + 1)
#     #     plt.imshow(img)
#     # plt.show()
#
#     # fig2 = plt.figure()
#     # for i in range(8):
#     #     img = fake100[i].permute([1, 2, 0])
#     #     fig2.add_subplot(2, 4, i + 1)
#     #     plt.imshow(img)
#     # plt.show()
#
#     fig3 = plt.figure()
#     # for i in range(1):
#     img = low_res[0].permute([1, 2, 0])
#         # fig3.add_subplot(2, 4, i + 1)
#     plt.imshow(img)
#     plt.show()


def PSNR(original, compressed):
    mse = np.mean((original.numpy() - compressed.numpy()) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

psnr_srresnet, psnr_srgan50, psnr_srgan100, ssim_srresnet, ssim_srgan50, ssim_srgan100 = [],[], [], [], [], []
for idx, (low_res, high_res) in enumerate(set5_loader):
    # print("Low", low_res[0], "High", high_res[0])
    # print(idx)
    if(idx == 50):
        break
    low_res = low_res.to('cpu')
    high_res = high_res.permute((0, 2, 3, 1)).to('cpu')#0, 1, 2, 3 flickr train loader #0, 2, 3, 1 set14
    low_res_for_g = low_res.permute((0, 1, 2, 3)).float()#0, 3, 1, 2 flickr train loader #0, 1, 2, 3 set14
    high_res = scale(high_res)
    # fig = plt.figure()
    # fig.add_subplot(1, 4, 1)
    #
    # plt.imshow(high_res[0])

    fakeresnet = genresnet(low_res_for_g).detach()
    fakeresnet = scale(fakeresnet)
    fakeresnet = fakeresnet.permute((0, 2, 3, 1)).to('cpu')

    fake50 = gen50(low_res_for_g).detach()
    fake50 = scale(fake50)
    fake50 = fake50.permute((0, 2, 3, 1)).to('cpu')

    # fake100 = gen100(low_res_for_g).detach()
    # fake100 = scale(fake100)
    # fake100 = fake100.permute((0, 2, 3, 1)).to('cpu')

    # fig.add_subplot(1, 4, 2)
    # plt.imshow(fakeresnet[0])
    #
    # fig.add_subplot(1, 4, 3)
    # plt.imshow(fake50[0])
    #
    # fig.add_subplot(1, 4, 4)
    # plt.imshow(low_res[0])
    # plt.show()
    print(PSNR(high_res,fakeresnet),(idx+1))
    print(PSNR(high_res,fake50),(idx+1))
    psnr_srresnet.append(PSNR(high_res,fakeresnet))
    psnr_srgan50.append(PSNR(high_res,fake50))
    # psnr_srgan100.append(PSNR(high_res,fake100))

#
#
    ssim_srresnet.append(ssim(high_res[0].numpy(), fakeresnet[0].numpy(),
         data_range=high_res.numpy().max() - high_res.numpy().min(), multichannel=True))
    ssim_srgan50.append(ssim(high_res[0].numpy(), fake50[0].numpy(),
                              data_range=high_res.numpy().max() - high_res.numpy().min(), multichannel=True))
    # ssim_srgan100.append(ssim(high_res[0].numpy(), fake100[0].numpy(),
    #                           data_range=high_res.numpy().max() - high_res.numpy().min(), multichannel=True))
#
#
#
#
#
#
def Average(lst):
    return sum(lst) / len(lst)
print("Average PSNR SRResnet 100 epochs on set5 DS", Average(psnr_srresnet))
print("Average PSNR SRGAN 70 epochs on set5 DS", Average(psnr_srgan50))
# print("Average PSNR SRGAN350:", Average(psnr_srgan100))
print("Average SSIM SRResnet 100 epochs on set5 DS", Average(ssim_srresnet))
print("Average SSIM SRGAN 70 epochs on set5 DS", Average(ssim_srgan50))
# print("Average SSIM SRGAN350:", Average(ssim_srgan100))

#

#plotting losses
# with open("g_losses50LDS", "rb") as fp:   # Unpickling
#  g_losses = pickle.load(fp)
# # with open("d_losses2500", "rb") as fp:   # Unpickling
# #  d_losses = pickle.load(fp)
# #
# # print(g_losses)
# #
# plt.plot(g_losses, label='Discriminator Real Scores')
# plt.xlabel = "Epochs"
# plt.xlabel = "score"
# # plt.plot(d_losses, label='discriminator loss')
# plt.legend()
# #
# plt.show()
#
with open("d_losses290FlickrDS", "rb") as fp:   # Unpickling
 g_losses = pickle.load(fp)
with open("d_losses_train290FlickrDS", "rb") as fp:   # Unpickling
 g_losses_train = pickle.load(fp)
# # with open("d_losses2500", "rb") as fp:   # Unpickling
# #  d_losses = pickle.load(fp)
#
# # print(g_losses)
# for (i, item) in enumerate(g_losses):
#     print(i, item)
plt.plot(g_losses, label='Discriminator losses on validation data')
plt.plot(g_losses_train, label='Discriminator losses on training data')
plt.legend()
# #
plt.show()

