import math
import pickle

import torch
from torch import nn
from torch import optim

from datasets2 import train_loader, val_loader
from model import GeneratorResNet, Discriminator
from vggLoss import VGGLoss


def train_fn(dataset, val_loader, disc, gen, opt_gen, opt_disc, mse, bce, vgg_loss):
    disc_val_losses = 0
    gen_val_losses = 0
    disc_val_scores = 0
    gen_val_scores = 0
    for idx, (low_res, high_res) in enumerate(dataset):


        # counter
        # convert data from tensorflow to pytorch tensor
        low_res = low_res.permute((0, 3, 1, 2))
        high_res = high_res.permute((0, 3, 1, 2))
        opt_gen.zero_grad()
        fake = gen(low_res)
        disc_fake = disc(fake)
        # l2_loss = mse(fake, high_res)

        adversarial_loss = bce(disc_fake, torch.ones_like(disc_fake))
        # if(math.isnan(adversarial_loss)):
        #     nan_indices.append(idx+1)
        #     continue
        loss_for_vgg =  (0.006) * vgg_loss(fake, high_res) #1/(6*6) = 1/W*H, 0.006=divide feature maps by 12.75
        gen_loss = loss_for_vgg + 1e-3 * adversarial_loss

        gen_loss.backward()
        opt_gen.step()

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        # if int(epoch+1) % 3 == 0 or (epoch+1)<20:
        opt_disc.zero_grad()
        disc_real = disc(high_res)
        disc_fake = disc(fake.detach())

        disc_loss_real = bce(disc_real, torch.ones_like(disc_real))
        disc_loss_fake = bce(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (disc_loss_fake + disc_loss_real)
        loss_disc.backward()
        opt_disc.step()

        if idx == 1:
            eval(idx)
        disc_val_losses += loss_disc.item()
        gen_val_losses += gen_loss.item()
        disc_val_scores += disc_real.mean().item()
        gen_val_scores += disc_fake.mean().item()

    d_losses_train.append(disc_val_losses / 133)
    g_losses_train.append(gen_val_losses / 133)
    real_scores_train.append(disc_val_scores / 133)
    fake_scores_train.append(gen_val_scores / 133)

    # if int((epoch+1)) % 2 == 0:
    print('Epoch [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
      .format(epoch, epochs, disc_val_losses / 133, (gen_val_losses * 10 ** 3) / 133,
              disc_val_scores / 133, gen_val_scores / 133), ',Training Disc Training')
    # else:
    #     print('Epoch [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
    #           .format(epoch, epochs, disc_val_losses / 133, (gen_val_losses * 10 ** 3) / 133,
    #                   disc_val_scores / 133, gen_val_scores / 133), ',Training Disc NOT Training')





def eval(i):
    disc_val_losses = 0
    gen_val_losses = 0
    disc_val_scores = 0
    gen_val_scores = 0
    for idx, (low_res, high_res) in enumerate(val_loader):
        low_res = low_res.permute((0, 3, 1, 2))
        high_res = high_res.permute((0, 3, 1, 2))
        opt_gen.zero_grad()
        fake = gen(low_res)
        disc_fake = disc(fake)
        # l2_loss = mse(fake, high_res)

        adversarial_loss = bce(disc_fake, torch.ones_like(disc_fake))
        # if(math.isnan(adversarial_loss)):
        #     nan_indices.append(idx+1)
        #     continue
        loss_for_vgg = (0.006) * vgg_loss(fake, high_res)
        gen_loss = loss_for_vgg + 1e-3 * adversarial_loss
        opt_disc.zero_grad()

        disc_real = disc(high_res)
        disc_fake = disc(fake.detach())

        disc_loss_real = bce(
            disc_real, torch.ones_like(disc_real)
        )
        disc_loss_fake = bce(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (disc_loss_fake + disc_loss_real)

        disc_val_losses += loss_disc.item()
        gen_val_losses += gen_loss.item()
        disc_val_scores += disc_real.mean().item()
        gen_val_scores += disc_fake.mean().item()
    d_losses.append(disc_val_losses/53)
    g_losses.append(gen_val_losses/53)
    real_scores.append(disc_val_scores/53)
    fake_scores.append(gen_val_scores/53)
    # if int((epoch+1)) % 2 == 0:
    print('Epoch [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
              .format(epoch, epochs, disc_val_losses / 53, (gen_val_losses * 10 ** 3) / 53,
                      disc_val_scores / 53, gen_val_scores / 53), ',Validation')
    # else:
    #     print('Epoch [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
    #           .format(epoch, epochs, disc_val_losses / 53, (gen_val_losses * 10 ** 3) / 53,
    #                   disc_val_scores / 53, gen_val_scores / 53), ',Validation Disc NOT Training')
    # print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
    #       .format(epoch, epochs, i, 216, disc_val_losses/10, gen_val_scores/10,
    #               disc_val_scores/10, gen_val_scores/10))



def train_srresnet(dataset, val_loader, gen, opt_gen, mse):
    for idx, (low_res, high_res) in enumerate(dataset):

        # counter
        # convert data from tensorflow to pytorch tensor
        low_res = low_res.permute((0, 3, 1, 2))
        high_res = high_res.permute((0, 3, 1, 2))
        opt_gen.zero_grad()
        fake = gen(low_res)
        gen_loss = mse(fake, high_res)
        gen_loss.backward()
        opt_gen.step()

        if idx == 0:
            eval_srresnet(idx)
        # if(idx%50 == 0):
        #     print('Batch', idx, math.isnan(loss_disc))


def eval_srresnet(i):
    gen_val_losses = 0
    for idx, (low_res, high_res) in enumerate(val_loader):
        low_res = low_res.permute((0, 3, 1, 2))
        high_res = high_res.permute((0, 3, 1, 2))
        opt_gen.zero_grad()
        fake = gen(low_res)
        gen_loss = mse(fake, high_res)

        gen_val_losses += gen_loss.item()

    srresnet_losses.append(gen_val_losses/10)
    print('Epoch [{}/{}], g_loss: {:.4f}'
          .format(epoch, epochs, gen_val_losses / 10))


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def set_learning_rate(op_d, lr):
    for x in op_d.param_groups:
        x['lr'] = lr



def save_ckp(state, checkpoint_dir):
    f_path = checkpoint_dir
    torch.save(state, f_path)


def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer


# for name, param in gen.named_parameters():
#     if param.requires_grad:
#         print (name, param.data)
#     break
torch.manual_seed(42)
gen = GeneratorResNet().to('cuda')
# gen.load_state_dict(torch.load('G.ckptNewVGGLOSS100'))
# gen.eval()

disc = Discriminator().to('cuda')


# disc.load_state_dict(torch.load('D.ckptLOSSNewVGGLOSS100'))
# disc.eval()




mse = nn.MSELoss()
bce = nn.BCEWithLogitsLoss()
vgg_loss = VGGLoss()
epochs = 100
# opt_gen = optim.Adam(gen.parameters(), lr=1e-4, betas=(0.9, 0.999))
# gen, opt_gen = load_ckp('G.ckptsrresnet75Flickr DS lr=1.25e-05', gen, opt_gen)
# gen.eval()



# opt_gen = optim.Adam(gen.parameters(), lr=1e-4, betas=(0.9, 0.999))
# lr_gen = (1e-4)/16
# # gen, opt_gen = load_ckp('G.ckptsrresnet50LDS lr=5e-05', gen, opt_gen)
# opt_gen = optim.Adam(gen.parameters(), lr=lr_gen, betas=(0.9, 0.999))
# #
# with open("srresnetLoss75Flickr DS lr= 1.25e-05", "rb") as fp:   # Unpickling
#  srresnet_losses = pickle.load(fp)
srresnet_losses = []
# for epoch in range(75,epochs):
#     train_srresnet(train_loader, val_loader, gen, opt_gen, mse)
#     if (epoch + 1) % 25 == 0:
#         with open("srresnetLoss" + str(epoch + 1) + "Flickr DS lr= "+ str(lr_gen) , "wb") as fp:  # Pickling
#             pickle.dump(srresnet_losses, fp)
#         ckpt = {
#             'state_dict': gen.state_dict(),
#             'optimizer': opt_gen.state_dict()
#         }
#         save_ckp(ckpt, 'G.ckptsrresnet' + str(epoch + 1)+ "Flickr DS lr=" + str(lr_gen))
#     if (epoch + 1) % 25 == 0:
#         lr_gen /= 2
#         set_learning_rate(opt_gen, lr_gen)
#         print(lr_gen)



lr_gen = float(1e-6)
lr_disc = float(1e-6)
opt_disc = optim.Adam(disc.parameters(), lr=lr_disc, betas=(0.9, 0.999))
opt_gen = optim.Adam(gen.parameters(), lr=lr_gen, betas=(0.9, 0.999))

# ckpt = {
#             'state_dict': gen.state_dict(),
#             'optimizer': opt_gen.state_dict()
#         }
# save_ckp(ckpt, 'G.ckpt' + 'srresnet' +"FlickrDs")
#

gen, _ = load_ckp('D:\GUC S8\Models\Experiment6\G.ckptsrresnet100Flickr DS lr=6.25e-06', gen, opt_gen)
# disc, _ = load_ckp('D.ckpt250FlickrDS', disc, opt_disc)
opt_disc = optim.Adam(disc.parameters(), lr=lr_disc, betas=(0.9, 0.999))
opt_gen = optim.Adam(gen.parameters(), lr=lr_gen, betas=(0.9, 0.999))


#

# with open("g_losses250FlickrDS", "rb") as fp:   # Unpickling
#  g_losses = pickle.load(fp)
# with open("d_losses250FlickrDS", "rb") as fp:   # Unpickling
#  d_losses = pickle.load(fp)
# with open("real_scores250FlickrDS", "rb") as fp:   # Unpickling
#  real_scores = pickle.load(fp)
# with open("fake_scores250FlickrDS", "rb") as fp:   # Unpickling
#  fake_scores = pickle.load(fp)
#
# with open("g_losses_train250FlickrDS", "rb") as fp:   # Unpickling
#  g_losses_train = pickle.load(fp)
# with open("d_losses_train250FlickrDS", "rb") as fp:   # Unpickling
#  d_losses_train = pickle.load(fp)
# with open("real_scores_train250FlickrDS", "rb") as fp:   # Unpickling
#  real_scores_train = pickle.load(fp)
# with open("fake_scores_train250FlickrDS", "rb") as fp:   # Unpickling
#  fake_scores_train = pickle.load(fp)



# torch.autograd.set_detect_anomaly(True)
epochs = 1000


d_losses, g_losses, real_scores, fake_scores = [], [], [], []
d_losses_train, g_losses_train, real_scores_train, fake_scores_train = [], [], [], []

torch.manual_seed(42)
for epoch in range(epochs):
    train_fn(train_loader, val_loader, disc, gen, opt_gen, opt_disc, mse, bce, vgg_loss)
    if (epoch + 1) % 10 == 0:
        with open("d_losses" + str(epoch + 1) + "FlickrDS", "wb") as fp:  # Pickling
            pickle.dump(d_losses, fp)
        with open("g_losses" + str(epoch + 1)+ "FlickrDS", "wb") as fp:  # Pickling
            pickle.dump(g_losses, fp)
        with open("real_scores" + str(epoch + 1)+ "FlickrDS", "wb") as fp:  # Pickling
            pickle.dump(real_scores, fp)
        with open("fake_scores" + str(epoch + 1)+ "FlickrDS", "wb") as fp:  # Pickling
            pickle.dump(fake_scores, fp)
        with open("d_losses_train" + str(epoch + 1) + "FlickrDS", "wb") as fp:  # Pickling
            pickle.dump(d_losses_train, fp)
        with open("g_losses_train" + str(epoch + 1)+ "FlickrDS", "wb") as fp:  # Pickling
            pickle.dump(g_losses_train, fp)
        with open("real_scores_train" + str(epoch + 1)+ "FlickrDS", "wb") as fp:  # Pickling
            pickle.dump(real_scores_train, fp)
        with open("fake_scores_train" + str(epoch + 1)+ "FlickrDS", "wb") as fp:  # Pickling
            pickle.dump(fake_scores_train, fp)
        ckpt = {
            'state_dict': gen.state_dict(),
            'optimizer': opt_gen.state_dict()
        }
        save_ckp(ckpt, 'G.ckpt' + str(epoch + 1)+ "FlickrDS")
        ckpt = {
            'state_dict': disc.state_dict(),
            'optimizer': opt_disc.state_dict()
        }
        save_ckp(ckpt, 'D.ckpt' + str(epoch + 1) + "FlickrDS")
    # if (epoch + 1) % 2 == 0:
    #     lr_gen/=2
    #     lr_disc/=2
    #     set_learning_rate(opt_gen, lr_gen)
    #     set_learning_rate(opt_disc, lr_disc)
    #     print('lr_gen:', lr_gen)
    #     print('lr_disc', lr_disc)

    # if (epoch + 1) == 2000:
    #     lr_gen = 1e-5
    #     lr_disc = 1e-5
    #     set_learning_rate(opt_disc, opt_disc, lr_disc)
    #     set_learning_rate(opt_gen, opt_gen, lr_gen)


# torch.save(gen.state_dict(), 'G.ckpt200')
# torch.save(disc.state_dict(), 'D.ckpt200')
# plt.plot(d_losses, '-')
# plt.plot(g_losses, '-')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.legend(['Discriminator', 'Generator'])
# plt.title('Losses')
# plt.show()
#
#
# plt.plot(real_scores, '-')
# plt.plot(fake_scores, '-')
# plt.xlabel('epoch')
# plt.ylabel('score')
# plt.legend(['Real Score', 'Fake score'])
# plt.title('Scores')
# plt.show()



