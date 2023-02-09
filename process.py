

from data import DIV2K

# from torchsr.datasets import Div2K
# from torchsr.models import ninasr_b0

import matplotlib.pyplot as plt
import torch


train_loader = DIV2K(scale=4,  # 2, 3, 4 or 8
                     downgrade='bicubic',  # 'bicubic', 'unknown', 'mild' or 'difficult'
                     subset='train')  # Training dataset are images 001 - 800


# Create a tf.data.Dataset
train_ds = train_loader.dataset(batch_size=16,
                                random_transform=False,
                                repeat_count=None)
# idx = 1
# for hr in train_ds:
#     print(idx)
#     idx+=1


# print(torch.cuda.is_available())

idx = 0
# for lr, hr in train_ds:
#     lr_np = lr.numpy()
#     # print(idx)
#     # lr = torch.tensor(lr_np)
#     # low_res = lowres_transform(image=lr_np)["image"]
#     lr = torch.tensor(lr_np)
#     print(idx)
#     # print(torch.max(lr[0]))
#
#
#
#     hr_np = hr.numpy()
#     # hi_res = highres_transform(image=hr_np)["image"]
#     hr = torch.tensor(hr_np)
#     if(idx==12 or idx==175):
#         plt.imshow(hr[0])
#         plt.show()
#         plt.imshow(lr[0])
#         plt.show()
#     # print(hr.shape)
#     # print(torch.max(hr[0]))
#     # fig, ax = plt.subplots(figsize=(12, 6))
#     # ax.set_xticks([]);
#     # ax.set_yticks([])
#     # ax.imshow(make_grid(lr, nrow=16))
#     # ax.show()
#     # for img in hr:
#     # fig = plt.figure()
#     # for i in range(8):
#     #     img = hr[i]
#     #     fig.add_subplot(2, 4, i + 1)
#     #     plt.imshow(img)
#     # plt.show()
#     idx+=1


