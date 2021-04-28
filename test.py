import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm
from transforms import *

from dataset import XzwDataSet

# img = Image.open("E:\\2.jpg")
# plt.figure()
# plt.imshow(img)
# plt.show()
# print(np.array(img).shape)


# tick = (60 - 8 * 2 + 1) / float(5)
# sample_start_pos = np.array([int(tick / 2.0 + tick * x) for x in range(5)])
# offsets = []
# for p in sample_start_pos:
#     offsets.extend(range(p, p + 8 * 2, 2))
#
# print(sample_start_pos)
# checked_offsets = []
# for f in offsets:
#     new_f = int(f) + 1
#     print(111, new_f)
#     if new_f < 1:
#         new_f = 1
#     elif new_f >= 60:
#         new_f = 60 - 1
#     checked_offsets.append(new_f)
# print(len(checked_offsets))

# if __name__ == '__main__':
#
#     train_loader = torch.utils.data.DataLoader(
#             XzwDataSet('', 'data.txt',
#                        sample_frames=16,
#                        modality='RGB',
#                        image_tmpl="{:05d}.jpg" ,
#                        transform=torchvision.transforms.Compose([
#                        Stack(),
#                        ToTorchFormatTensor(),
#                    ])),
#             batch_size=2, shuffle=True,
#             num_workers=1, pin_memory=True)
#
#     for i, (input, target) in enumerate(train_loader):
#         # measure data loading time
#
#         # target = target.cuda(async=True)
#         # input_var = torch.autograd.Variable(input)
#         input_var = input
#         # target_var = torch.autograd.Variable(target)
#         np_img = np.asarray(input_var)
#         target_var = torch.tensor(np_img)
#         print(target_var.size())
#         img = target_var.permute(0, 2, 3, 4, 1)
#         plt.figure()
#         plt.imshow(img[0][0])
#         plt.show()


a = torch.ones(3,4,2,2)
split = a.size(1) // 2
print(split)
# temp_x = torch.zeros(a.size())
# temp_y = torch.zeros(a.size())
# print(temp_x.size())
# print(temp_y.size())

temp_x = a[:, 0:split, :,:]
temp_y = a[:,split:,:,:]
print(temp_x.size(),66666)
print(temp_y.size())
relation_z = torch.matmul(temp_x, temp_y)
print(relation_z.size())
