from torch import nn
import os
from ops.basic_ops import ConsensusModule, Identity
from transforms import *
from torch.nn.init import normal, constant
import time
from torch.nn.init import xavier_uniform_, constant_

# 参照STM 对relation 信息进行加工

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, head_conv=1):
        super(Bottleneck, self).__init__()
        if head_conv == 1:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm3d(planes)
        elif head_conv == 3:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 1), bias=False, padding=(1, 0, 0))
            self.bn1 = nn.BatchNorm3d(planes)
        else:
            raise ValueError("Unsupported head_conv!")
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=(1, 3, 3), stride=(1,stride,stride), padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)



        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out



# nn.Conv3d + bn + relu
class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=False)  # verify bias false

        # verify defalt value in sonnet
        self.bn = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# nn.Conv2d + bn + relu  <!-- Actually using nn.Conv3d in order to keep the same shape of every branch. --!>
class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=(1, kernel_size, kernel_size), stride=(1, stride, stride),
                              padding=(0, padding, padding), bias=False)    # verify bias false
        self.bn = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# nn.Conv2d + bn + relu  <!-- Actually using nn.Conv2d is faster than nn.Conv3d. --!>
class BasicConv_2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super(BasicConv_2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)    # verify bias false
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-3, momentum=0.1, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# separable 3d = (2 + 1)d    <!-- Actually this method costs much time.--!>
class STConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(STConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=(1, kernel_size, kernel_size),
                              stride=(1, stride, stride), padding=(0, padding, padding))
        self.conv2 = nn.Conv3d(out_planes, out_planes, kernel_size=(kernel_size, 1, 1), stride=(stride, 1, 1),
                               padding=(padding, 0, 0))

        self.bn = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu = nn.ReLU(inplace=True)

        self.bn2 = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu2 = nn.ReLU(inplace=True)

        # nn.init.normal(self.conv2.weight,mean=0,std=0.01)
        # nn.init.constant(self.conv2.bias,0)

    def forward(self, x):
        x = self.conv(x)
        # print('conv',x.size())
        # x=self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        # print('conv2',x.size())
        x = self.bn2(x)
        x = self.relu2(x)
        return x

# Note the operations here for S3D-G:
# If we set two convs: 1xkxk + kx1x1, it's as follows: (p=(k-1)/2)
# BasicConv3d(input,output,kernel_size=(1,k,k),stride=1,padding=(0,p,p))
# Then BasicConv3d(output,output,kernel_size=(k,1,1),stride=1,padding=(p,0,0))

# ---------------------------- Group Model beginning ---------------------
# Group Block contains 2D，3D，（2+1）D，Relation branches
class GroupBlock(nn.Module):
    def __init__(self, inplanes_2d, outplanes_2d, kernel_size2d, stride2d, padding2d,
                 inplanes_3d, outplanes_3d, kernel_size3d, stride3d, padding3d):
        super(GroupBlock, self).__init__()

        self.branch0 = nn.Sequential(

        )


# For 2D branch, we use Inception-Like structure to extract features
class Branch_2D(nn.Module):
    def __init__(self, in_channels, out_channels=512, scaled=1.0):
        super(Branch_2D, self).__init__()
        out_channels = int(out_channels * scaled)

        self.conv2d_1a = BasicConv2d(in_channels, 32, kernel_size=7, stride=2, padding=3)         # 112 112
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)                   # 112
        self.conv2d_2b = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)        # 112
        # self.maxpool_3a = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2))                     #
        self.conv2d_3b = BasicConv2d(32, 32, kernel_size=1, stride=1)                   #
        self.conv2d_4a = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)                  #
        self.maxpool_5a = nn.MaxPool3d((1, 3, 3), stride=(2, 2, 2), padding=(0, 1, 1))                     #
        self.mixed_5b = Mixed_block()   # 320 out   28
        self.conv2d_7b = BasicConv2d(224, out_channels, kernel_size=1, stride=1)                     # 512 16 28 28
        self.avgpool_1a = nn.AvgPool2d(8, count_include_pad=False)  # ++++++++++++++

    def forward(self, x):
        x = self.conv2d_1a(x)
        # print(x.size())
        x = self.conv2d_2a(x)

        x = self.conv2d_2b(x)
        # x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.maxpool_5a(x)
        x = self.mixed_5b(x)
        x = self.conv2d_7b(x)
        # print(x.size(),'2d')
        #x = self.avgpool_1a(x)

        return x



class Mixed_block(nn.Module):

    def __init__(self):
        super(Mixed_block, self).__init__()

        self.branch0 = BasicConv2d(32, 48, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(32, 48, kernel_size=1, stride=1),
            BasicConv2d(48, 64, kernel_size=5, stride=1, padding=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(32, 48, kernel_size=1, stride=1),
            BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1),
            BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool3d((1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), count_include_pad=False),
            BasicConv2d(32, 48, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out



# For 3D Branch, now we use simple C3D strutrue
class Branch_3D(nn.Module):
    def __init__(self, in_channels, out_channels=128, scaled=1.0):      # 3d channels will be scaled in low/mid/high level features
        super(Branch_3D, self).__init__()

        out_channels = int(out_channels * scaled)
        self.conv1 = BasicConv3d(in_channels, 32, kernel_size=(3, 7, 7), stride=(2, 2, 2), padding=(1, 3, 3))     # +++++++++++++kernel_size, stride should be modified+++++++++++++++
        self.conv2 = BasicConv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)
        self.conv3 = BasicConv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)
        self.conv3_1 = BasicConv3d(32, out_channels, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1)

    def forward(self, x):
        x = self.conv1(x)
        # print(x.size(), '3D1')
        x = self.conv2(x)
        # print(x.size(), '3D2')
        x = self.conv3(x)
        # print(x.size(), '3D3')
        x = self.conv3_1(x)
        # print(x.size(), '3D')

        return x

# For (2+1)D Branch, we use separable convnet
class Branch_STConv(nn.Module):
    def __init__(self, in_channels, out_channels=256, scaled=1.0):
        super(Branch_STConv, self).__init__()
        out_channels = int(out_channels * scaled)

        self.conv1 = STConv3d(in_channels, 32, kernel_size=7, stride=2, padding=3)
        self.conv2 = STConv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = STConv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3_1 = BasicConv3d(64, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.conv4 = STConv3d(64, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv3_1(x)
        x = self.conv4(x)
        # print(x.size(), '2+1D')

        return x


# For Relation Branch, we use the method that is from ARTNet
class Branch_Relation(nn.Module):
    def __init__(self, in_channels, out_channels=256, num_frames=16, scaled=1.0):
        super(Branch_Relation, self).__init__()
        out_channels = int(out_channels * scaled)

        #self.input = input
        self.num_frames = num_frames
        self.conv1 = BasicConv3d(in_channels, 32, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                                 padding=(1, 3, 3))                 # +++++++++++++kernel_size, stride should be modified+++++++++++++++
        self.conv2 = BasicConv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1)
        self.conv3 = STConv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = STConv3d(64, out_channels, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        # self.relation = self._calc_relation()




    def forward(self, x):
        x = self._calc_relation(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool(x)
        # print(x.size(), 'Relation')
        return x

    def _calc_relation(self, x):        # x.shape should be (C,T,H,W)
        split = self.num_frames // 2

        temp_x = x[:, :, 0:split, :, :]       # ++++++++ need to test the shape of input x ++++++++++++++++++++++++
        temp_y = x[:, :, split:, :, :]
        # print(temp_x.size())
        # print(temp_y.shape)
        relation_z = torch.matmul(temp_x, temp_y)
        # print(relation_z.size())

        return relation_z






class XzwModel(nn.Module):
    def __init__(self, num_classes=101, modality='RGB', basemodel=None, pretrained_parts="scrach", dropout=0.5, num_frames=16 ):
        super(XzwModel, self).__init__()

        self.modality = modality
        self.reshape = True
        self.input_size = 224
        self.dropout = dropout
        self.num_frames = num_frames
        self.pretrained_parts = pretrained_parts
        # if before conv
        # self.process = nn.Sequential(
        #     BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3),  #
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        #     BasicConv2d(64, 64, kernel_size=1),
        #     BasicConv2d(64, 192, kernel_size=3, stride=1, padding=1),  # 112
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        # )
        self.conv1_7x7 = BasicConv_2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.pool1_3x3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)#nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 0, 0)) # nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.conv2_3x3_reduce = BasicConv_2d(64, 64, kernel_size=1)
        self.conv2_3x3 = BasicConv_2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.pool2_3x3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)#nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 0, 0)) # nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        # inception_3a
        self.inception_3a_1x1 = BasicConv_2d(192, 64, kernel_size=1)

        self.inception_3a_3x3_reduce = BasicConv_2d(192, 64, kernel_size=1)
        self.inception_3a_3x3 = BasicConv_2d(64, 64, kernel_size=3, padding=1)

        self.inception_3a_double_3x3_reduce = BasicConv_2d(192, 64, kernel_size=1)
        self.inception_3a_double_3x3_1 = BasicConv_2d(64, 96, kernel_size=3, padding=1)
        self.inception_3a_double_3x3_2 = BasicConv_2d(96, 96, kernel_size=3, padding=1)

        self.inception_3a_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)# nn.AvgPool3d((1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.inception_3a_pool_proj = BasicConv_2d(192, 32, kernel_size=1)

        # inception_3b
        self.inception_3b_1x1 = BasicConv_2d(256, 64, kernel_size=1)

        self.inception_3b_3x3_reduce = BasicConv_2d(256, 64, kernel_size=1)
        self.inception_3b_3x3 = BasicConv_2d(64, 96, kernel_size=3, padding=1)

        self.inception_3b_double_3x3_reduce = BasicConv_2d(256, 64, kernel_size=1)
        self.inception_3b_double_3x3_1 = BasicConv_2d(64, 96, kernel_size=3, padding=1)
        self.inception_3b_double_3x3_2 = BasicConv_2d(96, 96, kernel_size=3, padding=1)

        self.inception_3b_pool =  nn.AvgPool2d(kernel_size=3, stride=1, padding=1)#nn.AvgPool3d((1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.inception_3b_pool_proj = BasicConv_2d(256, 64, kernel_size=1)

        # inception_3c
        self.inception_3c_double_3x3_reduce = BasicConv_2d(320, 64, kernel_size=1)
        self.inception_3c_double_3x3_1 = BasicConv_2d(64, 96, kernel_size=3, padding=1) # 27 27


        # ---------------- 2d branch ------------------
        self.inception_3c_double_3x3_2 = BasicConv_2d(96, 96, kernel_size=3, padding=1, stride=2)
        self.inception_3c_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # inception_4a
        self.inception_4a_1x1 = BasicConv_2d(416, 224, kernel_size=1)

        self.inception_4a_3x3_reduce = BasicConv_2d(416, 64, kernel_size=1)
        self.inception_4a_3x3 = BasicConv_2d(64, 96, kernel_size=3, padding=1)

        self.inception_4a_double_3x3_reduce = BasicConv_2d(416, 96, kernel_size=1)
        self.inception_4a_double_3x3_1 = BasicConv_2d(96, 128, kernel_size=3, padding=1)
        self.inception_4a_double_3x3_2 = BasicConv_2d(128, 128, kernel_size=3, padding=1)

        self.inception_4a_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.inception_4a_pool_proj = BasicConv_2d(416, 128, kernel_size=1)

        # inception_4b
        self.inception_4b_1x1 = BasicConv_2d(576, 192, kernel_size=1)

        self.inception_4b_3x3_reduce = BasicConv_2d(576, 96, kernel_size=1)
        self.inception_4b_3x3 = BasicConv_2d(96, 128, kernel_size=3, padding=1)

        self.inception_4b_double_3x3_reduce = BasicConv_2d(576, 96, kernel_size=1)
        self.inception_4b_double_3x3_1 = BasicConv_2d(96, 128, kernel_size=3, padding=1)
        self.inception_4b_double_3x3_2 = BasicConv_2d(128, 128, kernel_size=3, padding=1)

        self.inception_4b_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.inception_4b_pool_proj = BasicConv_2d(576, 128, kernel_size=1)

        # inception_4c
        self.inception_4c_1x1 = BasicConv_2d(576, 160, kernel_size=1)

        self.inception_4c_3x3_reduce = BasicConv_2d(576, 128, kernel_size=1)
        self.inception_4c_3x3 = BasicConv_2d(128, 160, kernel_size=3, padding=1)

        self.inception_4c_double_3x3_reduce = BasicConv_2d(576, 128, kernel_size=1)
        self.inception_4c_double_3x3_1 = BasicConv_2d(128, 160, kernel_size=3, padding=1)
        self.inception_4c_double_3x3_2 = BasicConv_2d(160, 160, kernel_size=3, padding=1)

        self.inception_4c_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.inception_4c_pool_proj = BasicConv_2d(576, 128, kernel_size=1)

        # inception_4d
        self.inception_4d_1x1 = BasicConv_2d(608, 96, kernel_size=1)

        self.inception_4d_3x3_reduce = BasicConv_2d(608, 128, kernel_size=1)
        self.inception_4d_3x3 = BasicConv_2d(128, 192, kernel_size=3, padding=1)

        self.inception_4d_double_3x3_reduce = BasicConv_2d(608, 160, kernel_size=1)
        self.inception_4d_double_3x3_1 = BasicConv_2d(160, 160, kernel_size=3, padding=1)
        self.inception_4d_double_3x3_2 = BasicConv_2d(160, 192, kernel_size=3, padding=1)

        self.inception_4d_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.inception_4d_pool_proj = BasicConv_2d(608, 192, kernel_size=1)

        # inception_4e
        self.inception_4e_1x1 = BasicConv_2d(672, 128, kernel_size=1)

        self.inception_4e_3x3_reduce = BasicConv_2d(672, 128, kernel_size=1)
        self.inception_4e_3x3 = BasicConv_2d(128, 192, kernel_size=3, padding=1, stride=2)

        self.inception_4e_double_3x3_reduce = BasicConv_2d(672, 192, kernel_size=1)
        self.inception_4e_double_3x3_1 = BasicConv_2d(192, 256, kernel_size=3, padding=1)
        self.inception_4e_double_3x3_2 = BasicConv_2d(256, 256, kernel_size=3, padding=1, stride=2)

        self.inception_4e_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.inception_4e_pool_proj = BasicConv_2d(672, 128, kernel_size=1)
        # -------------- 2d branch end ----------------

        # -------------- conv3d -----------------------
        self.res_3a_2 = BasicConv3d(96, 128, kernel_size=3, padding=1, stride=1)
        self.res_3b_1 = BasicConv3d(128, 128, kernel_size=3, padding=1, stride=1)
        self.res_3b_2 = BasicConv3d(128, 128, kernel_size=3, padding=1, stride=1)

        self.res_4a_1 = BasicConv3d(128, 256, kernel_size=3, padding=1, stride=2)
        self.res_4a_2 = BasicConv3d(256, 256, kernel_size=3, padding=1, stride=1)
        self.res_4a_down = BasicConv3d(128, 256, kernel_size=3, padding=1, stride=2)

        self.res_4b_1 = BasicConv3d(256, 256, kernel_size=3, padding=1, stride=1)
        self.res_4b_2 = BasicConv3d(256, 256, kernel_size=3, padding=1, stride=1)

        self.res_5a_1 = BasicConv3d(256, 512, kernel_size=3, padding=1, stride=2)
        self.res_5a_2 = BasicConv3d(512, 512, kernel_size=3, padding=1, stride=1)
        self.res_5a_down = BasicConv3d(256, 512, kernel_size=3, padding=1, stride=2)

        self.res_5b_1 = BasicConv3d(512, 512, kernel_size=3, padding=1, stride=1)
        self.res_5b_2 = BasicConv3d(512, 512, kernel_size=3, padding=1, stride=1)
        # --------------- 3d end --------------------

        # -------------- ST 2+1 d -----------------------
        self.st_res_3a_2 = STConv3d(512, 128, kernel_size=3, padding=1, stride=1)
        self.st_res_3b_1 = STConv3d(128, 128, kernel_size=3, padding=1, stride=1)
        self.st_res_3b_2 = STConv3d(128, 128, kernel_size=3, padding=1, stride=1)

        self.st_res_4a_1 = STConv3d(128, 256, kernel_size=3, padding=1, stride=1)
        self.st_res_4a_2 = STConv3d(256, 256, kernel_size=3, padding=1, stride=1)
        self.st_res_4a_down = STConv3d(128, 256, kernel_size=3, padding=1, stride=1)

        self.st_res_4b_1 = STConv3d(256, 256, kernel_size=3, padding=1, stride=1)
        self.st_res_4b_2 = STConv3d(256, 256, kernel_size=3, padding=1, stride=1)

        self.st_res_5a_1 = STConv3d(256, 512, kernel_size=3, padding=1, stride=1)
        self.st_res_5a_2 = STConv3d(512, 512, kernel_size=3, padding=1, stride=1)
        self.st_res_5a_down = STConv3d(256, 512, kernel_size=3, padding=1, stride=1)

        self.st_res_5b_1 = STConv3d(512, 512, kernel_size=3, padding=1, stride=1)
        self.st_res_5b_2 = STConv3d(512, 512, kernel_size=3, padding=1, stride=1)
        # ---------------  ST 2+1 d end --------------------

        ''''''

        # --------------- rel branch --------------------
        self.rel_inception_3c_double_3x3_2 = BasicConv_2d(96, 96, kernel_size=3, padding=1, stride=2)
        self.rel_inception_3c_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # inception_4a
        self.rel_inception_4a_1x1 = BasicConv_2d(96, 224, kernel_size=1)

        self.rel_inception_4a_3x3_reduce = BasicConv_2d(96, 64, kernel_size=1)
        self.rel_inception_4a_3x3 = BasicConv_2d(64, 96, kernel_size=3, padding=1)

        self.rel_inception_4a_double_3x3_reduce = BasicConv_2d(96, 96, kernel_size=1)
        self.rel_inception_4a_double_3x3_1 = BasicConv_2d(96, 128, kernel_size=3, padding=1)
        self.rel_inception_4a_double_3x3_2 = BasicConv_2d(128, 128, kernel_size=3, padding=1)

        self.rel_inception_4a_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.rel_inception_4a_pool_proj = BasicConv_2d(96, 128, kernel_size=1)

        # inception_4b
        self.rel_inception_4b_1x1 = BasicConv_2d(576, 192, kernel_size=1)

        self.rel_inception_4b_3x3_reduce = BasicConv_2d(576, 96, kernel_size=1)
        self.rel_inception_4b_3x3 = BasicConv_2d(96, 128, kernel_size=3, padding=1, stride=2)

        self.rel_inception_4b_double_3x3_reduce = BasicConv_2d(576, 96, kernel_size=1)
        self.rel_inception_4b_double_3x3_1 = BasicConv_2d(96, 128, kernel_size=3, padding=1)
        self.rel_inception_4b_double_3x3_2 = BasicConv_2d(128, 128, kernel_size=3, padding=1, stride=2)

        self.rel_inception_4b_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.rel_inception_4b_pool_proj = BasicConv_2d(576, 128, kernel_size=1)

        # --------------- rel branch end --------------------

        self.global_pool_2d_pre = nn.AvgPool2d(kernel_size=7, stride=1, ceil_mode=True)
        self.global_pool2D_reshape_consensus = nn.AvgPool3d((int(self.num_frames/1), 1, 1), stride=1, ceil_mode=True)

        self.global_pool_3d = nn.AvgPool3d((int(self.num_frames/4), 7, 7), stride=1, ceil_mode=True)
        self.dropout = nn.Dropout(p=self.dropout)


        self.predict2d = nn.Linear(1120, num_classes)
        self.predict3d = nn.Linear(512, num_classes)
        self.predict_rel = nn.Linear(832, num_classes)


        self.linear2 = nn.Linear(num_classes, num_classes)

        if self.pretrained_parts is not None:
            self.linear = nn.Linear(2464, num_classes)
            self.new_fc = None

        if self.new_fc is None:
            xavier_uniform_(self.linear.weight)
            constant_(self.linear.bias, 0)
        else:
            xavier_uniform_(self.new_fc.weight)
            constant_(self.new_fc.bias, 0)

    def forward(self, x):
        # xx = self.process(x)
        # print(xx.size())


        x = self.conv1_7x7(x)
        x = self.pool1_3x3(x)
        x = self.conv2_3x3_reduce(x)
        x = self.conv2_3x3(x)
        x = self.pool2_3x3(x)
        # print(x.size(),'x')

        # inception_3a
        branch_3a_1 = self.inception_3a_1x1(x)
        branch_3a_2 = self.inception_3a_3x3_reduce(x)
        branch_3a_2 = self.inception_3a_3x3(branch_3a_2)
        branch_3a_3 = self.inception_3a_double_3x3_reduce(x)
        branch_3a_3 = self.inception_3a_double_3x3_1(branch_3a_3)
        branch_3a_3 = self.inception_3a_double_3x3_2(branch_3a_3)
        branch_3a_4 = self.inception_3a_pool(x)
        branch_3a_4 = self.inception_3a_pool_proj(branch_3a_4)
        branch_3a_output = torch.cat((branch_3a_1, branch_3a_2, branch_3a_3, branch_3a_4),1)
        # print(branch_3a_output.size(), 'branch_3a_output')

        # inception_3b
        branch_3b_1 = self.inception_3b_1x1(branch_3a_output)
        branch_3b_2 = self.inception_3b_3x3_reduce(branch_3a_output)
        branch_3b_2 = self.inception_3b_3x3(branch_3b_2)
        branch_3b_3 = self.inception_3b_double_3x3_reduce(branch_3a_output)
        branch_3b_3 = self.inception_3b_double_3x3_1(branch_3b_3)
        branch_3b_3 = self.inception_3b_double_3x3_2(branch_3b_3)
        branch_3b_4 = self.inception_3b_pool(branch_3a_output)
        branch_3b_4 = self.inception_3b_pool_proj(branch_3b_4)
        branch_3b_output = torch.cat((branch_3b_1, branch_3b_2, branch_3b_3, branch_3b_4), 1)
        # print(branch_3b_output.size(), 'branch_3b_output')

        # inception_3c
        branch_3c_1 = self.inception_3c_double_3x3_reduce(branch_3b_output)
        branch_3c_1 = self.inception_3c_double_3x3_1(branch_3c_1)
        # print(branch_3c_1.size(), 'branch_3c_1')
        branch_3c_1_0 = branch_3c_1
        # before 3d need to transpose the size of input
        branch_3c_1 = torch.transpose(branch_3c_1.view((-1, self.num_frames) + branch_3c_1.size()[1:]), 1, 2)
        # print(branch_3c_1.size(), 'branch_3c_1')


        # ---------------------------conv3d--------------------------------------
        res_3a_2 = self.res_3a_2(branch_3c_1)       # elment wise should no BN & ReLU
        res_3b_1 = self.res_3b_1(res_3a_2)
        res_3b_2 = self.res_3b_2(res_3b_1)
        res_3b = torch.add(res_3a_2, 1, res_3b_2)
        # print(res_3b.size(), 'res_3b')

        res_4a_1 = self.res_4a_1(res_3b)
        res_4a_2 = self.res_4a_2(res_4a_1)
        res_4a_down = self.res_4a_down(res_3b)
        res_4a = torch.add(res_4a_2, 1, res_4a_down)
        # print(res_4a.size(), 'res_4a')

        res_4b_1 = self.res_4b_1(res_4a)
        res_4b_2 = self.res_4b_2(res_4b_1)
        res_4b = torch.add(res_4b_2, 1, res_4a)
        # print(res_4b.size(), 'res_4b')

        res_5a_1 = self.res_5a_1(res_4b)
        res_5a_2 = self.res_5a_2(res_5a_1)
        res_5a_down = self.res_5a_down(res_4b)
        res_5a = torch.add(res_5a_2, 1, res_5a_down)
        # print(res_5a.size(), 'res_5a')

        res_5b_1 = self.res_5b_1(res_5a)
        res_5b_2 = self.res_5b_2(res_5b_1)
        res_5b = torch.add(res_5b_2, 1, res_5a)
        # print(res_5b.size(), 'res_5b')
        # --------------------------- conv3d end ------------------------------
        ''''''

        # ---------------------------ST 2+1 d--------------------------------------
        st_res_3a_2 = self.st_res_3a_2(res_5b)       # elment wise should no BN & ReLU
        st_res_3b_1 = self.st_res_3b_1(st_res_3a_2)
        st_res_3b_2 = self.st_res_3b_2(st_res_3b_1)
        st_res_3b = torch.add(st_res_3a_2, 1, st_res_3b_2)
        # print(st_res_3b.size(), 'res_3b')

        st_res_4a_1 = self.st_res_4a_1(st_res_3b)
        st_res_4a_2 = self.st_res_4a_2(st_res_4a_1)
        st_res_4a_down = self.st_res_4a_down(st_res_3b)
        st_res_4a = torch.add(st_res_4a_2, 1, st_res_4a_down)
        # print(st_res_4a.size(), 'res_4a')

        st_res_4b_1 = self.st_res_4b_1(st_res_4a)
        st_res_4b_2 = self.st_res_4b_2(st_res_4b_1)
        st_res_4b = torch.add(st_res_4b_2, 1, st_res_4a)
        # print(st_res_4b.size(), 'res_4b')

        st_res_5a_1 = self.st_res_5a_1(st_res_4b)
        st_res_5a_2 = self.st_res_5a_2(st_res_5a_1)
        st_res_5a_down = self.st_res_5a_down(st_res_4b)
        st_res_5a = torch.add(st_res_5a_2, 1, st_res_5a_down)
        # print(st_res_5a.size(), 'res_5a')

        st_res_5b_1 = self.st_res_5b_1(st_res_5a)
        st_res_5b_2 = self.st_res_5b_2(st_res_5b_1)
        st_res_5b = torch.add(st_res_5b_2, 1, st_res_5a)
        # print(st_res_5b.size(), 'res_5b')
        # --------------------------- ST 2+1 d end ------------------------------
        ''''''
        # branch 2d
        # inception_3c
        inception_3c_double_3x3_2 = self.inception_3c_double_3x3_2(branch_3c_1_0)
        # print(inception_3c_double_3x3_2.size(), 'inception_3c_double_3x3_2')

        inception_3c_pool = self.inception_3c_pool(branch_3b_output)
        # print(inception_3c_pool.size(), 'inception_3c_pool')

        inception_3c_output = torch.cat((inception_3c_double_3x3_2, inception_3c_pool), 1)
        # print(inception_3c_output.size(), 'inception_3c_output')



        # inception_4a
        inception_4a_1x1 = self.inception_4a_1x1(inception_3c_output)
        inception_4a_3x3_reduce = self.inception_4a_3x3_reduce(inception_3c_output)
        inception_4a_3x3 = self.inception_4a_3x3(inception_4a_3x3_reduce)
        inception_4a_double_3x3_reduce = self.inception_4a_double_3x3_reduce(inception_3c_output)
        inception_4a_double_3x3_1 = self.inception_4a_double_3x3_1(inception_4a_double_3x3_reduce)
        inception_4a_double_3x3_2 = self.inception_4a_double_3x3_2(inception_4a_double_3x3_1)
        inception_4a_pool = self.inception_4a_pool(inception_3c_output)
        inception_4a_pool_proj = self.inception_4a_pool_proj(inception_4a_pool)
        inception_4a_output = torch.cat((inception_4a_1x1, inception_4a_3x3, inception_4a_double_3x3_2, inception_4a_pool_proj), 1)
        # print(inception_4a_output.size(), 'inception_4a_output')

        # inception_4b
        inception_4b_1x1 = self.inception_4b_1x1(inception_4a_output)
        inception_4b_3x3_reduce = self.inception_4b_3x3_reduce(inception_4a_output)
        inception_4b_3x3 = self.inception_4b_3x3(inception_4b_3x3_reduce)
        inception_4b_double_3x3_reduce = self.inception_4b_double_3x3_reduce(inception_4a_output)
        inception_4b_double_3x3_1 = self.inception_4b_double_3x3_1(inception_4b_double_3x3_reduce)
        inception_4b_double_3x3_2 = self.inception_4b_double_3x3_2(inception_4b_double_3x3_1)
        inception_4b_pool = self.inception_4b_pool(inception_4a_output)
        inception_4b_pool_proj = self.inception_4b_pool_proj(inception_4b_pool)
        inception_4b_output = torch.cat((inception_4b_1x1, inception_4b_3x3, inception_4b_double_3x3_2, inception_4b_pool_proj), 1)
        # print(inception_4b_output.size(), 'inception_4b_output')

        # inception_4c
        inception_4c_1x1 = self.inception_4c_1x1(inception_4b_output)
        inception_4c_3x3_reduce = self.inception_4c_3x3_reduce(inception_4b_output)
        inception_4c_3x3 = self.inception_4c_3x3(inception_4c_3x3_reduce)
        inception_4c_double_3x3_reduce = self.inception_4c_double_3x3_reduce(inception_4b_output)
        inception_4c_double_3x3_1 = self.inception_4c_double_3x3_1(inception_4c_double_3x3_reduce)
        inception_4c_double_3x3_2 = self.inception_4c_double_3x3_2(inception_4c_double_3x3_1)
        inception_4c_pool = self.inception_4c_pool(inception_4b_output)
        inception_4c_pool_proj = self.inception_4c_pool_proj(inception_4c_pool)
        inception_4c_output = torch.cat((inception_4c_1x1, inception_4c_3x3, inception_4c_double_3x3_2, inception_4c_pool_proj), 1)
        # print(inception_4c_output.size(), 'inception_4c_output')

        # inception_4d
        inception_4d_1x1 = self.inception_4d_1x1(inception_4c_output)
        inception_4d_3x3_reduce = self.inception_4d_3x3_reduce(inception_4c_output)
        inception_4d_3x3 = self.inception_4d_3x3(inception_4d_3x3_reduce)
        inception_4d_double_3x3_reduce = self.inception_4d_double_3x3_reduce(inception_4c_output)
        inception_4d_double_3x3_1 = self.inception_4d_double_3x3_1(inception_4d_double_3x3_reduce)
        inception_4d_double_3x3_2 = self.inception_4d_double_3x3_2(inception_4d_double_3x3_1)
        inception_4d_pool = self.inception_4d_pool(inception_4c_output)
        inception_4d_pool_proj = self.inception_4d_pool_proj(inception_4d_pool)
        inception_4d_output = torch.cat((inception_4d_1x1, inception_4d_3x3, inception_4d_double_3x3_2, inception_4d_pool_proj), 1)
        # print(inception_4d_output.size(), 'inception_4d_output')

        # inception_4e
        inception_4e_1x1 = self.inception_4e_1x1(inception_4d_output)
        inception_4e_3x3_reduce = self.inception_4e_3x3_reduce (inception_4d_output)
        inception_4e_3x3 = self.inception_4e_3x3 (inception_4e_3x3_reduce)
        inception_4e_double_3x3_reduce = self.inception_4e_double_3x3_reduce (inception_4d_output)
        inception_4e_double_3x3_1 = self.inception_4e_double_3x3_1 (inception_4e_double_3x3_reduce)
        inception_4e_double_3x3_2 = self.inception_4e_double_3x3_2 (inception_4e_double_3x3_1)
        inception_4e_pool = self.inception_4e_pool (inception_4d_output)
        inception_4e_pool_proj = self.inception_4e_pool_proj (inception_4e_pool)
        inception_4e_output = torch.cat((inception_4e_3x3, inception_4e_double_3x3_2, inception_4e_pool), 1)
        # print(inception_4e_output.size(), 'inception_4e_output')

        global_pool_2d_pre = self.global_pool_2d_pre(inception_4e_output)
        # print(global_pool_2d_pre.size(), 'global_pool_2d_pre')      # 64 1120 1 1

        global_pool_2d = self.dropout(global_pool_2d_pre)

        global_pool_2d = torch.transpose(global_pool_2d.view((-1, self.num_frames) + global_pool_2d.size()[1:]), 1, 2)

        global_pool2D_reshape_consensus = self.global_pool2D_reshape_consensus(global_pool_2d)
        # print(global_pool2D_reshape_consensus.size(), 'global_pool2D_reshape_consensus')


        global_pool_3d = self.global_pool_3d(st_res_5b)
        # st_global_pool_3d = self.global_pool_3d(st_res_5b)
        # print(global_pool_3d.size(), 'global_pool_3d')
        global_pool_3d = self.dropout(global_pool_3d)
        # st_global_pool_3d = self.dropout(st_global_pool_3d)

        # -------------------- rel --------------------

        # For relation branch
        temp = self._calc_relation(branch_3c_1)
        # print(temp.size(), 'temp')       # same as 3D
        rel_input = torch.transpose(temp, 1, 2).contiguous()
        # print(rel_input.size(), 'rel_input')       # same as 3D
        rel_input = rel_input.view((-1, rel_input.size()[2], rel_input.size()[3], rel_input.size()[4]))
        # print(rel_input.size(), 'rel_input')       # same as 3D

        rel_inception_3c_double_3x3_2 = self.rel_inception_3c_double_3x3_2(rel_input)
        # print(rel_inception_3c_double_3x3_2.size(), '----------------------')
        # inception_4a
        rel_inception_4a_1x1 = self.rel_inception_4a_1x1(rel_inception_3c_double_3x3_2)
        rel_inception_4a_3x3_reduce = self.rel_inception_4a_3x3_reduce(rel_inception_3c_double_3x3_2)
        rel_inception_4a_3x3 = self.rel_inception_4a_3x3(rel_inception_4a_3x3_reduce)
        rel_inception_4a_double_3x3_reduce = self.rel_inception_4a_double_3x3_reduce(rel_inception_3c_double_3x3_2)
        rel_inception_4a_double_3x3_1 = self.rel_inception_4a_double_3x3_1(rel_inception_4a_double_3x3_reduce)
        rel_inception_4a_double_3x3_2 = self.rel_inception_4a_double_3x3_2(rel_inception_4a_double_3x3_1)
        rel_inception_4a_pool = self.rel_inception_4a_pool(rel_inception_3c_double_3x3_2)
        rel_inception_4a_pool_proj = self.rel_inception_4a_pool_proj(rel_inception_4a_pool)
        rel_inception_4a_output = torch.cat(
            (rel_inception_4a_1x1, rel_inception_4a_3x3, rel_inception_4a_double_3x3_2, rel_inception_4a_pool_proj), 1)
        # print(rel_inception_4a_output.size(), 'rel_inception_4a_output')

        # inception_4b
        rel_inception_4b_1x1 = self.rel_inception_4b_1x1(rel_inception_4a_output)
        rel_inception_4b_3x3_reduce = self.rel_inception_4b_3x3_reduce(rel_inception_4a_output)
        rel_inception_4b_3x3 = self.rel_inception_4b_3x3(rel_inception_4b_3x3_reduce)
        rel_inception_4b_double_3x3_reduce = self.rel_inception_4b_double_3x3_reduce(rel_inception_4a_output)
        rel_inception_4b_double_3x3_1 = self.rel_inception_4b_double_3x3_1(rel_inception_4b_double_3x3_reduce)
        rel_inception_4b_double_3x3_2 = self.rel_inception_4b_double_3x3_2(rel_inception_4b_double_3x3_1)
        rel_inception_4b_pool = self.rel_inception_4b_pool(rel_inception_4a_output)
        rel_inception_4b_pool_proj = self.rel_inception_4b_pool_proj(rel_inception_4b_pool)
        rel_inception_4b_output = torch.cat(
            (rel_inception_4b_3x3, rel_inception_4b_double_3x3_2, rel_inception_4b_pool), 1)
        # print(rel_inception_4b_output.size(), 'rel_inception_4b_output')

        rel_global_pool_2d_pre = self.global_pool_2d_pre(rel_inception_4b_output)
        # print(rel_global_pool_2d_pre.size(), 'rel_global_pool_2d_pre')

        rel_global_pool_2d = self.dropout(rel_global_pool_2d_pre)

        rel_global_pool_2d = torch.transpose(
            rel_global_pool_2d.view((-1, self.num_frames) + rel_global_pool_2d.size()[1:]), 1, 2)

        rel_global_pool2D_reshape_consensus = self.global_pool2D_reshape_consensus(rel_global_pool_2d)
        # print(rel_global_pool2D_reshape_consensus.size(), 'rel_global_pool2D_reshape_consensus')

        # -------------------------- rel end -------------------

        # ===
        global_pool_2 = global_pool2D_reshape_consensus.view(global_pool2D_reshape_consensus.size(0), -1)
        global_pool_3 = global_pool_3d.view(global_pool_3d.size(0), -1)
        rel_global_pool2D_reshape_consensus_0 = rel_global_pool2D_reshape_consensus.view(rel_global_pool2D_reshape_consensus.size(0), -1)
        out2 = self.predict2d(global_pool_2)
        out3 = self.predict3d(global_pool_3)
        out4 = self.predict_rel(rel_global_pool2D_reshape_consensus_0)

        # consesus = out2 + out3 + out4
        # consesus = self.linear2(consesus)
        # print(consesus.size())
        # ===

        global_pool = torch.cat((global_pool2D_reshape_consensus, global_pool_3d, rel_global_pool2D_reshape_consensus), 1)
        # global_pool = torch.cat((global_pool2D_reshape_consensus, st_global_pool_3d), 1)
        # global_pool = torch.cat((global_pool2D_reshape_consensus, global_pool_3d, st_global_pool_3d), 1)
        # print(global_pool.size(), 'global_pool')
        global_pool_3d = global_pool.view(global_pool.size(0), -1)


        out = self.linear(global_pool_3d)
        # print(out.size(), 'out')


        return out, out2, out3, out4


    # def _calc_relation(self, x):        # x.shape should be (N,C,T,H,W)
    #     # split = self.num_frames // 2
    #
    #     print(x.size())
    #     # temp_x = x[:, :, 0:split, :, :]       # ++++++++ need to test the shape of input x ++++++++++++++++++++++++
    #     # temp_y = x[:, :, split:, :, :]
    #     channels = x.size()[1]
    #     num = x.size()[0]
    #     # out = x[:, :, 0, :, :]
    #     # out = out.view((-1, x.size()[1], 1, x.size()[3], x.size()[4]))
    #
    #     # -----
    #     out3 = x[0, :, :, :, :]
    #     out3 = out3.view((-1, x.size()[1], x.size()[2], x.size()[3], x.size()[4]))
    #
    #     for k in range(num - 1):
    #         out2 = x[k, :, 0, :, :]
    #         out2 = out2.view((-1, 1, 1, x.size()[3], x.size()[4]))
    #
    #         # for j in range(channels):
    #         #     out = x[k, :, 0, :, :]
    #         #     out = out.view((1, -1, 1, x.size()[3], x.size()[4]))
    #
    #         for i in range(self.num_frames - 1):
    #             temp_x = x[k, 0, i, :, :]  # ++++++++ need to test the shape of input x ++++++++++++++++++++++++
    #             temp_y = x[k, 0, i + 1, :, :]
    #             print(temp_x.size())
    #             print(temp_y.size())
    #             # print(temp_y.data)
    #             temp = torch.matmul(temp_x, torch.inverse(temp_y + 1e-4))
    #             temp = temp.view((-1, 1, 1, x.size()[3], x.size()[4]))
    #             # print(temp.size())
    #             out2 = torch.cat((out2, temp), 2)
    #             # out2 = torch.cat((out2, out), 1)
    #         out3 = torch.cat((out3, out2), 0)
    #
    #     # -----
    #
    #     # for i in range(self.num_frames - 1):
    #     #     temp_x = x[:, :, i, :, :]  # ++++++++ need to test the shape of input x ++++++++++++++++++++++++
    #     #     temp_y = x[:, :, i + 1, :, :]
    #     #     # print(temp_x.size())
    #     #     # print(temp_y.size())
    #     #     temp = torch.matmul(temp_y, temp_x)
    #     #     temp = temp.view((-1, x.size()[1], 1, x.size()[3], x.size()[4]))
    #     #     # print(temp.size())
    #     #     out = torch.cat((out, temp), 2)
    #
    #     # print(temp_x.size())
    #     # print(temp_y.shape)
    #     # relation_z = torch.matmul(temp_x, temp_y)
    #     # print(relation_z.size())
    #
    #     return out3
    def _calc_relation(self, x):        # x.shape should be (N,C,T,H,W)
        # split = self.num_frames // 2

        # print(x.size())
        # temp_x = x[:, :, 0:split, :, :]       # ++++++++ need to test the shape of input x ++++++++++++++++++++++++
        # temp_y = x[:, :, split:, :, :]
        # channels = x.size()[1]
        out = x[:, :, 0, :, :]
        out = out.view((-1, x.size()[1], 1, x.size()[3], x.size()[4]))

        for i in range(self.num_frames - 1):
            temp_x = x[:, :, i, :, :]  # ++++++++ need to test the shape of input x ++++++++++++++++++++++++
            # print(temp_x.size())
            # temp_x = nn.Conv2d(temp_x.size(1), temp_x.size(1), kernel_size=3, stride=1, padding=1, bias=False).cuda()(temp_x)
            # print(temp_x.size())
            temp_y = x[:, :, i + 1, :, :]
            # print(temp_x.size())
            # print(temp_y.size())
            temp = torch.matmul(temp_y, temp_x)
            # temp = temp_y - temp_x
            temp = temp.view((-1, x.size()[1], 1, x.size()[3], x.size()[4]))
            # print(temp.size())
            out = torch.cat((out, temp), 2)

        # print(temp_x.size())
        # print(temp_y.shape)
        # relation_z = torch.matmul(temp_x, temp_y)
        # print(relation_z.size())

        return out

    def load_state_dict(self, path):
        target_weights = torch.load(path)
        own_state = self.state_dict()

        for name, param in target_weights.items():

            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    if len(param.size()) == 5 and param.size()[3] in [3, 7]:
                        own_state[name][:, :, 0, :, :] = torch.mean(param, 2)
                    else:
                        own_state[name].copy_(param)
                except Exception:
                    raise RuntimeError('While copying the parameter named {}.\
                                       whose dimensions in the model are {} and \
                                       whose dimensions in the checkpoint are {}.\
                                       '.format(name, own_state[name].size(), param.size()))
            else:
                print('{} meets error in locating parameters'.format(name))
        missing = set(own_state.keys()) - set(target_weights.keys())

        print('{} keys are not holded in target checkpoints'.format(len(missing)))

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv3d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.BatchNorm3d):  # enable BN
                bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self, mode='train'):
        resize_range_min = self.scale_size  # ASSERT 256
        if mode == 'train':
            resize_range_max = self.input_size * 320 // 224  # assert 320
            return torchvision.transforms.Compose(
                [GroupRandomResizeCrop([resize_range_min, resize_range_max], self.input_size),
                 GroupRandomHorizontalFlip(is_flow=False),  # 水平翻转
                 GroupColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05)])
        elif mode == 'val':
            return torchvision.transforms.Compose([GroupScale(resize_range_min),
                                                   GroupCenterCrop(self.input_size)])



if __name__=='__main__':

    num_frames = 16
    model = XzwModel(num_classes=101, num_frames=num_frames)


    model = model.cuda()
    data = torch.autograd.Variable(torch.rand(2, num_frames, 3, 224, 224)).cuda()
    input_var = data.view((-1, 3) + data.size()[-2:])
    print(input_var.size())
    time_start = time.time()
    out = model(input_var)[0]
    end = time.time()
    cost = end - time_start
    print(out.size(), cost)       # 2 1152 8 56 56
