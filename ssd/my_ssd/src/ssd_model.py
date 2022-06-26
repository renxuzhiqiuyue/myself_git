from tkinter.messagebox import NO
from attr import s
from sympy import im
import torch
from torch import nn, Tensor
from torch.jit.annotations import List

from .backbone import resnet50
from .utils import dboxes300_coco, Encode, PostProcess


class Backbone(nn.Module):

    def __init__(self, pretrain_path=None):
        super(Backbone, self).__init__()
        net = resnet50()
        self.out_channels = [1024, 512, 512, 256, 256, 256]

        if pretrain_path is not None:
            net.load_state_dict(torch.load(pretrain_path))

        self.feature_extractor = nn.Sequential(*list(net.children())[:7])

        conv4_block1 = self.feature_extractor[-1][0]

        # 修改conv4_block1的步距，从2->1
        