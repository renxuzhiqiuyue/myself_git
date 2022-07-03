import os
from re import L

import torch
import torch.nn as nn

from src import SSD300, Backbone

def createweight(weight):
    weightTrans = torch.cat([weight, weight, weight], dim = 1)
    weightTrans = torch.nn.Parameter(torch.cat([weightTrans, weightTrans, weightTrans], dim = 0))
    return weightTrans

class PdcNet(nn.Module):
    def __init__(self):
        super(PdcNet, self).__init__()
        pdcnets = []
        weights = [torch.FloatTensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]).reshape(1, 1, 3, 3),
                    torch.FloatTensor([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]).reshape(1, 1, 3, 3),
                    torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).reshape(1, 1, 3, 3),
                    torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).reshape(1, 1, 3, 3),
                    # torch.FloatTensor([[1, 0], [-1, 0]]).reshape(1, 1, 2, 2),
                    # torch.FloatTensor([[0, -1], [1, 0]]).reshape(1, 1, 2, 2),
                    ]
        for dilation in range(1, 3):
            for weight in weights:
                pdcnets.append(self.create_conv(createweight(weight), dilation))
        # print(pdcnets)
        # print(type(pdcnets), type(pdcnets[0]))
        self.layers = nn.ModuleList(pdcnets)

    def create_conv(self, weight, dilation=1):        
        kernel_size = int(weight.shape[-1])
        _bias = torch.nn.Parameter(torch.zeros(kernel_size))
        
        edgeConv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=kernel_size, stride=1, padding=1, dilation=dilation)
        # 自定义权重
        edgeConv.weight = weight
        edgeConv.bias = _bias
        return edgeConv

    def forward(self, idx, x):
        idx %= 12
        layer = self.layers[idx%2]
        layer2 = self.layers[(idx+1)%2]
        return layer2(layer(x))

def create_model(num_classes=21):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    # pre_train_path = "./src/resnet50.pth"
    pdcnet = PdcNet()
    backbone = Backbone()
    model = SSD300(backbone=backbone, pdcnet=pdcnet, num_classes=num_classes)

    # https://ngc.nvidia.com/catalog/models -> search ssd -> download FP32
    pre_ssd_path = "./src/nvidia_ssdpyt_fp32.pt"

    if os.path.exists(pre_ssd_path) is False:
        raise FileNotFoundError("nvidia_ssdpyt_fp32.pt not find in {}".format(pre_ssd_path))
    pre_model_dict = torch.load(pre_ssd_path, map_location='cpu')
    pre_weights_dict = pre_model_dict["model"]

    # 删除类别预测器权重，注意，回归预测器的权重可以重用，因为不涉及num_classes
    del_conf_loc_dict = {}
    for k, v in pre_weights_dict.items():
        split_key = k.split(".")
        if "conf" in split_key:
            continue
        del_conf_loc_dict.update({k: v})

    missing_keys, unexpected_keys = model.load_state_dict(del_conf_loc_dict, strict=False)
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)

    return model


def main(parser_data):

    model = create_model(num_classes=6+1)

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0005,
                                momentum=0.9, weight_decay=0.0005)
    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=5,
                                                   gamma=0.3)


if __name__ == '__main__':
    #model = create_model(num_classes=6+1)
    # backbone = Backbone()
    # print(backbone)
    # print(hasattr(backbone, "out_channels"))


    # pdcnet = Net()
    # print(pdcnet)
    # from torchsummary import summary
    # summary(pdcnet, input_size=[(3, 300, 300)], batch_size=2, device="cpu")

    # model = create_model(num_classes=6+1)

    # from torchsummary import summary
    # model.eval()
    # x = torch.randn(2, 3, 300, 300)
    # target = model(x)
    # print(target)
    # print(len(target), target[0].shape)
    # summary(model, input_size=[(3, 300, 300)], batch_size=2, device="cpu")
    # params = [p for p in model.parameters() if p.requires_grad]
    # print(len(params))

    import numpy as np
    input = np.random.randint(low=0, high=1, size=2*3*300*300).reshape(2, 3, 300, 300).astype(np.float32)
    input = torch.from_numpy(input)
    print(input)
    print(input.shape, type(input))
    idx = 1

    pdcnet = PdcNet()
    backbone = Backbone()
    model = SSD300(backbone=backbone, pdcnet=None, num_classes=6+1)
    model.eval()
    output = model(idx, input)
    print(output)