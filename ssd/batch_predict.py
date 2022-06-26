import os
import json
import time

import torch
from PIL import Image
#import matplotlib.pyplot as plt

import transforms
from src import SSD300, Backbone
from draw_box_utils import draw_box


def create_model(num_classes):
    backbone = Backbone()
    model = SSD300(backbone=backbone, num_classes=num_classes)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # create model
    # 目标检测数 + 背景
    num_classes = 6 + 1
    model = create_model(num_classes=num_classes)

    # load train weights
    train_weights = "./save_weights/ssd300.pth"
    model.load_state_dict(torch.load(train_weights, map_location=device)['model'])
    model.to(device)

    # read class_indict
    json_path = "./pascal_voc_classes.json"
    assert os.path.exists(json_path), "file '{}' dose not exist.".format(json_path)
    json_file = open(json_path, 'r')
    class_dict = json.load(json_file)
    json_file.close()
    category_index = {v: k for k, v in class_dict.items()}
    
    res = []
    # load image
    imgs_root = "/home/kzhang/datasets/fault_check/test/IMAGES"
    assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
    # 读取指定文件夹下所有jpg图像路径
    img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".jpg")]
    for img_path in img_path_list:
        original_img = Image.open(img_path)

        # from pil image to tensor, do not normalize image
        data_transform = transforms.Compose([transforms.Resize(),
                                            transforms.ToTensor(),
                                            transforms.Normalization()])
        img, _ = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        model.eval()  # 进入验证模式
        with torch.no_grad():

            #time_start = time_synchronized()
            predictions = model(img.to(device))[0]  # bboxes_out, labels_out, scores_out
            #time_end = time_synchronized()
            #print("inference+NMS time: {}".format(time_end - time_start))

            predict_boxes = predictions[0].to("cpu").numpy()
            predict_boxes[:, [0, 2]] = predict_boxes[:, [0, 2]] * original_img.size[0]
            predict_boxes[:, [1, 3]] = predict_boxes[:, [1, 3]] * original_img.size[1]
            predict_classes = predictions[1].to("cpu").numpy()
            predict_scores = predictions[2].to("cpu").numpy()

            if len(predict_boxes) == 0:
                print("没有检测到任何目标!")
            for [box, classes, scores] in zip(predict_boxes, predict_classes, predict_scores):
                box = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                res.append([int(img_path.split('/')[-1].split('.')[0]), box, classes-1, round(scores, 8)])
    #print(res)
    import pandas as pd
    res = pd.DataFrame(res, columns = ['image_id', 'bbox', 'category_id', 'confidence'])
    res = res.sort_values(by='image_id')
    res = res.reset_index(drop=True)
    res.to_csv('submission.csv', index=None)

if __name__ == "__main__":
    main()
