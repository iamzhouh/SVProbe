import CUB_200_2011_dataloader

import config
from torchvision import models
from torch import nn
import torch
from tqdm import tqdm

# config of train attribute:
class trainattr_config:
    model = 'resnet50'
    dataset_class = 200
    device = torch.device("cuda:0")
    checkpoint_path = 'checkpoint/lr_0.01_checkpoint_363_epoch_74%_accuracy.pkl'
    fea_extract_layer = 'layer4[2].conv3'
    attribute_number = 312
    filter_num = 2048
    

CLASS, train_dataloader = CUB_200_2011_dataloader.train_data_load()

# image index to image file path
path_idx_to_imagepath = config.path + 'images.txt'
idx_to_imagepath = [] # [['1 001.Black_footed_A...046_18.jpg'], ...]
with open(path_idx_to_imagepath,'r') as f:
    for line in f:
        idx_to_imagepath.append(list(line.strip('\n').split(',')))

# The name of the file corresponding to each batch
attribute_batch = []
attribute_mini = []
for idx, tex in enumerate(train_dataloader.dataset.imgs, 1):
    attribute_mini.append(tex)
    if idx%(config.BATCH_SIZE) == 0:
        attribute_batch.append(attribute_mini)
        attribute_mini = []
if attribute_mini != []:
    attribute_batch.append(attribute_mini)

import pickle
f_read = open('imgpath_to_attr.pkl', 'rb')
imgpath_to_attr_dict = pickle.load(f_read)
f_read.close()

if trainattr_config.model == "resnet50":
    net = models.resnet50(weights=models.resnet.ResNet50_Weights.IMAGENET1K_V2)
    num_ftrs = net.fc.in_features

    # unchange in_features，change out_features=200
    net.fc = nn.Sequential(nn.Linear(num_ftrs, trainattr_config.dataset_class),
                                nn.LogSoftmax(dim = 1))
    
checkpoint = torch.load(trainattr_config.checkpoint_path)
net.load_state_dict(checkpoint['model_state_dict'])
net.eval()
net.to(trainattr_config.device)
# print(net)

grad_map = [''] # [torch.Size([batchsize, 2048, 1, 1])]
def backward_hook(module, grad_in, grad_out): # define hook function
    grad_map[0] = grad_out[0]
exec("net.%s.register_full_backward_hook(backward_hook)" % (trainattr_config.fea_extract_layer))

# init filter attribute
filter_attr = {}
for i in range(trainattr_config.filter_num):
    filter_attr[str(i)] = {}

number = {}
for i in range(trainattr_config.attribute_number):
    number[str(i+1)] = 0

loss_fn = nn.CrossEntropyLoss() # none

# train attribute
def train(net, loader):
    for enu, data in enumerate(tqdm(loader, desc="Training attribute"), 0):
        input, targets = data
        input, targets = input.to(trainattr_config.device), targets.to(trainattr_config.device)
        output = net(input)
        loss = loss_fn(output, targets)
        loss.backward()

        batch_path = attribute_batch[enu]
        batch_attr = []
        for tup in batch_path:
            batch_attr.append(imgpath_to_attr_dict[tup[0].split('/')[-1]])

        weight = torch.squeeze(torch.nn.functional.adaptive_avg_pool2d(grad_map[0], (1,1))) # torch.Size([32, 2048])

        for idx in range(len(batch_attr)):  # 32
            for attr_ind in batch_attr[idx]:  # '280' 
                    number[attr_ind] += 1
                    for filter_idx in range(trainattr_config.filter_num):  # 2048
                        if attr_ind in filter_attr[str(filter_idx)]:
                            filter_attr[str(filter_idx)][str(attr_ind)] += weight[idx][filter_idx].item()
                        else:
                            filter_attr[str(filter_idx)][str(attr_ind)] = weight[idx][filter_idx].item()
        pass

train(net, train_dataloader)

# import math, numpy
# for (k, v) in filter_attr.items():
#     sum = numpy.sum(numpy.exp(numpy.array(list(v.values()))))
#     for (k2, v2) in v.items():
#         filter_attr[k][k2] = math.exp(v2) / sum

for (k, v) in filter_attr.items():
    for (k1, v1) in v.items():
        filter_attr[k][k1] = v1 / (number[k1]+1)

# save dict
trained_attr_save = open('trained_attribute+.pkl', 'wb') 
pickle.dump(filter_attr, trained_attr_save)
trained_attr_save.close()

# # read TF_IDF
# f_read = open('TF_IDF.pkl', 'rb')
# TF_IDF = pickle.load(f_read)
# f_read.close()

# # score of avgpool * TF_IDF
# for (filter_id, attr_value) in filter_attr.items():
#     for (k, v) in attr_value.items():
#         filter_attr[filter_id][k] = v * TF_IDF[str(k)]

# # save
# trained_attrxTF_save = open('trained_attributexTF_IDF.pkl', 'wb')
# pickle.dump(filter_attr, trained_attrxTF_save)
# trained_attrxTF_save.close()