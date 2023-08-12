import pickle
from torchvision import models
from torch import nn
import torch
from tqdm import tqdm
import config
import cv2

import CUB_200_2011_dataloader

# config of test attribute:
class testattr_config:
    model = 'resnet50'
    dataset_class = 200
    device = torch.device("cuda:1")
    checkpoint_path = 'checkpoint/lr_0.01_checkpoint_363_epoch_74%_accuracy.pkl'
    fea_extract_layer = 'layer4[2].conv3'
    attribute_number = 312
    gen_attr_num = 4
    filter_num = 2048
    filter_top = int(filter_num * 0.01)

f_read = open('trained_attribute+.pkl', 'rb')
filter_attr = pickle.load(f_read)
f_read.close()
# dict to tensor
tensor_h = len(filter_attr)
tensor_w = testattr_config.attribute_number
tensor_filter_attr = torch.zeros(tensor_h, tensor_w)
for (k, v) in filter_attr.items():
    for (k1, v1) in v.items():
        tensor_filter_attr[int(k)][int(k1)-1] = v1

# read TF_IDF
# f_read = open('TF_IDF.pkl', 'rb')
f_read = open('IDF.pkl', 'rb')
TF_IDF = pickle.load(f_read)
f_read.close()
# dict to tensor
tensor_TF_IDF = torch.zeros(1, len(TF_IDF))
for (k, v) in TF_IDF.items():
    tensor_TF_IDF[0][int(k)-1] = v

# tensor_filter_attr = torch.nn.functional.normalize(tensor_filter_attr, dim = 1)
# tensor_TF_IDF = torch.nn.functional.normalize(tensor_TF_IDF)

tensor_filter_attr = torch.nn.functional.normalize(tensor_filter_attr, dim = 1)

# read : attribute index to attribute text
path_attridx_to_attrtext = config.path + 'attributes.txt'
attridx_to_attrtext = [] # [['2 has_bill_shape::dagger'], ...]
with open(path_attridx_to_attrtext,'r') as f:
    for line in f:
        attridx_to_attrtext.append(list(line.strip('\n').split(',')))
dict_attrid_to_test = {}
for list in attridx_to_attrtext:
    dict_attrid_to_test[list[0].split()[0]] = list[0].split()[1]


# dataloader
CLASS, train_dataloader = CUB_200_2011_dataloader.train_data_load()
_, test_dataloader = CUB_200_2011_dataloader.test_data_load()

# create、load model
if testattr_config.model == "resnet50":
    net = models.resnet50(weights=models.resnet.ResNet50_Weights.IMAGENET1K_V2)
    num_ftrs = net.fc.in_features

    # unchange in_features，change out_features=200
    net.fc = nn.Sequential(nn.Linear(num_ftrs, testattr_config.dataset_class),
                                nn.LogSoftmax(dim = 1))

checkpoint = torch.load(testattr_config.checkpoint_path)
net.load_state_dict(checkpoint['model_state_dict'])
net.eval()
net.to(testattr_config.device)
# print(net)

grad_map = [''] # [torch.Size([batchsize, 2048, 1, 1])]
def backward_hook(module, grad_in, grad_out): # define hook function
    grad_map[0] = grad_out[0]
exec("net.%s.register_full_backward_hook(backward_hook)" % (testattr_config.fea_extract_layer))

feature_map = [''] # [torch.Size([batchsize, 2048, 1, 1])]
def forward_hook(module, fea_in, fea_out): # define hook function
    feature_map[0] = fea_out
exec("net.%s.register_forward_hook(forward_hook)" % (testattr_config.fea_extract_layer))

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

loss_fn = nn.CrossEntropyLoss() # none

# test attribute
def test(net, loader):
    test_result = {}
    for enu, data in enumerate(tqdm(loader, desc="Testing attribute"), 0):
        input, targets = data
        input, targets = input.to(testattr_config.device), targets.to(testattr_config.device)
        output = net(input)
        _, predicted = torch.max(output.data, dim = 1)

        loss = loss_fn(output, targets)
        loss.backward()

        grad = torch.squeeze(torch.nn.functional.adaptive_avg_pool2d(grad_map[0], (1,1)))
        grad = grad.unsqueeze(-1).to(testattr_config.device)
        
        for id in range(len(input)):
            sample_computing = tensor_filter_attr.clone().to(testattr_config.device)

            score = grad[id]
            # score = torch.nn.functional.normalize(score)

            sample_computing = sample_computing * score
            # sample_computing = sample_computing * class_weight

            # for (filter_id, attr_prob) in sample_computing.items():
                # for (attr_id, wei) in attr_prob.items():
                #     sample_computing[filter_id][attr_id] = wei * class_weight[int(filter_id)].clone().item() * avgpool_out[id][int(filter_id)].clone().item()

                # sample_computing[str(filter_id)] = attr_prob.clone() * class_weight[int(filter_id)].clone().item() * avgpool_out[id][int(filter_id)].clone().item()
                # sample_computing[filter_id].update((key, value * class_weight[int(filter_id)].clone().item() * avgpool_out[id][int(filter_id)].clone().item()) for key, value in sample_computing[filter_id].items())

            # final_attr = {}

            # from collections import Counter
            # for (ke, va) in sample_computing.items():
            #     final_attr = dict(Counter(final_attr) + Counter(va))

            # final_attr = torch.zeros(testattr_config.attribute_number)
            # for (key, v) in sample_computing.items():
            #     final_attr += v.clone()
            #     pass

            # sum
            final_attr = sample_computing.sum(0)

            
            # norm
            final_attr = torch.nn.functional.normalize(final_attr, dim = 0)

            # prod
            # final_attr = (10*sample_computing).cumprod(dim=0)[-1]

            # norm
            # final_attr_norm = (final_attr-final_attr.min())/(final_attr.max()-final_attr.min()) 

            # exp
            # final_attr_exp = torch.exp(final_attr_norm)

            # change to dic
            attr_to_prob = {}
            for i in range(final_attr.shape[0]):
                attr_to_prob[str(i+1)] = final_attr[i].item()

            # get max and min
            # attr_max = max(attr_to_prob.values())
            # attr_min = min(attr_to_prob.values())

            # get attr_score sum
            attr_sum = sum(attr_to_prob.values())

            # According to ‘value’ sorting
            attr_to_prob_sort = sorted(attr_to_prob.items(),  key=lambda d: d[1], reverse=True)

            # save to dict of final result
            test_result[attribute_batch[enu][id][0].split('/')[-1]] = attr_to_prob_sort
            # test_result[attribute_batch[enu][id][0]] = attr_to_prob_sort

            import matplotlib.pyplot as plt
            plt.figure(figsize=(10,10), dpi=100)
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1.5, hspace=None) # subplot space              

            plt.subplot(3,2,1)
            # draw image
            img = plt.imread(attribute_batch[enu][id][0])
            img = cv2.resize(img, (256, 256))
            
            plt.title("label:"+str(targets[id].item())+",predict:"+str(predicted[id].item()))
            plt.imshow(img)
            
            plt.subplot(3,2,2)
            # draw table
            # plt.rcParams["font.sans-serif"]=['SimHei']
            plt.rcParams["axes.unicode_minus"]=False
            # plt.xticks(fontproperties='Times New Roman', size='small',rotation=30)
            
            import numpy as np
            hot_map_list = []
            for e, da in enumerate(attr_to_prob_sort):
                plt.barh(da[0]+'::'+dict_attrid_to_test[da[0]], da[1])
                # plt.xticks(da[0], dict_attrid_to_test[da[0]],rotation = 30)

                attr_score_of_every_filter = sample_computing[:,int(da[0])-1].unsqueeze(-1).unsqueeze(-1)

                index = torch.topk(attr_score_of_every_filter, testattr_config.filter_top, dim=0) # 索引出过滤器得分的top
                threshold = index.values[-1].item() # 阈值
                zero = torch.zeros_like(attr_score_of_every_filter)
                # 小于threshold的用zero(0)替换,否则不变
                filter_score = torch.where(attr_score_of_every_filter < threshold, zero, attr_score_of_every_filter)

                hot_map = (filter_score* feature_map[0][id]).sum(0)
                hot_map = torch.nn.functional.relu(hot_map)
                hot_map_list.append(hot_map)

                if e == testattr_config.gen_attr_num-1:
                    break
            plt.title("attr_score")
            # plt.xlabel("attr")
            # plt.ylabel("score")
            
            for en, hm in enumerate(hot_map_list):
                plt.subplot(3, 2, en + 3)
                img_hm = hm.cpu().detach().numpy()
                # img_hm = np.transpose(img_hm, (1,2,0)) # 把channel那一维放到最后
                img_hm = img_hm - np.min(img_hm)
                img_hm = img_hm / (1e-20 + np.max(img_hm))
                img_hm = np.uint8(255 * img_hm)
                img_hm = cv2.resize(img_hm, (256, 256))
                img_hm = cv2.merge((img_hm,img_hm,img_hm))
                img_hm = cv2.applyColorMap(img_hm, cv2.COLORMAP_JET)
                img_hm = cv2.cvtColor(img_hm, cv2.COLOR_BGR2RGB)
                try:
                    img_hm = np.uint8(0.3 * img_hm + 0.5 * img)
                except:
                    pass
                    continue
                plt.imshow(img_hm)
            
            # save
            # plt.imsave('experiments/'+attribute_batch[enu][id][0].split('/')[-1], img)
            plt.savefig('experiments/'+attribute_batch[enu][id][0].split('/')[-1])
            plt.close()

            pass
    
    f_save = open('test_attr_result.pkl', 'wb')
    pickle.dump(test_result, f_save)
    f_save.close()

test(net, train_dataloader)
# test(net, test_dataloader)