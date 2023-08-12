 # *_*coding: utf-8 *_*
 # author --liming--
import os
import shutil
import numpy as np
import config
import time
import tqdm
 
 
# file path
path_attridx_to_attrtext = config.path + 'attributes.txt'
path_idx_to_imagepath = config.path + 'images.txt'
path_imageidx_to_attribute = config.path + 'attributes/image_attribute_labels.txt'

# image index to image file path
idx_to_imagepath = [] # [['1 001.Black_footed_A...046_18.jpg'], ...]
with open(path_idx_to_imagepath,'r') as f:
    for line in f:
        idx_to_imagepath.append(list(line.strip('\n').split(',')))
idx_to_imagepath_dict = {}  # {'Black_Footed_Albatro...046_18.jpg': '1', ...}
for el in idx_to_imagepath:
    idx_to_imagepath_dict[el[0].split(' ')[1].split('/')[1]] = el[0].split(' ')[0]

# attribute index to attribute text
# attridx_to_attrtext = [] # [['2 has_bill_shape::dagger'], ...]
# with open(path_attridx_to_attrtext,'r') as f:
#     for line in f:
#         attridx_to_attrtext.append(list(line.strip('\n').split(',')))

# image index to attribute
imageidx_to_attribute = [] # [['1 1 0 3 27.7080'], ...]
with open(path_imageidx_to_attribute,'r') as f:
    for line in f:
        imageidx_to_attribute.append(list(line.strip('\n').split(',')))

# extract element of 'is_present == 1'
is_present_1 = []
for element in tqdm.tqdm(imageidx_to_attribute, desc="extract 'is_present == 1' attribute"):
    if element[0].split(' ')[2] == '1':
        is_present_1.append(element)

# extract element of 'is_present == 1' and 'certainty_id == 4'
is_present_1_certainty_id_4 = []
for element in tqdm.tqdm(is_present_1, desc="extract element of 'is_present == 1' and 'certainty_id == 4'"):
    if element[0].split(' ')[3] == '4':
        is_present_1_certainty_id_4.append(element)

# collate
collate = {}    #{'1': ['135', '137', '151', '165', '197', '293', '308'], ...}
for ele in is_present_1_certainty_id_4:
    if ele[0].split(' ')[0] in collate:
        collate[ele[0].split(' ')[0]].append(ele[0].split(' ')[1])
    else:
        collate[ele[0].split(' ')[0]] = [ ele[0].split(' ')[1] ]

imgpath_to_attr_dict = idx_to_imagepath_dict  # {'Black_Footed_Albatro...046_18.jpg': ['135', '137', '151', '165', '197', '293', '308'], ...}
for (key, value) in imgpath_to_attr_dict.items():
    if value in collate:
        imgpath_to_attr_dict[key] = collate[value]
    else:
        imgpath_to_attr_dict[key] = []

# save dict
import pickle
f_save = open('imgpath_to_attr.pkl', 'wb')
pickle.dump(imgpath_to_attr_dict, f_save)
f_save.close()

# computing term frequency (TF)
exist_attr = []
for (key, value) in imgpath_to_attr_dict.items():
    exist_attr += value

from collections import Counter
exist_attr.sort()
counter = Counter(exist_attr)
TF = counter.copy()
for (k, v) in TF.items():
    TF[k] = v / len(exist_attr)

#save TF
# f2_save = open('TF.pkl', 'wb')
# pickle.dump(TF, f2_save)
# f2_save.close()

# computing IDF
IDF = {}
for i in range(312):
    IDF[str(i+1)] = 0

for (key, value) in IDF.items():
    for (key_1, value_1) in imgpath_to_attr_dict.items():
        if key in value_1:
            IDF[key] += 1

import math
for (key, value) in IDF.items():
    IDF[key] = math.log(len(imgpath_to_attr_dict)/(value+1))

# save IDF
idf_save = open('IDF.pkl', 'wb')
pickle.dump(IDF, idf_save)
idf_save.close()

# computing TF-IDF
TF_IDF = {}
for (k, v) in IDF.items():
    TF_IDF[k] = TF[k] * v
    pass

# # norm
# max = max(TF_IDF.values())
# min = min(TF_IDF.values())
# for (k, v) in TF_IDF.items():
#     TF_IDF[k] = (v - min)/(max - min)

#save TF-IDF
tf_idf_save = open('TF_IDF.pkl', 'wb')
pickle.dump(TF_IDF, tf_idf_save)
tf_idf_save.close()