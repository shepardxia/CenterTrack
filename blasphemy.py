"""
import numpy as np
import torch

box1 = torch.rand((5, 3, 4))
box2 = torch.rand((5, 4))
score = torch.zeros((box1.shape[0], box1.shape[1]))
box2 = box2.unsqueeze(1).repeat(1, box1.shape[1], 1)
print("box1 is ", box1)
print("box2 is ", box2)
score = box1[:,:,0] + box2[:,:,0]
print(score)
ret, keep = torch.max(score, dim=1)
print("score shape: ", score.shape)
print("ret shape", ret.shape)
print("keep shape", keep.shape)

what = torch.arange(box1.shape[0])
print("what the fuck is what", what)

print(box1[what, keep])
print("box1[what, keep] shape", box1[what, keep].shape)
"""

"""
l1 = []
l1.append([])
l1[0].append({1:"off with their fucking heads"})
print("l1 has type:", type(l1))
print("l1[0] has type:", type(l1[0]))
print("l1[0][0] has type:", type(l1[0][0]))
print("l1[0][0][1] is:", l1[0][0][1])
print("l1 is:", l1)
"""

"""
for i in range(100):
    try:
        assert i < 50
        st = "{:08}".format(i)
        print(st)
    except:
        break
"""

import mot_eval as mot
import json
import os





stat_folder = './bench/DETRAC-coco-stat'
result_list = []
if os.path.isdir(stat_folder):
            ls = os.listdir(stat_folder)
            for file_name in sorted(ls):
                ext = file_name[file_name.rfind('.') + 1:].lower()
                result_list.append(os.path.join(stat_folder, file_name))

gt_folder = './bench/DETRAC-Train-Annotations-XML-v3'
gt_list = []
if os.path.isdir(gt_folder):
            ls = os.listdir(gt_folder)
            for file_name in sorted(ls):
                ext = file_name[file_name.rfind('.') + 1:].lower()
                gt_list.append(os.path.join(gt_folder, file_name))

summary, acc = [], []
tot_time = 0
tot_frame = 0
tot_mota = 0
cnt = 0
for i in range(len(result_list)):
    preds = json.load(open(result_list[i]))
    gts, md = mot.parse_labels(gt_list[i])
    summary_i, acc_i = mot.evaluate_mot(preds[1], gts, md['ignored_regions'])
    print(summary_i)
    tot_mota += float(summary_i['mota'])
    cnt += 1
    tot_time += preds[0][0]
    tot_frame += preds[0][1]

print("time: ", tot_time)
print("frame: ", tot_frame)
print("average mota: ", tot_mota/cnt)

"""
gts, md = mot.parse_labels("./bench/DETRAC-Train-Annotations-XML-v3/MVI_40212_v3.xml")
preds = json.load(open("./results.json"))
summary, acc = mot.evaluate_mot(preds, gts, md['ignored_regions'])
print(summary)
"""
