import mot_eval as mot
import json
import os





stat_folder = './bench/test_results/crop_7_1'
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

frame_folder = './bench/DETRAC-train-data'
frame_list = []
if os.path.isdir(frame_folder):
            ls = os.listdir(frame_folder)
            for file_name in sorted(ls):
                ext = file_name[file_name.rfind('.') + 1:].lower()
                frame_list.append(os.path.join(frame_folder, file_name))

summary, acc = [], []
tot_time = 0
tot_frame = 0
tot_mota = 0
cnt = 0
for i in range(len(result_list)):
    preds = json.load(open(result_list[i]))
    gts, md = mot.parse_labels(gt_list[i])
    summary_i, acc_i = mot.evaluate_mot(preds[1], gts, frame_list[i], md['ignored_regions'])
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
