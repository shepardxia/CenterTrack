from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from matplotlib.font_manager import json_dump

import _init_paths

import os
import sys
import cv2
import json
import copy
import numpy as np
from opts import opts
from crop_detector import Detector


image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge', 'display']


def results_format_and_output(results, time, frames, save_name):
    dets = []
    for frame in results:
        frame_result = []
        for obj in results[frame]:
            frame_result.append({
                'id': obj['tracking_id'],
                'bbox': obj['bbox'].tolist(),
                'class': 1,
            })
        dets.append(frame_result)
    ret = []
    ret.append([time, frames])
    ret.append(dets)
    json.dump(ret, open('./bench/DETRAC-coco-stat/{}.json'.format(save_name), 'w'))

    

def demo(opt, read_folder=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    detector = Detector(opt)

    if read_folder is not None:
        is_video = False
        # Demo on images sequences
        if os.path.isdir(read_folder):
            image_names = []
            ls = os.listdir(read_folder)
            for file_name in sorted(ls):
                ext = file_name[file_name.rfind('.') + 1:].lower()
                if ext in image_ext:
                    image_names.append(os.path.join(read_folder, file_name))
    if opt.demo == 'webcam' or \
            opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
        is_video = True
        # demo on video stream
        cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
    else:
        is_video = False
        # Demo on images sequences
        if os.path.isdir(opt.demo):
            image_names = []
            ls = os.listdir(opt.demo)
            for file_name in sorted(ls):
                ext = file_name[file_name.rfind('.') + 1:].lower()
                if ext in image_ext:
                    image_names.append(os.path.join(opt.demo, file_name))
        else:
            image_names = [opt.demo]

    # Initialize output video
    out = None
    out_name = opt.demo[opt.demo.rfind('/') + 1:]
    print('out_name', out_name)
    if opt.save_video:
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        out = cv2.VideoWriter('../results/{}.mp4'.format(
            opt.exp_id + '_' + out_name), fourcc, opt.save_framerate, (
            opt.video_w, opt.video_h))

    if opt.debug < 5:
        detector.pause = False
    cnt = 0
    results = {}
    tot_time = 0
    while True:
        if is_video:
            _, img = cam.read()
            if img is None:
                save_and_exit(opt, out, results, out_name)
        else:
            if cnt < len(image_names):
                img = cv2.imread(image_names[cnt])
            else:
                results_format_and_output(results, tot_time, cnt, None \
                    if read_folder is None else read_folder[len(read_folder)-5:])
                return
        cnt += 1
        # resize the original video for saving video results
        if opt.resize_video:
            img = cv2.resize(img, (opt.video_w, opt.video_h))

        # skip the first X frames of the video
        if cnt < opt.skip_first:
            continue

        #cv2.imshow('input', img)

        # track or detect the image.
        
        if(opt.crop == True):
            if(cnt % 7 == 1):
                ret, f_time = detector.run(img, 0)
            else:
                ret, f_time = detector.run(img, 1)
        else:
            ret, f_time = detector.run(img)
        
        tot_time += f_time

        # log run time
        time_str = 'frame {} |'.format(cnt)
        for stat in time_stats:
            time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        #print(results)
        print(time_str)

        # results[cnt] is a list of dicts:
        #  [{'bbox': [x1, y1, x2, y2], 'tracking_id': id, 'category_id': c, ...}]
        results[cnt] = ret['results']

        # save debug image to video
        if opt.save_video:
            out.write(ret['generic'])
            if not is_video:
                cv2.imwrite('../results/demo{}.jpg'.format(cnt),
                            ret['generic'])
        # esc to quit and finish saving video
        if cv2.waitKey(1) == 27:    
            save_and_exit(opt, out, results, out_name)
            return
    save_and_exit(opt, out, results)


def save_and_exit(opt, out=None, results=None, out_name=''):
    if opt.save_results and (results is not None):
        save_dir = '../results/{}_results.json'.format(
            opt.exp_id + '_' + out_name)
        print('saving results to', save_dir)
        json.dump(_to_list(copy.deepcopy(results)),
                  open(save_dir, 'w'))
    if opt.save_video and out is not None:
        out.release()
    sys.exit(0)


def _to_list(results):
    for img_id in results:
        for t in range(len(results[img_id])):
            for k in results[img_id][t]:
                if isinstance(results[img_id][t][k], (np.ndarray, np.float32)):
                    results[img_id][t][k] = results[img_id][t][k].tolist()
    return results


if __name__ == '__main__':
    opt = opts().init()
    folder = './bench/DETRAC-train-data'
    bench_names = []
    bench_dir = os.listdir(folder)
    folder_paths = []
    for folder_name in sorted(bench_dir):
        fl = folder_name[folder_name.rfind('.') + 1:].lower()
        folder_paths.append(os.path.join(folder, folder_name))
    for path in folder_paths:
        print(path[len(path)-5:])
    for path in folder_paths:
        demo(opt, path)
