# Crop-CenterTrack

## Abstract
Crop-CenterTrack is a crop-based version of CenterTrack that excels in traffic monitoring context. Crop-CenterTrack performs detection and tracking on cropped region instead of on the entire frame on selected frames to speed up detection and simplify matching in tracking. For method details please refer to Crop_CenterTrack under the same folder. For details on the baselien refer to README_old.md

Contact: [weitao.xia@vanderbilt.edu]

## Environment requirements
The environment requirement is the same as baseline CenterTrack, however one might run into issues because of the outdated instructions. Specifying the following packages to earlier versions are confirmed to work on both Windows 10/11 and Linux:

-Python 3.6.13

-Pytorch 1.4.0

-torchvision 0.5.0


## Use Crop-CenterTrack
Currently only a limited range of functions offered by CenterTrack is offered on Crop-CenterTrack. The primary python file to run the tracker is crop_ct.py under src
COCO-Tracking was used to develop and test the extended version and the tracker is set to only show tracking results for car, truck, and bus category.

### Additional command line options

--crop              sets the tracker to crop mode, where as without the flag the baseline is used

--crop_size x       sets the size of the crops (defaulted to 64), as measured by pixels

--in_dir x          sets the input, which is a folder of folder of image sequences. The option --demo is still available for other input types

--crop_cycle x      sets the crop cycle, meaning that the tracker will perform (x-1) crop frames after every full frame

--out_dir x         sets the output folder, where the output will be in the format of one json file for each input folder of images


### Required flags to run:

tracking            to perform tracking rather than detection
--load_model        follow instructions for baseline to add a model
--keep_res          necessary if cropping
--in_dir or --demo  select input

### Tips on flags:
refer to debug flag in opts.py for visualization

### Command line example, with terminal opened in root dir:
python ./src/crop_ct.py tracking --load_model ./models/coco_tracking.pth --in_dir ./bench/DETRAC-train-data --crop --keep_res

will perform tracking on the dataset placed in DETRAC-train-data and saving the results to default folder ./results with each json file named by the finished date time, using pre-trained coco tracking in models

### Output format
Each output json file will be a list dumped, of the form [[total time in seconds, total frames looked], [list of dictionaries with keys: id, bbox, and class]]

