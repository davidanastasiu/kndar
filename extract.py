import sys
import os
from os.path import isfile, join
kapao_path = join(os.getcwd(), 'kapao')
sys.path.append(kapao_path)  # add kapao/ to path

import argparse
import torch
import yaml
import numpy as np

from kapao.utils.general import check_img_size
from kapao.utils.datasets import LoadImages
from kapao.models.experimental import attempt_load
from kapao.val import run_nms, post_process_batch

from features import process_dataset
from utilities import select_device

"""
Naturalistic Driving Action Recognition
Track 3 for the 2022 AI City Challenge
"""

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  # extract options
  parser.add_argument('-ds', '--dataset', default='', help='path to dataset root (e.g., A1 folder)')
  parser.add_argument('-v', '--view', type=str, default='dash', help='which camera view (dash, rear, right, or all)')
  parser.add_argument('-f', '--face', action='store_true', help='process face features')
  parser.add_argument('-s', '--skip', type=int, default=5, help="process a key frame every skip frames")
  parser.add_argument('-p', '--output-path', default='', help='path to store extracted features')

  # video output options
  parser.add_argument('--color', type=int, nargs='+', default=[255, 255, 255], help='pose color')
  parser.add_argument('-d', '--display', action='store_true', help='display inference results')
  parser.add_argument('-w', '--write_video', action='store_true', help='write video file, unless --display')
  parser.add_argument('--fps-size', type=int, default=1)
  parser.add_argument('--start', type=int, default=0, help='start time (s)')
  parser.add_argument('--end', type=int, default=-1, help='end time (s), -1 for remainder of video')
  parser.add_argument('--kp-size', type=int, default=2, help='keypoint circle size')
  parser.add_argument('--kp-thick', type=int, default=1, help='keypoint circle thickness')
  parser.add_argument('--line-thick', type=int, default=2, help='line thickness')
  parser.add_argument('--alpha', type=float, default=0.4, help='pose alpha')
  parser.add_argument('--kp-obj', action='store_true', help='plot keypoint objects only')
  parser.add_argument('--csv', action='store_true', help='write results so csv file')

  # model options
  parser.add_argument('--data', type=str, default='data/coco-kp.yaml')
  parser.add_argument('--imgsz', type=int, default=640)
  parser.add_argument('--weights', default='kapao_s_coco.pt')
  parser.add_argument('--device', default='', help='cuda device, i.e. 0 or cpu')
  parser.add_argument('--half', action='store_true')
  parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
  parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
  parser.add_argument('--no-kp-dets', action='store_true', help='do not use keypoint objects')
  parser.add_argument('--conf-thres-kp', type=float, default=0.5)
  parser.add_argument('--conf-thres-kp-person', type=float, default=0.2)
  parser.add_argument('--iou-thres-kp', type=float, default=0.45)
  parser.add_argument('--overwrite-tol', type=int, default=50)
  parser.add_argument('--scales', type=float, nargs='+', default=[1])
  parser.add_argument('--flips', type=int, nargs='+', default=[-1])

  args = parser.parse_args()

  weights = args.weights if isfile(args.weights) else join(kapao_path, args.weights)
  dpath = args.data if isfile(args.data) else join(kapao_path, args.data)

  with open(dpath) as f:
    data = yaml.safe_load(f)  # load data dict

  # add inference settings to data dict
  data['imgsz'] = args.imgsz
  data['conf_thres'] = args.conf_thres
  data['iou_thres'] = args.iou_thres
  data['use_kp_dets'] = not args.no_kp_dets
  data['conf_thres_kp'] = args.conf_thres_kp
  data['iou_thres_kp'] = args.iou_thres_kp
  data['conf_thres_kp_person'] = args.conf_thres_kp_person
  data['overwrite_tol'] = args.overwrite_tol
  data['scales'] = args.scales
  data['flips'] = [None if f == -1 else f for f in args.flips]
  data['count_fused'] = False

  # check some inputs
  if not args.dataset:
    print("Missing dataset: see the --dataset option.")
    exit()
  # check other parameters
  if args.view.lower() not in ["rear", "dash", "right", "all"]:
    print("Invalid view mode: try one of dash, rear, right, or all.")
    exit()
  if args.skip < 1:
    print("Invalid skip argument. It should be at least 1.")
    exit()

  view = args.view.strip().lower()

  # get ready for processing
  ndevices = len(args.device.strip().split(","))
  device = select_device(args.device, batch_size=ndevices)
  print('Using device: {}'.format(device))

  model = attempt_load(weights, map_location=device)  # load FP32 model
  half = args.half & (device.type != 'cpu')
  if half:  # half precision only supported on CUDA
    model.half()
  stride = int(model.stride.max())  # model stride
  imgsz = check_img_size(args.imgsz, s=stride)  # check image size
  if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

  # feature extraction
  fdata, flabels = process_dataset(
    args.dataset, view, device, model, data,
    opath=args.output_path, skip=args.skip,
    img_size=imgsz, stride=stride, half=half,
    face=args.face
  )
