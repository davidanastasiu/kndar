"""
Extract complex features from the videos.
"""

import os
import cv2
import torch
import imageio
import numpy as np
import face_recognition
from math import sqrt
from tqdm import tqdm
from collections import defaultdict
from os.path import isfile, isdir, join, basename, splitext

from kapao.utils.general import check_img_size
from kapao.utils.datasets import LoadImages
from kapao.models.experimental import attempt_load
from kapao.val import run_nms, post_process_batch

from datasets import parse_labels, get_label, video_view, get_video_ids
from geometry import driver_bbox, cosangle, distance


def pose_features(pose, last_pose, bbox, w=1920, h=1080):
  """
  Extract features from pose and the last pose.
    pose: current frame pose
    last_pose: last stored pose (potentially from some time ago, see --skip)
    w: image width
    h: image height
  return: vector of features for this frame
  """
  llc = (0, h)
  luc = (0, 0)
  angles = [
    cosangle(pose[0], llc, luc),           # nose, left lower corner, left upper corner (of the image)
    cosangle(pose[9], llc, luc),           # left wrist, left lower corner, left upper corner (of the image)
    cosangle(pose[10], llc, luc),          # right wrist, left lower corner, left upper corner (of the image)

    cosangle(pose[5], pose[7], pose[9]),   # left elbow (left shoulder, left elbow, left wrist)
    cosangle(pose[7], pose[5], pose[11]),  # left shoulder (left elbow)
    cosangle(pose[6], pose[8], pose[10]),  # right elbow (right shoulder, right elbow, right wrist)
    cosangle(pose[8], pose[6], pose[12]),  # right shoulder
    cosangle(pose[1], pose[0], pose[2]),   # left eye, nose, right eye
    cosangle(pose[0], pose[5], pose[6]),   # nose, left shoulder, right shoulder
    cosangle(pose[0], pose[6], pose[5]),   # nose, right shoulder, left shoulder
    cosangle(pose[0], pose[3], pose[1]),   # nose, left ear, left eye
    cosangle(pose[0], pose[4], pose[2]),   # nose, right ear, right eye
    cosangle(pose[3], pose[12], pose[6]),  # left ear, right hip, right shoulder
    cosangle(pose[4], pose[11], pose[5]),  # right ear, left hip, left shoulder
    cosangle(pose[5], pose[12], pose[6]),  # left shoulder, right hip, right shoulder
    cosangle(pose[6], pose[11], pose[5])   # right shoulder, left hip, left shoulder
  ]
  # diag = sqrt(w**2 + h**2)  # use diagonal to normalize distance from corners
  # bd = distance(bbox[:2], bbox[-2:])  # use bounding box diagonal to normalize other distances
  # use w to normalize all distances
  distances = [
    distance(pose[0], llc)/w,  # nose to lower left corner
    distance(pose[9], llc)/w,  # left wrist to lower left corner
    distance(pose[10], llc)/w, # right wrist to lower left corner
    distance(pose[0], luc)/w,  # nose to upper left corner
    distance(pose[9], luc)/w,  # left wrist to upper left corner
    distance(pose[10], luc)/w, # right wrist to upper left corner

    distance(pose[0], pose[5])/w,   # nose to left shoulder
    distance(pose[0], pose[6])/w,   # nose to right shoulder
    distance(pose[0], pose[3])/w,   # nose to left ear
    distance(pose[0], pose[4])/w,   # nose to right ear
    distance(pose[1], pose[2])/w,   # left eye to right eye
    distance(pose[3], pose[4])/w,   # left ear to right ear
    distance(pose[3], pose[9])/w,   # left ear to left wrist
    distance(pose[4], pose[10])/w,  # right ear to right wrist
    distance(pose[7], pose[11])/w,  # left elbow to left hip
    distance(pose[8], pose[12])/w,  # right elbow to right hip
    distance(pose[9], pose[11])/w,  # left wrist to left hip
    distance(pose[10], pose[12])/w, # right wrist to right hip
  ]
  positions = [
    (bbox[2] - bbox[0])/w,   # relative bounding box width
    (bbox[3] - bbox[1])/h,   # relative bounding box height
    (bbox[0] + (bbox[2] - bbox[0])/2)/w,  # relative position of bounding box center x
    (bbox[1] + (bbox[3] - bbox[1])/2)/h,  # relative position of bounding box center x
  ]
  shifts = [ distance(pose[i], last_pose[i])/(0.25*w) for i in range(11) ] # face and upper body
  last_pose_angles = [
    cosangle(last_pose[0], llc, luc),  # nose, left lower corner, left upper corner (of the image)
    cosangle(last_pose[9], llc, luc),  # left wrist, left lower corner, left upper corner (of the image)
    cosangle(last_pose[10], llc, luc), # right wrist, left lower corner, left upper corner (of the image)
    cosangle(last_pose[5], last_pose[7], last_pose[9]),   # left elbow
    cosangle(last_pose[7], last_pose[5], last_pose[11]),  # left shoulder
    cosangle(last_pose[6], last_pose[8], last_pose[10]),  # right elbow
    cosangle(last_pose[8], last_pose[6], last_pose[12]),  # right shoulder
  ]
  angle_shifts = [
    abs(angles[i] - last_pose_angles[i]) for i in range(len(last_pose_angles))
  ]

  return np.array(angles + distances + positions + shifts + angle_shifts)


def face_features(img, pose, pf):
  """
  Extract face features for the person identified in the pose.
  img: original image of the frame
  pose: pose of the person whose face we are targetting
  pf: pose features; result will be concatenated to pf
  return pf
  """
  w = int(distance(pose[3], pose[4]))
  h = int(1.5 * w)
  x, y = pose[0,:2].astype(int) # nose position
  xmin = max(x-w, 0)
  xmax = min(x+w, img.shape[1])
  ymin = max(y-h, 0)
  ymax = min(y+h, img.shape[0])
  face = img[ymin:ymax, xmin:xmax]
  landmarks = face_recognition.face_landmarks(face)
  d = 0
  if landmarks:
    d = distance(landmarks[0]['top_lip'][9], landmarks[0]['bottom_lip'][9])
  return np.hstack((pf, [d/w]))


def extract(vpath, vid, device, model, data,
    view=None, labels=None, skip=1,
    img_size=640, stride=32, half=False,
    face=False
  ):
  """
  Extract features from the given video.
  vpath: the path to video
  vid: the video id
  device: device, i.e., 'cpu' or '0' or '0,1,2,3'
  model: Torch kapao coco-kp model
  data: data structure with algorithm parameters for kapao
  view: which camera view for the given video, "dash", "rear", or "right"
  labels: list of labels for this specific video
  skip: extract features every skip frames
  img_size: the width of the image
  stride: convolution stride
  half: use half precision
  face: whether to process the face through the face_recognition package
  """
  imgsz = check_img_size(img_size, s=stride)  # check image size
  dataset = LoadImages(vpath, img_size=img_size, stride=stride, auto=True)

  if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

  cap = dataset.cap
  cap.set(cv2.CAP_PROP_POS_MSEC, 0)
  fps = cap.get(cv2.CAP_PROP_FPS)
  n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

  dataset = tqdm(dataset, desc=f'Extracting features from {vpath}', total=n)
  vfeatures = {}
  last_i = None
  for i, (path, img, im0, _) in enumerate(dataset):
    img2 = img.copy()
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img = img / 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
      img = img[None]  # expand for batch dim

    if i % skip == 0: # only apply model every skip frames
      out = model(img, augment=True, kp_flip=data['kp_flip'], scales=data['scales'], flips=data['flips'])[0]
      person_dets, kp_dets = run_nms(data, out)
      bboxes, poses, _, _, _ = post_process_batch(data, img, [], [[im0.shape[:2]]],
                               person_dets, kp_dets)
      if len(bboxes) > 0:
        # keep only predictions for driver (on the right side of screen)
        if last_i is not None: # we have a previous proper detection
          di = driver_bbox(bboxes, last_bboxes)
          if di is not None: # at least one person detected
            bboxes = [bboxes[di]]
            poses = [poses[di]]
            pf = pose_features(poses[0], last_poses[0], bboxes[0], w=w, h=h)
            if face:
              pf = face_features(im0, poses[0], pf)
            l = -1
            if labels:
              l = get_label(labels, i/fps)
            vfeatures[i] = (pf, l)
            last_bboxes = bboxes
            last_poses = poses
            last_i = i
        # update last key frame
        last_bboxes = bboxes
        last_poses = poses
        last_i = i

  cap.release()

  return vfeatures


def merge(feats):
  """
  Merge features from different views of the same video/experiment.
  The resulting feature vectors will have 3x the dimensionality of the single view vectors.
  feats:  dictionary with video_ids as keys and lists of extracted features
          (as lists of (np.array of features, label) pairs) as values
  """
  nfeats = {}
  for vid, vl in feats.items():
    if len(vl) == 3:
      keys = list(set(vl[0].keys()).intersection(vl[1].keys()).intersection(vl[2].keys()))
      nfeats[vid] = {k: (
                          np.hstack((vl[0][k][0], vl[1][k][0], vl[2][k][0])), # combined features
                          vl[0][k][1] # label
                        ) for k in keys}
    elif len(vl) == 1:
      nfeats[vid] = vl[0]
    else:
      print(f"Error: incorrect number of views for video id {vid}. Only extracted {len(vl)} view features.")
      exit()
  return nfeats


def combine(feats):
  """
  Combine the list of extracted features for all the videos in a single dataset
  feats:  dictionary with video_ids as keys and extracted features
          (as lists of (np.array of features, label) pairs) as values
  """
  # combine the features and label + vid + fid separate
  fs = []
  ls = []
  for vid in sorted(feats.keys()):
    v = feats[vid]
    for fid in sorted(v.keys()):
      f, l = v[fid]
      fs.append(f)
      ls.append((l, vid, fid))
  return np.vstack(fs), np.vstack(ls)


def process_dataset(dataset, view, device, model, data,
  opath=None, skip=1, img_size=640, stride=32, half=False,
  face=False
):
  """
  Extract features from a set of videos in a given dataset
  dataset: path to dataset, e.g., ./A1
  view: which camera view for the given video, "dash", "rear", "right", or "all"
  device: device, i.e., 'cpu' or '0' or '0,1,2,3'
  model: Torch kapao coco-kp model
  data: data structure with algorithm parameters for kapao
  skip: extract features every skip frames
  img_size: the width of the image
  stride: convolution stride
  half: use half precision
  face: whether to process the face through the face_recognition package
  """

  # get info about the dataset
  video_ids, _ = get_video_ids(dataset)
  labels = parse_labels(dataset)
  users = [d for d in os.listdir(dataset)
            if isdir(join(dataset, d)) and d.lower().startswith("user")
          ]
  vpaths = [join(dataset, u, f) for u in users
              for f in os.listdir(join(dataset, u)) if f[-4:].lower() == '.mp4'
           ]
  vpaths = sorted([(p, video_ids[basename(p)], video_view(p)) for p in vpaths], key=lambda x: (x[1], x[2]))
  if view != "all":
    # filter to only videos of the chosen type
    vpaths = [p for p in vpaths if p[2] == view]

  feats = defaultdict(list)
  for vpath, vid, vv in vpaths:
    # get labels for video
    vlabels = labels[labels.video_id == vid][['ts_start', 'ts_end', 'activity_id']].values.tolist()
    feats[vid].append(extract(
      vpath, vid, device, model, data,
      view=view, labels=vlabels, skip=skip,
      img_size=img_size, stride=stride, half=half,
      face=face
    ))
  # merge features for each video id, if need be, then combine into a single dataset
  feats = merge(feats)
  fdata, flabels = combine(feats)
  # store restuls
  if opath is not None:
    if not opath.endswith(".npz") and not not opath.endswith(".npy"):
      opath = f"{opath}.npz"
    print(f"Wrote features to {opath}")
    if opath.endswith(".npz"):
      np.savez_compressed(opath, data=fdata, labels=flabels)
    else:
      np.savez(opath, data=fdata, labels=flabels)
  return fdata, flabels
