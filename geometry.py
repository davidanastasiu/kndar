"""Compute angles between connected segments in skeleton"""
from math import sqrt
import numpy as np


def iou(b1, b2):
  """
  Intersection over union (overlap) between two bounding boxes.
    b1: first bounding box (xmin, ymin, xmax, ymax)
    b2: second bounding box (xmin, ymin, xmax, ymax)
  return: iou score
  """
  xA = max(b1[0], b2[0])
  yA = max(b1[1], b2[1])
  xB = min(b1[2], b2[2])
  yB = min(b1[3], b2[3])

  ai = max(0, xB-xA + 1) * max(0, yB-yA + 1)
  a1 = (b1[2] - b1[0] + 1) * (b1[3] - b1[1] + 1)
  a2 = (b2[2] - b2[0] + 1) * (b2[3] - b2[1] + 1)

  return ai / float(a1 + a2 - ai)


def driver_bbox(bboxes, last_bboxes, width=1920):
  """
  Find the bbox for the driver - right side of the image and wider than one in the back
    bboxes: list of detected person bounding boxes
    last_bboxes: last bounding boxes
  return: index in bboxes for identified driver
  """
  if not bboxes:
    return None
  i = 0
  ib = bboxes[0]
  iw = ib[2] - ib[0]
  ic = ib[0] + iw/2
  for j in range(1, len(bboxes)):
    b = bboxes[j]
    w = b[2] - b[0]
    c = b[0] + w/2
    # filter out bboxes with centers on the left side of the screen
    if c < width * 0.4:
      continue
    if c >= ic and w >= iw * 0.75:
      i = j
      ib = b
      iw = w
      ic = c
  # if the center is still on the left side of the screen, filter it out
  if ic < width * 0.4:
    return None
  return i


def cosangle_np(a, b, c):
  """
  Compute the cosine of the angle between ab and bc, i.e., angle ABC,
  given points a, b, and c.
  """
  ab = b - a
  bc = c - b
  return ab.dot(bc) / (sqrt(ab.dot(ab)) * sqrt(bc.dot(bc)))


def cosangle(a, b, c):
  """
  Compute the cosine of the angle between ab and bc, i.e., angle ABC,
  given points a, b, and c.
  """
  ab = (b[0] - a[0], b[1] - a[1])
  bc = (c[0] - b[0], c[1] - b[1])
  return (ab[0]*bc[0] + ab[1]*bc[1]) / (sqrt(ab[0]**2 + ab[1]**2) * sqrt(bc[0]**2 + bc[1]**2))


def distance(a, b):
  """
  Compute the euclidean distance between two points a and b.
  """
  c1 = a[0] - b[0]
  c2 = a[1] - b[1]
  return sqrt(c1**2 + c2**2)

