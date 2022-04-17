import os
import json
import pickle
import numpy as np
import pandas as pd
from pprint import pprint
from argparse import ArgumentParser


def merge(preds, mgap=30, sgap=90, minlen=300, maxp=0.15):
  """
  Merge consecutive label regions and eliminate outliers.

  preds: Predictions np.array in the format aid, vid, fid
  mgap: Maximum in-segment gap
  sgap: Minimum between-segment gap
  minlen: Minimum length of a segment
  maxp: Maximum expected percentage or an aid
  """
  preds = preds.copy()
  # 1) eliminate classes that are too abundant -- likely classifier mistakes
  t = np.sum(preds[:,0]!=18)
  elim = [i for i in range(18) if np.sum(preds[:,0]==i)/t > maxp]
  for c in elim:
    preds[preds[:,0]==c,0] = 18  # change to N/A
  # 2) fill in any N/A gaps between same-predictions if gap length < mgap
  lvi = 0
  lvj = 0
  gc = 0
  lv = None
  lvid = preds[0,1]

  for i in range(preds.shape[0]):
    v, vid = preds[i,0:2]
    if vid != lvid:
      lvid = vid
      # start a new search
      if lv is not None and lvj > lvi:
        preds[lvi:lvj,0] = lv
      lv = None
    if v == 18:
      if lv is not None:
        gc += 1 # count latest gap only if in active match
        if gc > mgap:
          # gap is too large, reset merge search
          if lvj > lvi:
            preds[lvi:lvj,0] = lv
          lv = None
      continue
    gc = 0 # end of gap
    if lv is None or v != lv:
      # start a new search
      if lv is not None and lvj > lvi:
        preds[lvi:lvj,0] = lv
      lv = v
      lvi = lvj = i
      continue
    if v == lv:
      lvj = i
      continue

  return preds


def merge_segments(pr, minlen=100):
  """
  Merge segments from label per key frame format to start and end time format.
  Eliminate segments that are too short.

  minlen: minimum length of a segment that should be kept
  pr: np.array containing predictions in rows as vid, fid, aid
  return: list of results as (vid, aid, ts_start, ts_end) tuples
  """
  p = pr[pr[:,0]!=18]
  results = []

  skip = 0
  for i in range(p.shape[0]):
    if i < skip:
      continue
    v, vid, f = p[i]
    e = s = i
    # look for end of sequence
    for j in range(i+1, p.shape[0]):
      if p[j,0] != v or p[j,1] != vid or p[j,2] > f+j:
        e = j-1
        skip = j
        if e > s and p[e,2]-p[s,2] >= minlen:
          results.append((vid, v, int(p[s,2]/30), int(p[e,2]/30)))
        break
  return results


def write_preds(pr=None, results=None, ofname=None, minlen=100):
  """
  Write predictions to a file.

  pr: np.array containing predictions in rows as vid, fid, aid
  ofname: path for results file to write results into
  minlen: minimum length of a segment that should be kept
  return: list of results as (vid, aid, ts_start, ts_end) tuples
  """
  assert results or pr
  if results is None:
    results = merge_segments(pr, minlen=minlen)

  if ofname is not None:
    with open(ofname, "w") as fh:
      for r in results:
        fh.write(' '.join([f"{l}" for l in r]))
        fh.write("\n")
    print("Results written to:", ofname)

  return results


def read_labels(fpath):
  """
  Read labels from file
  """
  labels = []
  if not fpath:
    raise ValueError("Missing labels file.")
  with open(fpath, "r") as fh:
    for l in fh:
      p = l.strip().split()
      if len(p) != 4:
        raise ValueError(f"Labels file {fpath} has incorrect number of values. Expected 'vid aid ts_start ts_end'. Got {l.strip()}. ")
      labels.append(tuple((int(i) for i in p)))

  return labels


def eval_ndar(test, pred):
  """
  Evaluate Multi-Class Product Counting & Recognition for Automated Retail Checkout submission.

  test : Labeled data for the test set, as list of (vid, aid, ts_start, ts_end) tuples
  pred : Predictions for the same videos, as (vid, aid, ts_start, ts_end) tuples
  test and pred are expected to be sorted by vid, ts_start, ts_end
  return (f1, precision, recall)
  """
  if test is None:
    return None

  found = np.zeros(len(test), dtype=bool)
  tp = 0

  for i, row in enumerate(pred):
    for j, qrow in enumerate(test):
      if (
        not found[j] and
        qrow[0] == row[0] and
        qrow[1] == row[1] and
        qrow[2]-1 <= row[2] and
        qrow[2]+1 >= row[2] and
        qrow[3]-1 <= row[3] and
        qrow[3]+1 >= row[3]
      ):
        tp += 1
        found[j] = True
        break

  precision = tp / len(pred)   # tp / # retrieved items (len(pred))
  recall = tp / len(test)    # tp / # relevant items (len(test))
  f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

  return f1, precision, recall



def tune(ds, model, dstype="train", lpath='labels/A1.txt',
     mgaps='30', sgaps='90', minlens='300', maxps='0.15'
  ):
  """
  Tune merge meta-parameters.

  ds: Path to dataset file (pkl)
  model: Model that should be applied to the dataset
  dstype: train, val, or test
  lpath: ground truth labels for ds
  mgaps: one or more comma-separated mgap values to test
  sgaps: one or more comma-separated sgap values to test
  minlens: one or more comma-separated minlen values to test
  maxps: one or more comma-separated maxp values to test
  return best parameters choice plus merge results
  """
  # read dataset
  file = np.load(ds, allow_pickle=True)
  X_test = file['data']
  y_test = file['labels'][:,0]
  # get predicitions using activity classification model
  preds = model.predict(X_test)
  y_vid_fid = file['labels'][:,1:3].copy()
  preds = np.hstack((preds.reshape(-1, 1), y_vid_fid))
  # get ground-truth labels and make sure they are sorted
  labels = sorted(read_labels(lpath), key=lambda x: (x[0], x[2], x[3]))

  bestf1 = -1
  best = None

  mgaps = [int(i) for i in mgaps.split(",")]
  sgaps = [int(i) for i in sgaps.split(",")]
  minlens = [int(i) for i in minlens.split(",")]
  maxps = [float(i) for i in maxps.split(",")]

  ofname = os.path.splitext(args.model)[0] + f"-result-{dstype}.txt"

  for mgap in mgaps:
    for sgap in sgaps:
      for minlen in minlens:
        for maxp in maxps:
          pr = merge(preds, mgap=mgap, sgap=sgap, minlen=minlen, maxp=maxp)
          rr = sorted(merge_segments(pr, minlen=minlen), key=lambda x: (x[0], x[2], x[3]))
          f1, precision, recall = eval_ndar(labels, rr)
          print(f"mgap={mgap}, sgap={sgap}, minlen={minlen}, maxp={maxp}: f1 {f1}")
          if f1 > bestf1:
            bestf1 = f1
            best = (f1, precision, recall, rr, pr, mgap, sgap, minlen, maxp)
  # write out the best result
  f1, precision, recall, rr, pr, mgap, sgap, minlen, maxp = best
  write_preds(results=rr, ofname=ofname)
  print(f"best {dstype} options: mgap={mgap}, sgap={sgap}, minlen={minlen}, maxp={maxp}")
  print(f"Result:\n    f1 {f1}\n    precision {precision}\n    recall {recall}")

  return best


def test(ds, model, dstype="val", lpath=None, mgap=30, sgap=90, minlen=300, maxp=0.15):
  """
  Inference using model on dataset. Optionally evaluate performance if ground truth labels
  are provided.

  ds: Path to dataset file (pkl)
  model: Model that should be applied to the dataset
  dstype: train, val, or test
  lpath: ground truth labels for ds

  """
  # read dataset
  file = np.load(ds, allow_pickle=True)
  X_test = file['data']
  y_test = file['labels'][:,0]
  # get predicitions using activity classification model
  preds = model.predict(X_test)
  y_vid_fid = file['labels'][:,1:3].copy()
  preds = np.hstack((preds.reshape(-1, 1), y_vid_fid))

  ofname = os.path.splitext(args.model)[0] + f"-result-{dstype}.txt"

  pr = merge(preds, mgap=mgap, sgap=sgap, minlen=minlen, maxp=maxp)
  rr = sorted(merge_segments(pr, minlen=minlen), key=lambda x: (x[0], x[2], x[3]))
  write_preds(results=rr, ofname=ofname)

  if lpath is not None:
    # get ground-truth labels and make sure they are sorted
    labels = sorted(read_labels(lpath), key=lambda x: (x[0], x[2], x[3]))
    f1, precision, recall = eval_ndar(labels, rr)
    print(f"test result, mgap={mgap}, sgap={sgap}, minlen={minlen}, maxp={maxp}:")
    print(f"    f1 {f1}\n    precision {precision}\n    recall {recall}")

  return rr

if __name__ == '__main__':
  parser = ArgumentParser()

  # video options
  parser.add_argument('-tr', '--train', default='', help='training dataset')
  parser.add_argument('-te', '--test', default='', help='test dataset')
  parser.add_argument('-m', '--model', default='', help='model')
  parser.add_argument('-t', '--tune', action='store_true', help='Whether to tune parameters')
  parser.add_argument('--labels', type=str, default='labels/A1.txt', help="Training set labels, for tuning.")
  parser.add_argument('--test-labels', type=str, default=None, help="Test/Validation set labels.")
  parser.add_argument('-g', '--mgap', type=str, default='30', help="maximum merge gap length")
  parser.add_argument('-s', '--sgap', type=str, default='90', help="minimum space gap length")
  parser.add_argument('-l', '--minlen', type=str, default='90', help="minimum run length")
  parser.add_argument('-p', '--maxp', type=str, default='0.15', help="maximum class occurance probability")
  args = parser.parse_args()

  print("parameters:")
  pprint(vars(args), indent=1)

  trainds = args.train
  testds = args.test
  if not trainds:
    trainds = "-".join(os.path.basename(args.model).split("-")[:3]) + ".npz"
    if "-f-" in args.model:
      trainds = "-".join(os.path.basename(args.model).split("-")[:4]) + ".npz"
  if not testds:
    testds = trainds.replace('A1', 'A2')
  print("training set:", trainds)
  print("testing set:", testds)

  with open(args.model,'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']


  if args.tune:
    print("Tuning meta-parameters using training set...")
    _, _, mgap, sgap, minlen, maxp = tune(trainds, model, dstype="train", lpath=args.labels,
      mgaps=args.mgap, sgaps=args.sgap, minlens=args.minlen, maxps=args.maxp)
    print()
    print("Performing inference on test set using best meta-parameters...")
    test(testds, model, dstype="test", lpath=args.test_labels,
      mgap=mgap, sgap=sgap, minlen=minlen, maxp=maxp)
    exit()

  mgaps = args.mgap.split(',')
  sgaps = args.sgap.split(',')
  minlens = args.minlen.split(',')
  maxps = args.maxp.split(',')
  if len(mgaps) > 1 or len(sgaps) > 1 or len(minlens) > 1 or len(maxps) > 1:
    print("Are you trying to tune? Use --tune. Otherwise, include only one value for mgaps, sgaps, minlens, and maxps.")
    exit()
  mgap = int(mgaps[0])
  sgap = int(sgaps[0])
  minlen = int(minlens[0])
  maxp = float(maxps[0])
  print("Performing inference on test set using best meta-parameters...")
  test(testds, model, dstype="test", lpath=args.test_labels,
    mgap=mgap, sgap=sgap, minlen=minlen, maxp=maxp)
