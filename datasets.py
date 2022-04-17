"""Specifics about the AIC22 datasets"""
import os
from os.path import join, isdir, isfile, basename, dirname
import numpy as np
import pandas as pd

activities = [
  'Normal Forward Driving',
  'Drinking',
  'Phone Call(right)',
  'Phone Call(left)',
  'Eating',
  'Text (Right)',
  'Text (Left)',
  'Hair / makeup',
  'Reaching behind',
  'Adjust control panel',
  'Pick up from floor (Driver)',
  'Pick up from floor (Passenger)',
  'Talk to passenger at the right',
  'Talk to passenger at backseat',
  'Yawning',
  'Hand on head',
  'Singing with music',
  'Shaking or dancing with music',
  'N/A' # in between activities
]
activity_ids = {k:v for v,k in enumerate(activities)}

def get_video_ids(fpath):
  """
  Assign video ids for the videos in the dataset, unless video_ids.csv file exists
    fpath: root path of the dataset, e.g., the A1 folder
  """
  # if video_ids file exists, read it
  video_ids = {}
  video_lists = {}
  if isfile(join(fpath, 'video_ids.csv')):
    with open(join(fpath, 'video_ids.csv'), 'r') as fh:
      fh.readline()
      for l in fh:
        p = l.strip().split(',')
        vid = int(p[0])
        video_lists[vid] = p[1:]
        for v in p[1:]:
          video_ids[v] = vid
    return video_ids, video_lists
  # otherwise build the file
  users = [d for d in os.listdir(fpath) if isdir(join(fpath, d)) and d.lower().startswith("user_id")]
  files = [f for u in users for f in os.listdir(join(fpath, u)) if f[-4:].lower() == '.mp4']
  files.sort(key=lambda x: (x[x.index('id_')+3:], x))
  keys = {}
  for f in files:
    k = f[f.index('id_')+3:]
    if k not in keys:
      keys[k] = len(keys) + 1
    if keys[k] not in video_lists:
      video_lists[keys[k]] = []
    video_lists[keys[k]].append(f)
  # write video_ids.csv file, in the format of the organizers (not my choice...)
  with open(join(fpath, 'video_ids.csv'), "w") as fh:
    fh.write("video_id,video_files\n")
    for k in sorted(video_lists.keys()):
      fh.write(f"{k},")
      fh.write(",".join(video_lists[k]))
      fh.write("\n")
  # build videos to ids dictionary
  video_ids = {v:k for k,vs in video_lists.items() for v in vs}
  return video_ids, video_lists


def video_view(fpath):
  """
  Get view of the video based on file name or path
  """
  fpath = fpath.strip().lower()
  if 'right' in fpath:
    return 'right'
  elif 'rear' in fpath:
    return 'rear'
  elif 'dash' in fpath:
    return 'dash'
  return None


def parse_user_labels(fpath, ds, view='Dashboard', video_ids=None):
  """
  Parse the csv labels file for one user
    fpath: path to user labels csv file
    ds: dataset, e.g., A1
    view: which camera view labels should be extracted from
    video_ids: dictionary mapping video name to video id
  """
  # get video names to video_ids dictionary
  if video_ids is None:
    # build list of video_ids
    video_ids = get_video_ids(dirname(dirname(fpath)))
  # time to read the labels file
  df = pd.read_csv(fpath)
  # fix column names, which are not always the same across label files
  assert len(df.columns) == 8
  df.columns = ['User ID', 'Filename', 'Camera View', 'Activity Type', 'Start Time',
       'End Time', 'Label', 'Appearance Block']
  # names in labels files don't have NoAudio_ or extension and sometimes have different capitalization
  def normalize_name(x):
    x = x.strip().lower()
    side = video_view(x)
    if x.endswith('.mp4'):
      x = x[:-4]
    p = x.replace('noaudio_', '').split('_')
    uid = p[-2]
    view_id = p[-1]
    return '_'.join((side, uid, view_id))
  video_ids = {normalize_name(k): v
              for k,v in video_ids.items()
           }
  # process video ids in the dataframe
  video_id = []
  vid = -1
  vname = ''
  for i,r in df.iterrows():
    if not pd.isna(r['Filename']) and len(r['Filename'].strip()):
      vname = normalize_name(r['Filename'])
      if vname not in video_ids:
        print(video_ids)
      vid = video_ids[vname]
    df.loc[i, 'Filename'] = vname
    video_id.append(vid)
  df['video_id'] = video_id
  # fix Camera View, which sometimes uses different codes
  def fix_view(x):
    x = x.lower()
    if 'dash' in x:
      return 'Dashboard'
    elif 'right' in x:
      return 'Right Side Window'
    elif 'rear' in x:
      return 'Rearview'
  df['Camera View'] = df['Camera View'].apply(fix_view)
  # process activity id
  if type(df['Label'][0]) == str:
    df['activity_id'] = df['Label'].apply(lambda x: int(x.split()[1]))
  else:
    df['activity_id'] = df['Label']
  # turn time string into seconds, using 3 types of coding time (grrr!)
  def t2s(x):
    if '.' in x:
      p = x.split('.')[0].split(':')
      return 60 * int(p[0]) + int(p[1])
    p = x.split(':')
    if len(p) == 1:
      return int(p[0])
    elif len(p) == 2:
      return 60 * int(p[0]) + int(p[1])
    return 3600 * int(p[0]) + 60 * int(p[1]) + int(p[2])
  df['ts_start'] = df['Start Time'].apply(t2s)
  df['ts_end'] = df['End Time'].apply(t2s)
  # return only requested data
  df = df[df['Camera View'] == view][['video_id', 'activity_id', 'ts_start', 'ts_end']]
  df = df.replace('NA ', np.nan) # extra space caused pandas not to recognize it as NA
  df = df.dropna(axis='rows')
  df.activity_id = df.activity_id.astype(int)
  return df.sort_values(by=['video_id', 'ts_start']).reset_index(drop=True)


def check_equal_views(fpath, ds, video_ids=None):
  """
  Check if the timestamps are equivalent across the views, i.e., views are synchronized
    fpath: path to labels file for a given user
    ds: dataset, one of A1, A2, or B
    video_ids: dictionary mapping video name to video id
  """
  if video_ids is None:
    video_ids = get_video_ids(fpath)
  df1 = parse_user_labels(fpath, ds, view='Dashboard', video_ids=video_ids)
  df2 = parse_user_labels(fpath, ds, view='Right Side Window', video_ids=video_ids)
  df3 = parse_user_labels(fpath, ds, view='Rearview', video_ids=video_ids)
  if not(df1.equals(df2) and df2.equals(df3)):
    if not df1.equals(df2):
      print(f'Dashboard view not equal to Right Side Window view for {fpath[:-4]}')
      print('Dashboard', df1.to_string(index=False))
      print('Right Side Window', df2.to_string(index=False))
    if not df2.equals(df3):
      print(f'Right Side Window view not equal to Rearview view for {fpath[:-4]}')
      print('Right Side Window', df2.to_string(index=False))
      print('Rearview', df3.to_string(index=False))
  return df1.equals(df2) and df2.equals(df3)


def parse_labels(fpath, check=False):
  """
  Parse csv files containing class labels and create data frame
    fpath: path to root of dataset, e.g., A1 folder
  """
  ds = basename(fpath)
  if ds not in ['A1', 'A2', 'B']:
    print("Invalid dataset directory. Should be A1, A2, or B.")
    exit()
  lpath = join('labels', f'{ds}.txt')
  if isfile(lpath):
    return pd.read_csv(lpath, sep=' ',
      names=['video_id', 'activity_id', 'ts_start', 'ts_end'])
  video_ids = get_video_ids(fpath)
  users = [d for d in os.listdir(fpath) if isdir(join(fpath, d)) and d.lower().startswith("user_id")]
  csv_files = []
  for u in users:
    for f in os.listdir(join(fpath, u)):
      if basename(f).lower().startswith("user") and f.endswith(".csv"):
        csv_files.append(join(fpath, u, f))
  if not csv_files:
    return pd.DataFrame(columns = ['video_id', 'activity_id', 'ts_start', 'ts_end'])
  if check:
    # Note: User_id_49381.csv has a discrepancy between the dash view and the right side window view
    # Changed the record ,,Dashboard,Distracted,0:04:32,0:04:55,4,None
    # to ,,Dashboard,Distracted,0:04:32,0:04:54,4,None
    # to match the right side window and rearview views
    for f in csv_files:
      try:
        check_equal_views(f, ds, video_ids)
      except Exception as e:
        print(f, e)
  df = pd.concat([parse_user_labels(f, ds, video_ids=video_ids) for f in csv_files])
  df = df.sort_values(by=['video_id', 'ts_start']).reset_index(drop=True)
  assert not df.empty
  if not isdir('labels'):
    os.mkdir('labels')
  df.to_csv(lpath, sep=' ', index=False, header=False)
  return df


def get_label(labels, ts):
  """
  Get activity_id for a given time in the video
      labels: list of activity tuples ('ts_start', 'ts_end', 'activity_id'), sorted by ts_start
      ts: time, in seconds, for the given frame
  Construct labels from the dataset labels df as follows, where video_id is the ID of the video:
  labels = df[df.video_id == video_id][['ts_start', 'ts_end', 'activity_id']].values.tolist()
  """
  ts = int(ts)
  for ts_start, ts_end, activity_id in labels:
    if ts >= ts_start and ts <= ts_end:
      return int(activity_id)
    if ts_start > ts:
      return 18
  return 18
