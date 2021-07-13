import os
from tqdm import tqdm
import numpy as np
import pickle as pkl
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='../datasets/desk', help='Path to the train data file.')
parser.add_argument('--use_landmarks', action='store_true')
opts = parser.parse_args()

d = {}
lib = {}
root = opts.data_root     # change to target directory
tgt_folder = 'keypoints' if not opts.use_landmarks else 'landmarks'
for vid in os.listdir(root):
  if not vid.endswith('.mp4'):
    continue
  print('working on {}'.format(vid))
  lms = []
  keys = []
  fid = 0
  for lm_txt in tqdm(sorted(os.listdir(os.path.join(root, vid, tgt_folder)))):
    if not len(lm_txt) == 9:
      continue
    key = lm_txt[:-4]
    lm = np.loadtxt(os.path.join(root, vid, tgt_folder, lm_txt), delimiter=',', dtype=np.float32)
    if not lm.shape == (68, 2):
      continue
    lms.append(lm)
    keys.append(key)
  d[vid] = np.stack(lms, axis=0)
  lib[vid] = keys
with open(os.path.join(root, 'landmarks.pickle'), 'wb') as f:
  pkl.dump(d, f)
with open(os.path.join(root, 'library.pickle'), 'wb') as f:
  pkl.dump(lib, f)