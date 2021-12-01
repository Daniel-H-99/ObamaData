import os
from tqdm import tqdm
import numpy as np
import pickle as pkl
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='../datasets/desk', help='Path to the train data file.')
parser.add_argument('--use_landmarks', action='store_true')
parser.add_argument('--aux_dir', type=str, default=None, help='directory where auxiliary landmarks are stored')
parser.add_argument('--landmarks_dir', type=str, default=None, help='directory where landmarks are stored')

opts = parser.parse_args()



root = opts.data_root     # change to target directory
tgt_folder = 'keypoints' if not opts.use_landmarks else 'landmarks'
if opts.landmarks_dir is not None:
  tgt_folder = opts.landmarks_dir

use_aux_dir = opts.aux_dir is not None

d = {}
lib = {}
if use_aux_dir:
  d_aux = {}

for vid in os.listdir(root):
  if not vid.endswith('.mp4'):
    continue
  print('working on {}'.format(vid))
  lms = []
  keys = []
  if use_aux_dir:
    aux_lms = []
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
    if use_aux_dir:
      aux_lm = np.loadtxt(os.path.join(root, vid, opts.aux_dir, lm_txt), delimiter=',', dtype=np.float32)
      aux_lms.append(aux_lm)
  d[vid] = np.stack(lms, axis=0)
  if use_aux_dir:
    d_aux[vid] = np.stack(aux_lms, axis=0)
  lib[vid] = keys


# with open('../datasets/train/rawKp.pickle', 'rb') as f:
#   d = pkl.load(f)

# for vid in os.listdir('../datasets/train'):
#   if not vid.endswith('.mp4'):
#     continue
#   print('working on {}'.format(vid))
#   mouths_aligned = []
#   fid = 0
#   for lm_txt in tqdm(sorted(os.listdir(os.path.join('../datasets/train', vid, gt_folder)))):
#     if not len(lm_txt) == 9:
#       continue
#     key = lm_txt[:-4]
#     kpNorm = d[vid][key][4]
#     # lm = np.loadtxt(os.path.join('../datasets/train', vid, gt_folder, lm_txt), delimiter=',', dtype=np.float32)
#     if not lm.shape == (20, 2):
#       continue
#     mouths_aligned.append(kpNorm)
#   mouths[vid] = np.array(mouths_aligned)

with open(os.path.join(root, 'landmarks.pickle'), 'wb') as f:
  pkl.dump(d, f)
with open(os.path.join(root, 'library.pickle'), 'wb') as f:
  pkl.dump(lib, f)
if use_aux_dir:
  with open(os.path.join(root, 'aux_landmarks.pickle'), 'wb') as f:
    pkl.dump(d_aux, f)  
# with open(os.path.join(root, 'mouths.pickle'), 'wb') as f:
#   pkl.dump(mouths, f)