import os
import shutil

dir = os.path.join('../data_preprocessed/lof/kkj04.mp4')
os.makedirs(os.path.join(dir, 'split'), exist_ok=True)
os.makedirs(os.path.join(dir, 'split', 'full'), exist_ok=True)
os.makedirs(os.path.join(dir, 'split', 'img'), exist_ok=True)
for img_path in os.listdir(os.path.join(dir,'full')):
    key = img_path[:-4]
    fid = int(key) - 1
    if fid >= 27999:
        src_path = os.path.join(dir, 'full', key + '.png')
        tgt_path = os.path.join(dir, 'split', 'full', '{:05d}.png'.format(fid - 27998))
        shutil.move(src_path, tgt_path)
        src_path = os.path.join(dir, 'img', key + '.png')
        tgt_path = os.path.join(dir, 'split', 'img', '{:05d}.png'.format(fid - 27998))
        if os.path.exists(src_path):
            shutil.move(src_path, tgt_path)
