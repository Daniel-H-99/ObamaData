import os
import argparse
import numpy as np
import pickle as pkl
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='../datasets/train', help='Path to the train data file.')
parser.add_argument('--bfm', action='store_true', help='Path to the train data file.')
opts = parser.parse_args()

if opts.bfm:
    with open(os.path.join(opts.data_root, 'config.txt'), 'w') as f:
        videos = os.listdir(opts.data_root)
        d = {}
        for video in videos:
            if not '.mp4' in video:
                continue
            print("processing {}...".format(video))
            audios = sorted(os.listdir(os.path.join(opts.data_root, video, 'audio')))
            no_bfm = False
            try:
                bfms = np.loadtxt(os.path.join(opts.data_root, video, 'bfmcoeff.txt'), delimiter=',', dtype=np.float32)[:, 80:144]
            except:
                print("no bfmcoeff.txt")
                no_bfm = True
            bfms_dict = {}
            for audio in audios:
                if audio == '.ipynb_checkpoints':
                    continue
                index = audio.replace('.pickle', '')
                
                if not no_bfm:
                    bfms_dict[index] = bfms[int(index)]
                f.write(video + '/' + index + '\n')
            d[video] = bfms_dict
    bfmpklname = os.path.join(opts.data_root, 'bfms.pickle')
    with open(bfmpklname, "wb") as outputFile:
        pkl.dump(d, outputFile)
        
else:
    with open(os.path.join(opts.data_root, 'config.txt'), 'w') as f:
        videos = os.listdir(opts.data_root)
        for video in videos:
            if not '.mp4' in video:
                continue
            print("processing {}...".format(video))
            kps = sorted(os.listdir(os.path.join(opts.data_root, video, 'keypoints')))
            for kp in kps:
                if kp == '.ipynb_checkpoints':
                    continue
                index = kp.replace('.txt', '')
                audio = os.path.join(opts.data_root, video, 'audio', '{:05d}'.format(int(index) - 1) + '.pickle')
                if os.path.exists(audio):
                    f.write(video + '/' + kp.replace('.txt', '') + '\n')
