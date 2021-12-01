import os
import shutil

src_dir="/home/server25/minyeong_workspace/datasets/test_kkj/kkj04_1.mp4"
tgt_dir="/home/server25/minyeong_workspace/datasets/test_kkj/kkj04_1.mp4"

# subdirs = ["landmarks", "beta", "crop_region", "delta", "full", "gamma", "mask", "rotation", "translation"]
subdirs = ["img"]

for subdir in subdirs:
    tgt_subdir = os.path.join(tgt_dir, subdir)
    os.makedirs(tgt_subdir, exist_ok=True)
    for file in os.listdir(os.path.join(src_dir, subdir)):
        key = os.path.basename(file)
        id, ext = key.split('.')
        if int(id) > 28000:
            new_name = os.path.join("{:05d}.{}".format(int(id) - 28000, ext))
            shutil.move(os.path.join(src_dir, subdir, file), os.path.join(tgt_subdir, new_name))