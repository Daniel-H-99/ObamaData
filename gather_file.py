import os
import shutil

src_dir = "../datasets/advp_infer"
query = "result_advp/inference.mp4"
tgt_dir = "../3d_face_gcns/results"

for vid in os.listdir(src_dir):
    if not vid.endswith(".mp4"):
        continue
    shutil.copy(os.path.join(src_dir, vid, query), os.path.join(tgt_dir,vid))
