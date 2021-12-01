sh preprocess.sh
for file in ../data_preprocessed/lof/*.mp4
do
    mv $file ../datasets/desk/
done
# CUDA_VISIBLE_DEVICES=3 python fld.py #--data_root ../datasets/train
# python landmark_normalize.py
# python preprocess_mfcc.py
# python build_a2l_config.py --data_root ../datasets/train_k
# python build_pickles.py