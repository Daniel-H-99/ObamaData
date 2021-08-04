# sh preprocess.sh
# for file in ../data_preprocessed/lof/*.mp4
# do
#     mv $file ../datasets/desk/
# done
python fld.py #--data_root ../datasets/train
python landmark_normalize.py #--pca_path ../datasets/tmp/PCA.pickle #--data_root ../datasets/train
python preprocess_mfcc.py #--data_root ../datasets/train
python build_a2l_config.py #--data_root ../datasets/train
python build_pickles.py #--data_root ../datasets/train