sh preprocess.sh
for file in ../data_preprocessed/lof/*.mp4
do
    mv $file ../datasets/desk/
done
python fld.py
python landmark_normalize.py --pca_path ../datasets/tmp/PCA.pickle
python preprocess_mfcc.py