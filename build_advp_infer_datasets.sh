
src_dir=tts
tgt_dir=test_kkj
tgt_vid_name=kkj04_1.mp4

new_root=../datasets/"${src_dir}_${tgt_dir}"
mkdir -p $new_root

cd ../datasets/$src_dir

for src_vid_name in *.mp4
do 
    new_dir=../$new_root/"${src_vid_name}_${tgt_vid_name}"
    mkdir -p $new_dir
    cp ../$src_dir/"$src_vid_name"/audio.wav "$new_dir"
    cp -r ../$tgt_dir/$tgt_vid_name/keypoints "$new_dir"
    cp -r ../$tgt_dir/$tgt_vid_name/img "$new_dir"
done

cd ../../ObamaData
python landmark_normalize.py --data_root $new_root --pca_path ../datasets/train_kkj/PCA.pickle
python preprocess_mfcc.py --data_root $new_root --noise 0.005
python build_a2l_config.py --data_root $new_root
python build_pickles.py --data_root $new_root
