src_dir=data_raw/lof
tgt_dir=data_preprocessed/lof
home=~/minyeong_workspace

# start_time="00:00:10"
# end_time="00:02:00"

cd $home/$src_dir

for file in *.mp4
do
    # mkdir -p $home/$tgt_dir/$file
    # mkdir -p $home/$tgt_dir/$file/full
    # mkdir -p $home/$tgt_dir/$file/crop
    # ffmpeg -hide_banner -y -i $file -ss $start_time -t $end_time -r 25 $home/$tgt_dir/$file/full/%05d.png
    # ffmpeg -hide_banner -y -i $file -r 25 $home/$tgt_dir/$file/full/%05d.png
    CUDA_VISIBLE_DEVICES=1 python $home/ObamaData/util/crop_portrait.py --data_dir $home/$tgt_dir/$file --crop_level 2 --vertical_adjust 0.85 --dest_size 256
    # ffmpeg -loglevel panic -y -ss $start_time -i $file -t $end_time -strict -2 $home/$tgt_dir/$file/audio.wav
    # ffmpeg -loglevel panic -y -i $file -strict -2 $home/$tgt_dir/$file/audio.wav
    # rm -rf $home/$tgt_dir/$file/full
    # mv $home/$tgt_dir/$file/crop $home/$tgt_dir/$file/img
done

cd $home
