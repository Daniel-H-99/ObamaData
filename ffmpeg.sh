/snap/bin/ffmpeg -hide_banner -y -loglevel warning \
    -thread_queue_size 8192 -i /home/server25/minyeong_workspace/datasets/test_kkj/kkj04_1.mp4/lip/%05d.png \
    -ss 00:18:40 -i /home/server25/minyeong_workspace/datasets/train_kkj/kkj04.mp4/audio.wav \
    -vcodec libx264 -preset slower -profile:v high -crf 18 -pix_fmt yuv420p /home/server25/minyeong_workspace/datasets/test_kkj/kkj04_1.mp4/lip.mp4