#!/bin/bash
home=~/workspace/obamanet
target_folder=video
video_list=obama_addresses.txt

cd $home/$target_folder
while read line
do
    youtube-dl -f best[height=720][ext=mp4] $line
done < $video_list


