home=~/workspace/datasets/desk


cd $home

for file in *.mp4
do
    mv $file/crop $file/img
done

cd ../../
