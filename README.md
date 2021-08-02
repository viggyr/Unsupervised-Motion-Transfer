End to End interface for transferring the motion from source to target video.

This repo is mainly broken down into three parts

1. skeleton_to_human - Pytorch Lightning version of the Everybody Dance Now architecture.
2. motion_transfer - Transmomo architecture (Will be ported to Pytorch lightning soon.)
3. training - Use to train skeleton_to_human. (training for transmomo to be added soon after developing a pytorch lightning version of the same.)


### Requirements.txt
```
pip install -r requirements.txt
pip install detectron2 -f   https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html
```

Run on target as video or image
#### 1. Open configs/transmomo_solo_dance.yaml and set source_type as video to use a video of target or image to use an image of target
#### 2.  Move input videos to inputs folder. Note that the skeleton extraction takes long time. Hence, for the sake of the demo, the first line of animate.py is commented out and the skeleton sequences are read from the pre-extracted numpy files. If you want to run on a custom source video, then uncomment the first line and comment the second and third lines.

#### 3. Run the script
```
python animate.py -s inputs/video-clip.mp4 -t inputs/video-source-spiderman.mp4 -o outputs/spiderman-dancing.avi
```
#### 4. Check the outputs directory for the output video.

## Training
### For training skeleton-to-video refer (/skeleton-to-videe)
### For training transmomo model, refer https://github.com/viggyr/motion_transfer
