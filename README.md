### Contributions
* End-to-End interface for 3D avatar animation by leveraging opensource projects and collating them under a single roof. 
    * Created the entire repo with a structure according to the flow described in the report/ppt.
    * Wrote all the code under lib folder.
    * Fixed bugs in the open source repos - transmomo and everybodydancenow - Modified file motion_transfer/util/visualization function joints2image to typecast the color values. 
    
* Implemented target image support in motion_transfer. 
    * Added lines 52-85, 200-201, modified lines 178 to 185 in network.py.
    * Added lines 54-60 in transmomo_solo_dance.yaml.

## Usage

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
### For training skeleton-to-video refer https://github.com/viggyr/skeleton_to_human
### For training transmomo model, refer https://github.com/viggyr/motion_transfer