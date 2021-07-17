import os
from pathlib import Path

### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
from collections import OrderedDict
from pathlib import Path
from skeleton_to_human.options.test_options import TestOptions
from skeleton_to_human.data.custom_dataset_data_loader import CreateDataset
from skeleton_to_human.lit_models import Pose2Vid
from skeleton_to_human.models import Pose2VidHDModel
import skeleton_to_human.util.util as util
from lib.detect_keypoints import frame_from_video
import torch
from imageio import get_writer
import numpy as np
from tqdm import tqdm
import cv2


def save_frames_from_video(video_path, save_path, folder_name="test_A"):
    i=0
    save_dir = str(Path(save_path).parent/folder_name)
    video=cv2.VideoCapture(video_path)
    os.makedirs(save_dir,exist_ok=True)
    for img in frame_from_video(video):
        #success, img = video.read()
        #if success:
        cv2.imwrite(f"{save_dir}/{i:05d}.png",img)
        i+=1
        #else:
        #    break
    return save_dir

def convert_skeleton_to_target(video_path, save_path, first_frame):
    video = cv2.VideoCapture(video_path)
    save_dir = save_frames_from_video(video_path, save_path)
    opt = TestOptions().parse(save=False)
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.no_instance=True
    opt.name = "everybody_dance_now_temporal"
    opt.dataroot = str(Path(save_path).parent)
    os.makedirs(str(Path(save_path).parent/"test_B"), exist_ok=True)
    frame_save_path=str(Path(save_path).parent/f"test_B/{sorted([f for f in os.listdir(save_dir) if not f.startswith('.')])[0]}")
    cv2.imwrite(frame_save_path,first_frame)
    fps = video.get(cv2.CAP_PROP_FPS)
    dataset = CreateDataset(opt)

    # test
    #model = create_model(opt)
    model = Pose2VidHDModel(opt)
    lit_model = Pose2Vid.load_from_checkpoint(checkpoint_path="checkpoints/model.pt",args=opt, model=model)
    lit_model.eval()
    scripted_model = lit_model.to_torchscript(method="script", file_path=None)
   

    data = dataset[0]
    if opt.use_first_frame:
        prev_frame = data['image']
        start_from = 1
        from skimage.io import imsave
        #imsave('results/ref.png', util.tensor2im(prev_frame))
        generated = [util.tensor2im(prev_frame)]
    else:
        prev_frame = torch.zeros_like(data['image'])
        start_from = 0
        generated = []

    from skimage.io import imsave
    frames_path = str(Path(save_path).parent)
    os.makedirs(frames_path, exist_ok=True)
    for i in tqdm(range(start_from, dataset.clip_length)):
        label = data['label'][i:i+1]
        inst = None if opt.no_instance else data['inst'][i:i+1]

        cur_frame = scripted_model(label, inst, torch.unsqueeze(prev_frame, dim=0))
        prev_frame = cur_frame.data[0]
        imsave(f'{frames_path}/{i:05d}.png', util.tensor2im(prev_frame))
        generated.append(util.tensor2im(prev_frame))

   

    with get_writer(save_path, fps=fps) as writer:
        for im in generated:
            writer.append_data(im)
    #writer.close()
