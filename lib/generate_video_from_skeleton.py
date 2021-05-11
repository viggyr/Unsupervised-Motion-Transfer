import os
from pathlib import Path

### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
from collections import OrderedDict
from pathlib import Path
from options.test_options import TestOptions
from motion_transfer.data.custom_dataset_data_loader import CreateDataset
from motion_transfer.models.models import create_model
import motion_transfer.util.util as util
import torch
from imageio import get_writer
import numpy as np
from tqdm import tqdm


def save_frames_from_video(video_path, save_path):
    i=0
    save_dir = str(Path(save_path).parent/"frames")
    
    os.makedirs(save_dir)
    while video.isOpened():
        success, img = video.read()
        if success:
            cv2.imwrite(f"{save_dir}/{i:05d}.png",img)
            i+=1
        else:
            break
    return save_dir

def convert_skeleton_to_target(video_path, save_path)
    video = cv2.VideoCapture(video_path))
    save_dir = save_frames_from_video(video_path, save_path)
    opt = TestOptions().parse(save=False)
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.name = "everybody_dance_now_temporal"
    opt.dataroot = skeleton_video_path
    fps = video.get(cv2.CAP_PROP_FPS)
    dataset = CreateDataset(opt)

    # test
    model = create_model(opt)
    if opt.verbose:
        print(model)

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
    frames_path = str(Path(output_path).parent)
    os.makedirs(frames_path, exists_ok=True)
    for i in tqdm(range(start_from, dataset.clip_length)):
        label = data['label'][i:i+1]
        inst = None if opt.no_instance else data['inst'][i:i+1]

        cur_frame = model.inference(label, inst, torch.unsqueeze(prev_frame, dim=0))
        prev_frame = cur_frame.data[0]
        
        imsave(f'{frames_path}/{:05d}.png'.format(i), util.tensor2im(prev_frame))
        generated.append(util.tensor2im(prev_frame))

    result_dir = os.path.join(opt.results_dir, opt.name, opt.which_epoch)
    if not os.path.isdir(result_dir):
    os.makedirs(result_dir, exist_ok=True)

    with get_writer(, fps=fps) as writer:
    for im in generated:
        writer.append_data(im)
    writer.close()
