import numpy as np
from motion_transfer.lib.util.visualization import motion2video, hex2rgb,rgb2rgba
from motion_transfer.motion_transfer import transfer_motion_to_target

def generate_skeleton_video(motion_data, height, width, save_path):
    color=np.array(hex2rgb("#a50b69#b73b87#db9dc3"))
    motion2video(motion_data, height, width, save_path, color, bg_color=(0,0,0), save_frame=True, fps=50)


def transfer_motion_and_generate_video(source_keypoints, save_dir):
    #transferred_motion = transfer_motion_to_target(source_keypoints, target_keypoints, 512, 512, 512, 512)
    generate_skeleton_video(source_keypoints,512, 512, save_dir)