import numpy as np
from skeleton_to_human.lib.util.visualization import motion2video, hex2rgb,rgb2rgba

def generate_skeleton_video(motion_data, height, width, save_path):
    color=np.array(hex2rgb("#a50b69#b73b87#db9dc3"))
    motion2video(motion_data, 512, 512, save_path, color, bg_color=(0,0,0), save_frame=True, fps=50)


def transfer_motion_and_generate_video(source_video_sequence, target_video_sequence)
    transferred_motion = transfer_motion_to_target(config_path, source_keypoints_sequence, target_video_sequence, source_width, source_height, target_width, target_height)
    generate_skeleton_video(transferred_motion, save_dir)