import argparse
from lib.detect_keypoints import extract_sequence
from lib.generate_video_from_skeleton import convert_skeleton_to_target
from lib.transfer_motion import transfer_motion_and_generate_video
import numpy as np

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', type=str, required=True,
                        help='Path to the source video. Motion would be extracted from this video and applied to the target.')
    parser.add_argument('-t', '--target', type=str, required=True,
                        help='Path to the target image. Structure and view would be extracted from this and clubbed with the motion of source video.')
 
    parser.add_argument('--out_dir', type=str,default=".",
                        help="Path to output directory")
    args = parser.parse_args()

def retarget(source_path: str, target_path: str, output_video_path: str):
    #source_keypoints_sequence = extract_sequence(source_path)
    #target_keypoints_sequence = extract_sequence(target_path)
    source_keypoints = np.load("inputs/source.npy")
    target_keypoints = np.load("inputs/target.npy")
    transfer_motion_and_generate_video(source_keypoints, target_keypoints, "outputs/skeleton.mp4")
    #transferred_skeleton_video = generate_skeleton_video(transferred_keypoints_sequence)
    convert_skeleton_to_target("output/skeleton.mp4", output_video_path)

if __name__ == "__main__":
    main()