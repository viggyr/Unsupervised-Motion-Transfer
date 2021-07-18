import argparse
from lib.detect_keypoints import extract_sequence
from lib.generate_video_from_skeleton import convert_skeleton_to_target, save_frames_from_video
from lib.transfer_motion import transfer_motion_and_generate_video
import numpy as np
import cv2
import time

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', type=str, required=True,
                        help='Path to the source video. Motion would be extracted from this video and applied to the target.')
    parser.add_argument('-t', '--target', type=str, required=True,
                        help='Path to the target image. Structure and view would be extracted from this and clubbed with the motion of source video.')
 
    parser.add_argument('-o','--output_video_path', type=str,
                        help="Path to output directory")
    args = parser.parse_args()
    retarget(args.source, args.target, args.output_video_path)

def retarget(source_path: str, target_path: str, output_video_path: str):
    
    #source_keypoints = extract_sequence(source_path)
    target_keypoints,failed = extract_sequence(target_path)
    #np.save("inputs/target.npy", target_keypoints)
    #source_keypoints = np.load("inputs/source.npy")
    #target_keypoints = np.load("inputs/target.npy")
    print("Extracted keypoints.")
    #transfer_motion_and_generate_video(source_keypoints, target_keypoints, "outputs/skeleton.mp4")
    #print("Transferred motion.")
    #transferred_skeleton_video = generate_skeleton_video(transferred_keypoints_sequence)
    # _, frame = cv2.VideoCapture(target_path).read()
    # shape_dst = np.min(frame.shape[:2])
    # oh = (frame.shape[0] - shape_dst) // 2
    # ow = (frame.shape[1] - shape_dst) // 2
    # frame = frame[:shape_dst, ow:ow + shape_dst]
    # frame = cv2.resize(frame, (512, 512))
    # convert_skeleton_to_target("outputs/skeleton.mp4", output_video_path, frame)
    save_frames_from_video("outputs/skeleton.mp4", output_video_path, failed)
    save_frames_from_video(source_path, output_video_path, "train_B", failed)

if __name__ == "__main__":
    main()