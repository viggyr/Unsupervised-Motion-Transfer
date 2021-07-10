from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.detection_utils import read_image
import detectron2.data.transforms as T
import torch
import cv2
from detectron2.data import MetadataCatalog
import numpy as np
import os

cfg = get_cfg()
cfg.merge_from_file("configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
cfg.merge_from_list(["MODEL.WEIGHTS", "detectron2://COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x/137849621/model_final_a6e10b.pkl"])
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.8

def frame_from_video(video):
    i=0
    while video.isOpened():
        success, img = video.read()

        if success:
            shape_dst = np.min(img.shape[:2])
            oh = (img.shape[0] - shape_dst) // 2
            ow = (img.shape[1] - shape_dst) // 2
            img = img[:shape_dst, ow:ow + shape_dst]
            img = cv2.resize(img, (512, 512))
            yield img
        else:
            break

def convert_coco_to_openpose_cords(coco_keypoints):
    indices = [0, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 1, 2, 3, 4]
    body_25_points=coco_keypoints[:,indices,:2]
    ls_x, ls_y = body_25_points[0, 4, :]
    rs_x, rs_y = + body_25_points[0, 1, :]
    mid_shoulder_x, mid_shoulder_y = (ls_x + rs_x) / 2, (ls_y + rs_y) / 2

    lh_x, lh_y = body_25_points[0, 10, :]
    rh_x, rh_y = + body_25_points[0, 7, :]
    mid_hip_x, mid_hip_y = (lh_x + rh_x) / 2, (lh_y + rh_y) / 2

    body_25_points=np.insert(body_25_points, 1, np.array([mid_shoulder_x, mid_shoulder_y]) ,axis=1)
    body_25_points=np.insert(body_25_points, 8, np.array([mid_hip_x, mid_hip_y]) ,axis=1)
    return body_25_points[:,:15,:]

def predict(original_image):
    model = build_model(cfg)
    model.eval()
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    aug = T.ResizeShortestEdge(
                [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
            )
    with torch.no_grad():  
        height, width = original_image.shape[:2]
        image = aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = {"image": image, "height": height, "width": width}
        predictions = model([inputs])[0]
        return predictions["instances"].pred_keypoints
    
def generate_keypoints(video):
    frame_gen=frame_from_video(video)
    for frame in frame_gen:
        yield predict(frame)
    
def find_video_keypoints(video_path: str):
    video=cv2.VideoCapture(video_path)
    video_keypoints=generate_keypoints(video)
    final_keypoints=None
    for i,frame_keypoints in enumerate(video_keypoints):
        try:
            frame_keypoints_transformed=convert_coco_to_openpose_cords(frame_keypoints.to("cpu").numpy()[-1:,:,:]).transpose(1,2,0)
            print(f"Frame {i} extracted")
        except:
            print(f"Failed to extact keypoints for frame {i}.")
            continue
        if final_keypoints is not None:
            final_keypoints=np.concatenate((final_keypoints, frame_keypoints_transformed), axis=2)
        else:
            final_keypoints=frame_keypoints_transformed
    return final_keypoints

def extract_sequence(source_video_path):
    source_keypoints=find_video_keypoints(source_video_path)
    #target_keypoints=find_video_keypoints(target_video_path)
    return source_keypoints