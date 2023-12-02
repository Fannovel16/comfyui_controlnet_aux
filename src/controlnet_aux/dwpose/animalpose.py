import numpy as np
import cv2
import os
import cv2
from .dw_onnx.cv_ox_det import inference_detector as inference_onnx_yolox
from .dw_onnx.cv_ox_yolo_nas import inference_detector as inference_onnx_yolo_nas
from .dw_onnx.cv_ox_pose import inference_pose as inference_onnx_pose

from .dw_torchscript.jit_det import inference_detector as inference_jit_yolox
from .dw_torchscript.jit_pose import inference_pose as inference_jit_pose
from typing import List, Optional
from .types import PoseResult, BodyResult, Keypoint
from timeit import default_timer
from controlnet_aux.dwpose.util import guess_onnx_input_shape_dtype, get_ort_providers, get_model_type, is_model_torchscript
import json
import torch
import torch.utils.benchmark.utils.timer as torch_timer

def drawBetweenKeypoints(pose_img, keypoints, indexes, color, scaleFactor):
    ind0 = indexes[0] - 1
    ind1 = indexes[1] - 1
    
    point1 = (keypoints[ind0][0], keypoints[ind0][1])
    point2 = (keypoints[ind1][0], keypoints[ind1][1])

    thickness = int(5 // scaleFactor)


    cv2.line(pose_img, (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1])), color, thickness)


def drawBetweenKeypointsList(pose_img, keypoints, keypointPairsList, colorsList, scaleFactor):
    for ind, keypointPair in enumerate(keypointPairsList):
        drawBetweenKeypoints(pose_img, keypoints, keypointPair, colorsList[ind], scaleFactor)

def drawBetweenSetofKeypointLists(pose_img, keypoints_set, keypointPairsList, colorsList, scaleFactor):
    for keypoints in keypoints_set:
        drawBetweenKeypointsList(pose_img, keypoints, keypointPairsList, colorsList, scaleFactor)


def padImg(img, size, blackBorder=True):
    left, right, top, bottom = 0, 0, 0, 0

    # pad x
    if img.shape[1] < size[1]:
        sidePadding = int((size[1] - img.shape[1]) // 2)
        left = sidePadding
        right = sidePadding

        # pad extra on right if padding needed is an odd number
        if img.shape[1] % 2 == 1:
            right += 1

    # pad y
    if img.shape[0] < size[0]:
        topBottomPadding = int((size[0] - img.shape[0]) // 2)
        top = topBottomPadding
        bottom = topBottomPadding
        
        # pad extra on bottom if padding needed is an odd number
        if img.shape[0] % 2 == 1:
            bottom += 1

    if blackBorder:
        paddedImg = cv2.copyMakeBorder(src=img, top=top, bottom=bottom, left=left, right=right, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
    else:
        paddedImg = cv2.copyMakeBorder(src=img, top=top, bottom=bottom, left=left, right=right, borderType=cv2.BORDER_REPLICATE)

    return paddedImg

def smartCrop(img, size, center):

    width = img.shape[1]
    height = img.shape[0]
    xSize = size[1]
    ySize = size[0]
    xCenter = center[0]
    yCenter = center[1]

    if img.shape[0] > size[0] or img.shape[1] > size[1]:


        leftMargin = xCenter - xSize//2
        rightMargin = xCenter + xSize//2
        upMargin = yCenter - ySize//2
        downMargin = yCenter + ySize//2


        if(leftMargin < 0):
            xCenter += (-leftMargin)
        if(rightMargin > width):
            xCenter -= (rightMargin - width)

        if(upMargin < 0):
            yCenter -= -upMargin
        if(downMargin > height):
            yCenter -= (downMargin - height)


        img = cv2.getRectSubPix(img, size, (xCenter, yCenter))



    return img



def calculateScaleFactor(img, size, poseSpanX, poseSpanY):

    poseSpanX = max(poseSpanX, size[0])

    scaleFactorX = 1


    if poseSpanX > size[0]:
        scaleFactorX = size[0] / poseSpanX

    scaleFactorY = 1
    if poseSpanY > size[1]:
        scaleFactorY = size[1] / poseSpanY

    scaleFactor = min(scaleFactorX, scaleFactorY)


    return scaleFactor



def scaleImg(img, size, poseSpanX, poseSpanY, scaleFactor):
    scaledImg = img

    scaledImg = cv2.resize(img, (0, 0), fx=scaleFactor, fy=scaleFactor) 

    return scaledImg, scaleFactor

class AnimalPoseImage:
    def __init__(self, det_model_path: Optional[str] = None, pose_model_path: Optional[str] = None, torchscript_device="cuda"):
        self.det_filename = det_model_path and os.path.basename(det_model_path)
        self.pose_filename = pose_model_path and os.path.basename(pose_model_path)
        self.det, self.pose = None, None
        # return type: None ort cv2 torchscript
        self.det_model_type = get_model_type("AnimalPose",self.det_filename)
        self.pose_model_type = get_model_type("AnimalPose",self.pose_filename)
        # Always loads to CPU to avoid building OpenCV.
        cv2_device = 'cpu'
        cv2_backend = cv2.dnn.DNN_BACKEND_OPENCV if cv2_device == 'cpu' else cv2.dnn.DNN_BACKEND_CUDA
        # You need to manually build OpenCV through cmake to work with your GPU.
        cv2_providers = cv2.dnn.DNN_TARGET_CPU if cv2_device == 'cpu' else cv2.dnn.DNN_TARGET_CUDA
        ort_providers = get_ort_providers()

        if self.det_model_type is None:
            pass
        elif self.det_model_type == "ort":
            try:
                import onnxruntime as ort
                self.det = ort.InferenceSession(det_model_path, providers=ort_providers)
            except:
                print(f"Failed to load onnxruntime with {self.det.get_providers()}.\nPlease change EP_list in the config.yaml and restart ComfyUI")
                self.det = ort.InferenceSession(det_model_path, providers=["CPUExecutionProvider"])
        elif self.det_model_type == "cv2":
            try:
                self.det = cv2.dnn.readNetFromONNX(det_model_path)
                self.det.setPreferableBackend(cv2_backend)
                self.det.setPreferableTarget(cv2_providers)
            except:
                print("TopK operators may not work on your OpenCV, try use onnxruntime with CPUExecutionProvider")
                try:
                    import onnxruntime as ort
                    self.det = ort.InferenceSession(det_model_path, providers=["CPUExecutionProvider"])
                except:
                    print(f"Failed to load {det_model_path}, you can use other models instead")
        else:
            self.det = torch.jit.load(det_model_path)
            self.det.to(torchscript_device)

        if self.pose_model_type is None:
            pass
        elif self.pose_model_type == "ort":
            try:
                import onnxruntime as ort
                self.pose = ort.InferenceSession(pose_model_path, providers=ort_providers)
            except:
                print(f"Failed to load onnxruntime with {self.pose.get_providers()}.\nPlease change EP_list in the config.yaml and restart ComfyUI")
                self.pose = ort.InferenceSession(pose_model_path, providers=["CPUExecutionProvider"])
        elif self.pose_model_type == "cv2":
            self.pose = cv2.dnn.readNetFromONNX(pose_model_path)
            self.pose.setPreferableBackend(cv2_backend)
            self.pose.setPreferableTarget(cv2_providers)
        else:
            self.pose = torch.jit.load(pose_model_path)
            self.pose.to(torchscript_device)

        if self.pose_filename is not None:
            self.pose_input_size, _ = guess_onnx_input_shape_dtype(self.pose_filename)
    
    def __call__(self, oriImg) -> Optional[np.ndarray]:
        detect_classes = list(range(14, 23 + 1)) #https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml

        if is_model_torchscript(self.det):
            det_start = torch_timer.timer()
            det_result = inference_jit_yolox(self.det, oriImg, detect_classes=detect_classes)
            print(f"AnimalPose: Bbox {((torch_timer.timer() - det_start) * 1000):.2f}ms")
        else:
            det_start = default_timer()
            det_onnx_dtype = np.float32 if "yolox" in self.det_filename else np.uint8
            if "yolox" in self.det_filename:
                det_result = inference_onnx_yolox(self.det, oriImg, detect_classes=detect_classes, dtype=det_onnx_dtype)
            else:
                #FP16 and INT8 YOLO NAS accept uint8 input
                det_result = inference_onnx_yolo_nas(self.det, oriImg, detect_classes=detect_classes, dtype=det_onnx_dtype)
            print(f"AnimalPose: Bbox {((default_timer() - det_start) * 1000):.2f}ms")
        if (det_result is None) or (det_result.shape[0] == 0):
            openpose_dict = {
                'version': 'ap10k',
                'animals': [],
                'canvas_height': oriImg.shape[0],
                'canvas_width': oriImg.shape[1]
            }
            return np.zeros_like(oriImg), openpose_dict
        
        if is_model_torchscript(self.pose):
            pose_start = torch_timer.timer()
            keypoint_sets, scores = inference_jit_pose(self.pose, det_result, oriImg, self.pose_input_size)
            print(f"AnimalPose: Pose {((torch_timer.timer() - pose_start) * 1000):.2f}ms on {det_result.shape[0]} animals\n")
        else:
            pose_start = default_timer()
            _, pose_onnx_dtype = guess_onnx_input_shape_dtype(self.pose_filename)
            keypoint_sets, scores = inference_onnx_pose(self.pose, det_result, oriImg, self.pose_input_size, dtype=pose_onnx_dtype)
            print(f"AnimalPose: Pose {((default_timer() - pose_start) * 1000):.2f}ms on {det_result.shape[0]} animals\n")

        animal_kps_scores = []
        pose_img = np.zeros((oriImg.shape[0], oriImg.shape[1], 3), dtype = np.uint8)
        for (idx, keypoints) in enumerate(keypoint_sets):
            # don't use keypoints that go outside the frame in calculations for the center
            interorKeypoints = keypoints[((keypoints[:,0] > 0) & (keypoints[:,0] < oriImg.shape[1])) & ((keypoints[:,1] > 0) & (keypoints[:,1] < oriImg.shape[0]))]

            xVals = interorKeypoints[:,0]
            yVals = interorKeypoints[:,1]

            minX = np.amin(xVals)
            minY = np.amin(yVals)
            maxX = np.amax(xVals)
            maxY = np.amax(yVals)

            poseSpanX = maxX - minX
            poseSpanY = maxY - minY

            # find mean center

            xSum = np.sum(xVals)
            ySum = np.sum(yVals)

            xCenter = xSum // xVals.shape[0]
            yCenter = ySum // yVals.shape[0]
            center_of_keypoints = (xCenter,yCenter)

            # order of the keypoints for AP10k and a standardized list of colors for limbs
            keypointPairsList = [(1,2), (2,3), (1,3), (3,4), (4,9), (9,10), (10,11), (4,6), (6,7), (7,8), (4,5), (5,15), (15,16), (16,17), (5,12), (12,13), (13,14)]
            colorsList = [(255,255,255), (100,255,100), (150,255,255), (100,50,255), (50,150,200), (0,255,255), (0,150,0), (0,0,255), (0,0,150), (255,50,255), (255,0,255), (255,0,0), (150,0,0), (255,255,100), (0,150,0), (255,255,0), (150,150,150)] # 16 colors needed

            drawBetweenKeypointsList(pose_img, keypoints, keypointPairsList, colorsList, scaleFactor=1.0)
            score = scores[idx, ..., None]
            score[score > 1.0] = 1.0
            score[score < 0.0] = 0.0
            animal_kps_scores.append(np.concatenate((keypoints, score), axis=-1))
        
        openpose_dict = {
            'version': 'ap10k',
            'animals': [keypoints.tolist() for keypoints in animal_kps_scores],
            'canvas_height': oriImg.shape[0],
            'canvas_width': oriImg.shape[1]
        }
        return pose_img, openpose_dict