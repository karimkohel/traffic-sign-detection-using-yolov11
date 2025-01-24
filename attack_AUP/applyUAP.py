# apply_uap.py

import torch
import cv2
import numpy as np


UAP_PATH = "attack_AUP/universal_perturbation_u8.pt"
IMG_SIZE = 640
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_uap(uap_path):
    return torch.load(uap_path, map_location=DEVICE)

def preprocess_image(img_bgr, size=640):
    # Convert BGR -> RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # Resize
    img_rgb = cv2.resize(img_rgb, (size, size))
    # Convert to float tensor [0..1]
    tensor = torch.from_numpy(img_rgb).float() / 255.0
    # Rearrange to (C,H,W)
    tensor = tensor.permute(2,0,1).unsqueeze(0)
    return tensor

def postprocess_tensor(tensor):
    # Convert back to [0..255], HWC, BGR for visualization if needed
    img = tensor.squeeze(0).permute(1,2,0).cpu().numpy()
    img = np.clip(img, 0, 1) * 255
    img_bgr = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return img_bgr


def apply_uap(orgVidPath: str, attackedVideoPath: str, detections: list[tuple[int]]):
    uap = load_uap(UAP_PATH)  # shape [1,3,640,640]
    cap = cv2.VideoCapture(orgVidPath)
    frameNum = 0

    if not cap.isOpened():
        print(f"Error: Unable to open the input file {orgVidPath}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Save the frame to the new video
        if frameNum == 0:
            print("Creating the attacked video...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(attackedVideoPath, fourcc, cap.get(cv2.CAP_PROP_FPS), (frame.shape[1], frame.shape[0]))

        if frameNum == detections[0][4]:
            # Apply UAP to the frame
            x1, y1, x2, y2 = detections[0][:4]
            traffic_sign = frame[y1:y2, x1:x2]
            orig_tensor = preprocess_image(traffic_sign, IMG_SIZE).to(DEVICE)
            adv_tensor = torch.clamp(orig_tensor + uap, 0, 1)
            adv_tensor = postprocess_tensor(adv_tensor)
            adv_tensor = cv2.resize(adv_tensor, (x2 - x1, y2 - y1))
            frame[y1:y2, x1:x2] = adv_tensor
            print(frameNum)

            detections.pop(0)
        

        out.write(frame)
        
    
        frameNum += 1

    out.release()
    cap.release()