import cv2
import numpy as np
from skimage import exposure
from skimage.filters import unsharp_mask

def process_frame(frame, width, height):
    frame = frame.astype(np.float32) / 255.0  # 0-1 범위로 정규화
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = exposure.adjust_gamma(frame, gamma=0.5, gain=1)
    frame = exposure.equalize_adapthist(frame, clip_limit=0.03)
    frame = unsharp_mask(frame, radius=1, amount=1)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = (frame * 255).astype(np.uint8)  # 다시 0-255 범위로 변환
    return frame

def resize_frame(frame, width, height):
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
