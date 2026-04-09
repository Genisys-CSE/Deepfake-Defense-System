"""
DeepShield — Face Detection & Processing Utilities

Handles MTCNN face detection, cropping with margin, and soft-mask
generation for seamless paste-back.
"""

import torch
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image


def detect_and_crop_face(img_pil, mtcnn, margin=15, device='cpu'):
    """
    Detect the primary face and return the cropped face tensor + bounding box.

    Returns
    -------
    face_tensor : torch.Tensor  (C, H, W) in [0, 1]  or None
    img_pil     : PIL.Image     original image
    bbox        : tuple (x1, y1, x2, y2) with margin   or None
    """
    boxes, _ = mtcnn.detect(img_pil)
    if boxes is None or len(boxes) == 0:
        print("  ⚠ No face detected.")
        return None, img_pil, None

    box = boxes[0]
    x1, y1, x2, y2 = [int(b) for b in box]
    w, h = img_pil.size
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)

    face_crop_pil = img_pil.crop((x1, y1, x2, y2))
    face_tensor = T.ToTensor()(face_crop_pil).to(device)
    return face_tensor, img_pil, (x1, y1, x2, y2)


def create_soft_mask(width, height, ellipse_scale=0.9, blur_ksize=21, blur_sigma=10):
    """
    Create a smooth elliptical mask for blending the protected face back
    into the original image.  Edges taper to zero so there is no hard seam.

    Returns
    -------
    mask_pil : PIL.Image in mode 'L'
    """
    mask = np.zeros((height, width), dtype=np.float32)
    center = (width // 2, height // 2)
    radius_x = int(width // 2 * ellipse_scale)
    radius_y = int(height // 2 * ellipse_scale)
    cv2.ellipse(mask, center, (radius_x, radius_y), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), blur_sigma)
    mask_pil = Image.fromarray((mask * 255).astype(np.uint8)).convert('L')
    return mask_pil


def paste_face_back(orig_img_pil, protected_face_tensor, bbox):
    """
    Paste the protected face back onto the original image using a soft mask.

    Parameters
    ----------
    orig_img_pil           : PIL.Image
    protected_face_tensor  : torch.Tensor (C, H, W) in [0, 1]
    bbox                   : (x1, y1, x2, y2)

    Returns
    -------
    final_img : PIL.Image
    """
    x1, y1, x2, y2 = bbox
    face_w = x2 - x1
    face_h = y2 - y1

    protected_pil = T.ToPILImage()(protected_face_tensor.cpu())
    mask_pil = create_soft_mask(face_w, face_h)

    final_img = orig_img_pil.copy()
    final_img.paste(protected_pil, (x1, y1), mask=mask_pil)
    return final_img
