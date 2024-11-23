import glob
import logging
import math
import os
import random
import hashlib
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils.general import xywhn2xyxy, xyxy2xywhn

# Parameters
img_formats = ['bmp', 'jpg', 'jpeg', 'png']
logger = logging.getLogger(__name__)

def get_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))
    h = hashlib.md5(str(size).encode())
    h.update(''.join(paths).encode())
    return h.hexdigest()

def create_dataloader(path, imgsz, batch_size, stride, hyp=None, augment=False, cache=False,
                     rect=False, workers=8, image_weights=False, prefix=''):
    dataset = LoadImagesAndLabels(
        path,
        imgsz,
        batch_size,
        augment=augment,
        hyp=hyp,
        rect=rect,
        cache_images=cache,
        stride=stride,
        image_weights=image_weights,
        prefix=prefix
    )

    batch_size = min(batch_size, len(dataset))
    workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, workers])
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
        collate_fn=LoadImagesAndLabels.collate_fn
    )
    return dataloader, dataset

class LoadImagesAndLabels(Dataset):
    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None,
                 rect=False, cache_images=False, stride=32, image_weights=False, prefix=""):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.stride = stride
        self.path = path
        self.albumentations = None

        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = str(Path(p))
                parent = str(Path(p).parent) + os.sep
                if os.path.isfile(p):  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]
                elif os.path.isdir(p):  # folder
                    f += glob.glob(p + os.sep + '*.*')
                else:
                    raise Exception(f'{prefix}{p} does not exist')
            self.img_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in img_formats)
            assert self.img_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}')

        # Check cache
        self.label_files = img2label_paths(self.img_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix('.cache')
        try:
            cache = np.load(cache_path, allow_pickle=True).item()
            assert cache['version'] == self.__class__.__name__ and cache['hash'] == get_hash(self.label_files + self.img_files)
        except Exception:
            cache = self.cache_labels(cache_path, prefix)

        # Read cache
        [cache.pop(k) for k in ('hash', 'version')]
        labels, shapes = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes)
        self.img_files = list(cache.keys())
        self.label_files = img2label_paths(cache.keys())

        # Cache images into memory for faster training
        if cache_images:
            self.imgs, self.img_npy = [None] * len(self.img_files), [None] * len(self.img_files)
            gb = 0
            pbar = tqdm(range(len(self.img_files)), desc='Caching images')
            for i in pbar:
                self.imgs[i], self.img_npy[i] = self.load_image(i)
                gb += self.imgs[i].nbytes if self.imgs[i] is not None else 0
                pbar.desc = f'Caching images ({gb / 1E9:.1f}GB)'

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        for i, file in enumerate(tqdm(self.img_files, desc='Scanning images')):
            try:
                img = Image.open(file)
                img.verify()  # PIL verify
                shape = self.shapes[i] = exif_size(img)  # image size
                labels = []
                if os.path.isfile(self.label_files[i]):
                    labels = np.loadtxt(self.label_files[i], dtype=np.float32).reshape(-1, 5)
                x[file] = [labels, shape]
            except Exception as e:
                logger.info(f'WARNING: Ignoring corrupted image and/or label {file}: {e}')

        x['hash'] = get_hash(self.label_files + self.img_files)
        x['version'] = self.__class__.__name__
        np.save(path, x)  # save to *.npy file
        return x

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        # Load image
        img, (h0, w0), (h, w) = self.load_image(index)

        # Load labels
        labels = self.labels[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w0, h0)

        if self.augment:
            # Augment imagespace
            img, labels = random_perspective(
                img, labels,
                degrees=self.hyp['degrees'],
                translate=self.hyp['translate'],
                scale=self.hyp['scale'],
                shear=self.hyp['shear'],
                perspective=self.hyp['perspective']
            )

            # Augment colorspace
            augment_hsv(img, h_gain=self.hyp['hsv_h'], s_gain=self.hyp['hsv_s'], v_gain=self.hyp['hsv_v'])

            # Apply flip augmentation
            if random.random() < self.hyp['fliplr']:
                img = np.fliplr(img)
                if len(labels):
                    labels[:, 1] = 1 - labels[:, 1]

        nL = len(labels)  # number of labels
        if nL:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w, h)  # xyxy to xywh normalized

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        return torch.from_numpy(img), labels_out, self.img_files[index], (h0, w0)

    def load_image(self, i):
        # Loads 1 image from dataset index 'i', returns (im, original hw, resized hw)
        im = self.imgs[i]
        if im is None:  # not cached in ram
            path = self.img_files[i]
            im = cv2.imread(path)  # BGR
            assert im is not None, f'Image Not Found {path}'
            h0, w0 = im.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                interp = cv2.INTER_LINEAR if self.augment else cv2.INTER_AREA  # random.choice([cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
                im = cv2.resize(im, (int(w0 * r), int(h0 * r)), interpolation=interp)
            return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
        return self.imgs[i], self.img_hw0[i], self.img_hw[i]  # im, hw_original, hw_resized

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]

def augment_hsv(img, h_gain=0.5, s_gain=0.5, v_gain=0.5):
    r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

def random_perspective(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    if n:
        targets = targets.copy()
        targets[:, 1:] = xywhn2xyxy(targets[:, 1:], width, height)  # normalized xyxy to pixel xyxy

        # Transform corner points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

        # Create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        targets = targets.copy()
        targets[:, 1] = x.min(1)
        targets[:, 2] = y.min(1)
        targets[:, 3] = x.max(1)
        targets[:, 4] = y.max(1)

        # Convert from pixel xyxy to normalized xywh
        targets[:, 1:] = xyxy2xywhn(targets[:, 1:], width, height)

    return img, targets

def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    return s