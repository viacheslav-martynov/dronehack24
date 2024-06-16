# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics Results, Boxes and Masks classes for handling inference results.

Usage: See https://docs.ultralytics.com/modes/predict/
"""
import os
import platform

from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
import math
import re
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
import math
import warnings
from pathlib import Path

from typing import Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from PIL import __version__ as pil_version


class Colors:
    """
    Ultralytics default color palette https://ultralytics.com/.

    This class provides methods to work with the Ultralytics color palette, including converting hex color codes to
    RGB values.

    Attributes:
        palette (list of tuple): List of RGB color values.
        n (int): The number of colors in the palette.
        pose_palette (np.ndarray): A specific color palette array with dtype np.uint8.
    """

    def __init__(self):
        """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
        hexs = (
            "FF3838",
            "FF9D97",
            "FF701F",
            "FFB21D",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "2C99A8",
            "00C2FF",
            "344593",
            "6473FF",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)
        self.pose_palette = np.array(
            [
                [255, 128, 0],
                [255, 153, 51],
                [255, 178, 102],
                [230, 230, 0],
                [255, 153, 255],
                [153, 204, 255],
                [255, 102, 255],
                [255, 51, 255],
                [102, 178, 255],
                [51, 153, 255],
                [255, 153, 153],
                [255, 102, 102],
                [255, 51, 51],
                [153, 255, 153],
                [102, 255, 102],
                [51, 255, 51],
                [0, 255, 0],
                [0, 0, 255],
                [255, 0, 0],
                [255, 255, 255],
            ],
            dtype=np.uint8,
        )

    def __call__(self, i, bgr=False):
        """Converts hex color codes to RGB values."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """Converts hex color codes to RGB values (i.e. default PIL order)."""
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'

def is_ascii(s) -> bool:
    """
    Check if a string is composed of only ASCII characters.

    Args:
        s (str): String to be checked.

    Returns:
        (bool): True if the string is composed only of ASCII characters, False otherwise.
    """
    # Convert list, tuple, None, etc. to string
    s = str(s)

    # Check if the string is composed of only ASCII characters
    return all(ord(c) < 128 for c in s)

MACOS, LINUX, WINDOWS = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])

def is_dir_writeable(dir_path: Union[str, Path]) -> bool:
    """
    Check if a directory is writeable.

    Args:
        dir_path (str | Path): The path to the directory.

    Returns:
        (bool): True if the directory is writeable, False otherwise.
    """
    return os.access(str(dir_path), os.W_OK)


def get_user_config_dir(sub_dir="Ultralytics"):
    """
    Return the appropriate config directory based on the environment operating system.

    Args:
        sub_dir (str): The name of the subdirectory to create.

    Returns:
        (Path): The path to the user config directory.
    """
    if WINDOWS:
        path = Path.home() / "AppData" / "Roaming" / sub_dir
    elif MACOS:  # macOS
        path = Path.home() / "Library" / "Application Support" / sub_dir
    elif LINUX:
        path = Path.home() / ".config" / sub_dir
    else:
        raise ValueError(f"Unsupported operating system:")

    # GCP and AWS lambda fix, only /tmp is writeable
    if not is_dir_writeable(path.parent):
        
        path = Path("/tmp") / sub_dir if is_dir_writeable("/tmp") else Path().cwd() / sub_dir

    # Create the subdirectory if it does not exist
    path.mkdir(parents=True, exist_ok=True)

    return path

USER_CONFIG_DIR = Path(os.getenv("YOLO_CONFIG_DIR") or get_user_config_dir())

def check_font(font="Arial.ttf"):
    """
    Find font locally or download to user's configuration directory if it does not already exist.

    Args:
        font (str): Path or name of font.

    Returns:
        file (Path): Resolved font file path.
    """
    from matplotlib import font_manager

    # Check USER_CONFIG_DIR
    name = Path(font).name
    file = USER_CONFIG_DIR / name
    if file.exists():
        return file

    # Check system fonts
    matches = [s for s in font_manager.findSystemFonts() if font in s]
    if any(matches):
        return matches[0]

class Annotator:
    """
    Ultralytics Annotator for train/val mosaics and JPGs and predictions annotations.

    Attributes:
        im (Image.Image or numpy array): The image to annotate.
        pil (bool): Whether to use PIL or cv2 for drawing annotations.
        font (ImageFont.truetype or ImageFont.load_default): Font used for text annotations.
        lw (float): Line width for drawing.
        skeleton (List[List[int]]): Skeleton structure for keypoints.
        limb_color (List[int]): Color palette for limbs.
        kpt_color (List[int]): Color palette for keypoints.
    """

    def __init__(self, im, line_width=None, font_size=None, font="Arial.ttf", pil=False, example="abc"):
        """Initialize the Annotator class with image and line width along with color palette for keypoints and limbs."""
        non_ascii = not is_ascii(example)  # non-latin labels, i.e. asian, arabic, cyrillic
        input_is_pil = isinstance(im, Image.Image)
        self.pil = pil or non_ascii or input_is_pil
        self.lw = line_width or max(round(sum(im.size if input_is_pil else im.shape) / 2 * 0.003), 2)
        if self.pil:  # use PIL
            self.im = im if input_is_pil else Image.fromarray(im)
            self.draw = ImageDraw.Draw(self.im)
            try:
                font = check_font("Arial.Unicode.ttf" if non_ascii else font)
                size = font_size or max(round(sum(self.im.size) / 2 * 0.035), 12)
                self.font = ImageFont.truetype(str(font), size)
            except Exception:
                self.font = ImageFont.load_default()
            # Deprecation fix for w, h = getsize(string) -> _, _, w, h = getbox(string)
            self.font.getsize = lambda x: self.font.getbbox(x)[2:4]  # text width, height
        else:  # use cv2
            assert im.data.contiguous, "Image not contiguous. Apply np.ascontiguousarray(im) to Annotator input images."
            self.im = im if im.flags.writeable else im.copy()
            self.tf = max(self.lw - 1, 1)  # font thickness
            self.sf = self.lw / 3  # font scale
        # Pose
        self.skeleton = [
            [16, 14],
            [14, 12],
            [17, 15],
            [15, 13],
            [12, 13],
            [6, 12],
            [7, 13],
            [6, 7],
            [6, 8],
            [7, 9],
            [8, 10],
            [9, 11],
            [2, 3],
            [1, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6],
            [5, 7],
        ]

        self.limb_color = colors.pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
        self.kpt_color = colors.pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]

    def box_label(self, box, label="", color=(128, 128, 128), txt_color=(255, 255, 255), rotated=False):
        """Add one xyxy box to image with label."""
        if isinstance(box, torch.Tensor):
            box = box.tolist()
        if self.pil or not is_ascii(label):
            if rotated:
                p1 = box[0]
                # NOTE: PIL-version polygon needs tuple type.
                self.draw.polygon([tuple(b) for b in box], width=self.lw, outline=color)
            else:
                p1 = (box[0], box[1])
                self.draw.rectangle(box, width=self.lw, outline=color)  # box
            if label:
                w, h = self.font.getsize(label)  # text width, height
                outside = p1[1] - h >= 0  # label fits outside box
                self.draw.rectangle(
                    (p1[0], p1[1] - h if outside else p1[1], p1[0] + w + 1, p1[1] + 1 if outside else p1[1] + h + 1),
                    fill=color,
                )
                # self.draw.text((box[0], box[1]), label, fill=txt_color, font=self.font, anchor='ls')  # for PIL>8.0
                self.draw.text((p1[0], p1[1] - h if outside else p1[1]), label, fill=txt_color, font=self.font)
        else:  # cv2
            if rotated:
                p1 = [int(b) for b in box[0]]
                # NOTE: cv2-version polylines needs np.asarray type.
                cv2.polylines(self.im, [np.asarray(box, dtype=int)], True, color, self.lw)
            else:
                p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
            if label:
                w, h = cv2.getTextSize(label, 0, fontScale=self.sf, thickness=self.tf)[0]  # text width, height
                outside = p1[1] - h >= 3
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(
                    self.im,
                    label,
                    (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0,
                    self.sf,
                    txt_color,
                    thickness=self.tf,
                    lineType=cv2.LINE_AA,
                )

    def masks(self, masks, colors, im_gpu, alpha=0.5, retina_masks=False):
        """
        Plot masks on image.

        Args:
            masks (tensor): Predicted masks on cuda, shape: [n, h, w]
            colors (List[List[Int]]): Colors for predicted masks, [[r, g, b] * n]
            im_gpu (tensor): Image is in cuda, shape: [3, h, w], range: [0, 1]
            alpha (float): Mask transparency: 0.0 fully transparent, 1.0 opaque
            retina_masks (bool): Whether to use high resolution masks or not. Defaults to False.
        """
        if self.pil:
            # Convert to numpy first
            self.im = np.asarray(self.im).copy()
        if len(masks) == 0:
            self.im[:] = im_gpu.permute(1, 2, 0).contiguous().cpu().numpy() * 255
        if im_gpu.device != masks.device:
            im_gpu = im_gpu.to(masks.device)
        colors = torch.tensor(colors, device=masks.device, dtype=torch.float32) / 255.0  # shape(n,3)
        colors = colors[:, None, None]  # shape(n,1,1,3)
        masks = masks.unsqueeze(3)  # shape(n,h,w,1)
        masks_color = masks * (colors * alpha)  # shape(n,h,w,3)

        inv_alpha_masks = (1 - masks * alpha).cumprod(0)  # shape(n,h,w,1)
        mcs = masks_color.max(dim=0).values  # shape(n,h,w,3)

        im_gpu = im_gpu.flip(dims=[0])  # flip channel
        im_gpu = im_gpu.permute(1, 2, 0).contiguous()  # shape(h,w,3)
        im_gpu = im_gpu * inv_alpha_masks[-1] + mcs
        im_mask = im_gpu * 255
        im_mask_np = im_mask.byte().cpu().numpy()
        self.im[:] = im_mask_np if retina_masks else scale_image(im_mask_np, self.im.shape)
        if self.pil:
            # Convert im back to PIL and update draw
            self.fromarray(self.im)

    def kpts(self, kpts, shape=(640, 640), radius=5, kpt_line=True, conf_thres=0.25):
        """
        Plot keypoints on the image.

        Args:
            kpts (tensor): Predicted keypoints with shape [17, 3]. Each keypoint has (x, y, confidence).
            shape (tuple): Image shape as a tuple (h, w), where h is the height and w is the width.
            radius (int, optional): Radius of the drawn keypoints. Default is 5.
            kpt_line (bool, optional): If True, the function will draw lines connecting keypoints
                                       for human pose. Default is True.

        Note:
            `kpt_line=True` currently only supports human pose plotting.
        """
        if self.pil:
            # Convert to numpy first
            self.im = np.asarray(self.im).copy()
        nkpt, ndim = kpts.shape
        is_pose = nkpt == 17 and ndim in {2, 3}
        kpt_line &= is_pose  # `kpt_line=True` for now only supports human pose plotting
        for i, k in enumerate(kpts):
            color_k = [int(x) for x in self.kpt_color[i]] if is_pose else colors(i)
            x_coord, y_coord = k[0], k[1]
            if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
                if len(k) == 3:
                    conf = k[2]
                    if conf < conf_thres:
                        continue
                cv2.circle(self.im, (int(x_coord), int(y_coord)), radius, color_k, -1, lineType=cv2.LINE_AA)

        if kpt_line:
            ndim = kpts.shape[-1]
            for i, sk in enumerate(self.skeleton):
                pos1 = (int(kpts[(sk[0] - 1), 0]), int(kpts[(sk[0] - 1), 1]))
                pos2 = (int(kpts[(sk[1] - 1), 0]), int(kpts[(sk[1] - 1), 1]))
                if ndim == 3:
                    conf1 = kpts[(sk[0] - 1), 2]
                    conf2 = kpts[(sk[1] - 1), 2]
                    if conf1 < conf_thres or conf2 < conf_thres:
                        continue
                if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
                    continue
                if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
                    continue
                cv2.line(self.im, pos1, pos2, [int(x) for x in self.limb_color[i]], thickness=2, lineType=cv2.LINE_AA)
        if self.pil:
            # Convert im back to PIL and update draw
            self.fromarray(self.im)

    def rectangle(self, xy, fill=None, outline=None, width=1):
        """Add rectangle to image (PIL-only)."""
        self.draw.rectangle(xy, fill, outline, width)

    def text(self, xy, text, txt_color=(255, 255, 255), anchor="top", box_style=False):
        """Adds text to an image using PIL or cv2."""
        if anchor == "bottom":  # start y from font bottom
            w, h = self.font.getsize(text)  # text width, height
            xy[1] += 1 - h
        if self.pil:
            if box_style:
                w, h = self.font.getsize(text)
                self.draw.rectangle((xy[0], xy[1], xy[0] + w + 1, xy[1] + h + 1), fill=txt_color)
                # Using `txt_color` for background and draw fg with white color
                txt_color = (255, 255, 255)
            if "\n" in text:
                lines = text.split("\n")
                _, h = self.font.getsize(text)
                for line in lines:
                    self.draw.text(xy, line, fill=txt_color, font=self.font)
                    xy[1] += h
            else:
                self.draw.text(xy, text, fill=txt_color, font=self.font)
        else:
            if box_style:
                w, h = cv2.getTextSize(text, 0, fontScale=self.sf, thickness=self.tf)[0]  # text width, height
                outside = xy[1] - h >= 3
                p2 = xy[0] + w, xy[1] - h - 3 if outside else xy[1] + h + 3
                cv2.rectangle(self.im, xy, p2, txt_color, -1, cv2.LINE_AA)  # filled
                # Using `txt_color` for background and draw fg with white color
                txt_color = (255, 255, 255)
            cv2.putText(self.im, text, xy, 0, self.sf, txt_color, thickness=self.tf, lineType=cv2.LINE_AA)

    def fromarray(self, im):
        """Update self.im from a numpy array."""
        self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
        self.draw = ImageDraw.Draw(self.im)

    def result(self):
        """Return annotated image as array."""
        return np.asarray(self.im)

    def show(self, title=None):
        """Show the annotated image."""
        Image.fromarray(np.asarray(self.im)[..., ::-1]).show(title)

    def save(self, filename="image.jpg"):
        """Save the annotated image to 'filename'."""
        cv2.imwrite(filename, np.asarray(self.im))

    def get_bbox_dimension(self, bbox=None):
        """
        Calculate the area of a bounding box.

        Args:
            bbox (tuple): Bounding box coordinates in the format (x_min, y_min, x_max, y_max).

        Returns:
            angle (degree): Degree value of angle between three points
        """
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        return width, height, width * height

    def draw_region(self, reg_pts=None, color=(0, 255, 0), thickness=5):
        """
        Draw region line.

        Args:
            reg_pts (list): Region Points (for line 2 points, for region 4 points)
            color (tuple): Region Color value
            thickness (int): Region area thickness value
        """
        cv2.polylines(self.im, [np.array(reg_pts, dtype=np.int32)], isClosed=True, color=color, thickness=thickness)

    def draw_centroid_and_tracks(self, track, color=(255, 0, 255), track_thickness=2):
        """
        Draw centroid point and track trails.

        Args:
            track (list): object tracking points for trails display
            color (tuple): tracks line color
            track_thickness (int): track line thickness value
        """
        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(self.im, [points], isClosed=False, color=color, thickness=track_thickness)
        cv2.circle(self.im, (int(track[-1][0]), int(track[-1][1])), track_thickness * 2, color, -1)

    def queue_counts_display(self, label, points=None, region_color=(255, 255, 255), txt_color=(0, 0, 0)):
        """
        Displays queue counts on an image centered at the points with customizable font size and colors.

        Args:
            label (str): queue counts label
            points (tuple): region points for center point calculation to display text
            region_color (RGB): queue region color
            txt_color (RGB): text display color
        """

        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        center_x = sum(x_values) // len(points)
        center_y = sum(y_values) // len(points)

        text_size = cv2.getTextSize(label, 0, fontScale=self.sf, thickness=self.tf)[0]
        text_width = text_size[0]
        text_height = text_size[1]

        rect_width = text_width + 20
        rect_height = text_height + 20
        rect_top_left = (center_x - rect_width // 2, center_y - rect_height // 2)
        rect_bottom_right = (center_x + rect_width // 2, center_y + rect_height // 2)
        cv2.rectangle(self.im, rect_top_left, rect_bottom_right, region_color, -1)

        text_x = center_x - text_width // 2
        text_y = center_y + text_height // 2

        # Draw text
        cv2.putText(
            self.im,
            label,
            (text_x, text_y),
            0,
            fontScale=self.sf,
            color=txt_color,
            thickness=self.tf,
            lineType=cv2.LINE_AA,
        )

    def display_objects_labels(self, im0, text, txt_color, bg_color, x_center, y_center, margin):
        """
        Display the bounding boxes labels in parking management app.

        Args:
            im0 (ndarray): inference image
            text (str): object/class name
            txt_color (bgr color): display color for text foreground
            bg_color (bgr color): display color for text background
            x_center (float): x position center point for bounding box
            y_center (float): y position center point for bounding box
            margin (int): gap between text and rectangle for better display
        """

        text_size = cv2.getTextSize(text, 0, fontScale=self.sf, thickness=self.tf)[0]
        text_x = x_center - text_size[0] // 2
        text_y = y_center + text_size[1] // 2

        rect_x1 = text_x - margin
        rect_y1 = text_y - text_size[1] - margin
        rect_x2 = text_x + text_size[0] + margin
        rect_y2 = text_y + margin
        cv2.rectangle(im0, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)
        cv2.putText(im0, text, (text_x, text_y), 0, self.sf, txt_color, self.tf, lineType=cv2.LINE_AA)

    def display_analytics(self, im0, text, txt_color, bg_color, margin):
        """
        Display the overall statistics for parking lots
        Args:
            im0 (ndarray): inference image
            text (dict): labels dictionary
            txt_color (bgr color): display color for text foreground
            bg_color (bgr color): display color for text background
            margin (int): gap between text and rectangle for better display
        """

        horizontal_gap = int(im0.shape[1] * 0.02)
        vertical_gap = int(im0.shape[0] * 0.01)
        text_y_offset = 0
        for label, value in text.items():
            txt = f"{label}: {value}"
            text_size = cv2.getTextSize(txt, 0, self.sf, self.tf)[0]
            if text_size[0] < 5 or text_size[1] < 5:
                text_size = (5, 5)
            text_x = im0.shape[1] - text_size[0] - margin * 2 - horizontal_gap
            text_y = text_y_offset + text_size[1] + margin * 2 + vertical_gap
            rect_x1 = text_x - margin * 2
            rect_y1 = text_y - text_size[1] - margin * 2
            rect_x2 = text_x + text_size[0] + margin * 2
            rect_y2 = text_y + margin * 2
            cv2.rectangle(im0, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)
            cv2.putText(im0, txt, (text_x, text_y), 0, self.sf, txt_color, self.tf, lineType=cv2.LINE_AA)
            text_y_offset = rect_y2

    @staticmethod
    def estimate_pose_angle(a, b, c):
        """
        Calculate the pose angle for object.

        Args:
            a (float) : The value of pose point a
            b (float): The value of pose point b
            c (float): The value o pose point c

        Returns:
            angle (degree): Degree value of angle between three points
        """
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def draw_specific_points(self, keypoints, indices=None, shape=(640, 640), radius=2, conf_thres=0.25):
        """
        Draw specific keypoints for gym steps counting.

        Args:
            keypoints (list): list of keypoints data to be plotted
            indices (list): keypoints ids list to be plotted
            shape (tuple): imgsz for model inference
            radius (int): Keypoint radius value
        """
        if indices is None:
            indices = [2, 5, 7]
        for i, k in enumerate(keypoints):
            if i in indices:
                x_coord, y_coord = k[0], k[1]
                if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
                    if len(k) == 3:
                        conf = k[2]
                        if conf < conf_thres:
                            continue
                    cv2.circle(self.im, (int(x_coord), int(y_coord)), radius, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        return self.im

    def plot_angle_and_count_and_stage(
        self, angle_text, count_text, stage_text, center_kpt, color=(104, 31, 17), txt_color=(255, 255, 255)
    ):
        """
        Plot the pose angle, count value and step stage.

        Args:
            angle_text (str): angle value for workout monitoring
            count_text (str): counts value for workout monitoring
            stage_text (str): stage decision for workout monitoring
            center_kpt (int): centroid pose index for workout monitoring
            color (tuple): text background color for workout monitoring
            txt_color (tuple): text foreground color for workout monitoring
        """

        angle_text, count_text, stage_text = (f" {angle_text:.2f}", f"Steps : {count_text}", f" {stage_text}")

        # Draw angle
        (angle_text_width, angle_text_height), _ = cv2.getTextSize(angle_text, 0, self.sf, self.tf)
        angle_text_position = (int(center_kpt[0]), int(center_kpt[1]))
        angle_background_position = (angle_text_position[0], angle_text_position[1] - angle_text_height - 5)
        angle_background_size = (angle_text_width + 2 * 5, angle_text_height + 2 * 5 + (self.tf * 2))
        cv2.rectangle(
            self.im,
            angle_background_position,
            (
                angle_background_position[0] + angle_background_size[0],
                angle_background_position[1] + angle_background_size[1],
            ),
            color,
            -1,
        )
        cv2.putText(self.im, angle_text, angle_text_position, 0, self.sf, txt_color, self.tf)

        # Draw Counts
        (count_text_width, count_text_height), _ = cv2.getTextSize(count_text, 0, self.sf, self.tf)
        count_text_position = (angle_text_position[0], angle_text_position[1] + angle_text_height + 20)
        count_background_position = (
            angle_background_position[0],
            angle_background_position[1] + angle_background_size[1] + 5,
        )
        count_background_size = (count_text_width + 10, count_text_height + 10 + self.tf)

        cv2.rectangle(
            self.im,
            count_background_position,
            (
                count_background_position[0] + count_background_size[0],
                count_background_position[1] + count_background_size[1],
            ),
            color,
            -1,
        )
        cv2.putText(self.im, count_text, count_text_position, 0, self.sf, txt_color, self.tf)

        # Draw Stage
        (stage_text_width, stage_text_height), _ = cv2.getTextSize(stage_text, 0, self.sf, self.tf)
        stage_text_position = (int(center_kpt[0]), int(center_kpt[1]) + angle_text_height + count_text_height + 40)
        stage_background_position = (stage_text_position[0], stage_text_position[1] - stage_text_height - 5)
        stage_background_size = (stage_text_width + 10, stage_text_height + 10)

        cv2.rectangle(
            self.im,
            stage_background_position,
            (
                stage_background_position[0] + stage_background_size[0],
                stage_background_position[1] + stage_background_size[1],
            ),
            color,
            -1,
        )
        cv2.putText(self.im, stage_text, stage_text_position, 0, self.sf, txt_color, self.tf)

    def seg_bbox(self, mask, mask_color=(255, 0, 255), det_label=None, track_label=None):
        """
        Function for drawing segmented object in bounding box shape.

        Args:
            mask (list): masks data list for instance segmentation area plotting
            mask_color (tuple): mask foreground color
            det_label (str): Detection label text
            track_label (str): Tracking label text
        """
        cv2.polylines(self.im, [np.int32([mask])], isClosed=True, color=mask_color, thickness=2)

        label = f"Track ID: {track_label}" if track_label else det_label
        text_size, _ = cv2.getTextSize(label, 0, 0.7, 1)

        cv2.rectangle(
            self.im,
            (int(mask[0][0]) - text_size[0] // 2 - 10, int(mask[0][1]) - text_size[1] - 10),
            (int(mask[0][0]) + text_size[0] // 2 + 5, int(mask[0][1] + 5)),
            mask_color,
            -1,
        )

        cv2.putText(
            self.im, label, (int(mask[0][0]) - text_size[0] // 2, int(mask[0][1]) - 5), 0, 0.7, (255, 255, 255), 2
        )

    def plot_distance_and_line(self, distance_m, distance_mm, centroids, line_color, centroid_color):
        """
        Plot the distance and line on frame.

        Args:
            distance_m (float): Distance between two bbox centroids in meters.
            distance_mm (float): Distance between two bbox centroids in millimeters.
            centroids (list): Bounding box centroids data.
            line_color (RGB): Distance line color.
            centroid_color (RGB): Bounding box centroid color.
        """

        (text_width_m, text_height_m), _ = cv2.getTextSize(f"Distance M: {distance_m:.2f}m", 0, self.sf, self.tf)
        cv2.rectangle(self.im, (15, 25), (15 + text_width_m + 10, 25 + text_height_m + 20), line_color, -1)
        cv2.putText(
            self.im,
            f"Distance M: {distance_m:.2f}m",
            (20, 50),
            0,
            self.sf,
            centroid_color,
            self.tf,
            cv2.LINE_AA,
        )

        (text_width_mm, text_height_mm), _ = cv2.getTextSize(f"Distance MM: {distance_mm:.2f}mm", 0, self.sf, self.tf)
        cv2.rectangle(self.im, (15, 75), (15 + text_width_mm + 10, 75 + text_height_mm + 20), line_color, -1)
        cv2.putText(
            self.im,
            f"Distance MM: {distance_mm:.2f}mm",
            (20, 100),
            0,
            self.sf,
            centroid_color,
            self.tf,
            cv2.LINE_AA,
        )

        cv2.line(self.im, centroids[0], centroids[1], line_color, 3)
        cv2.circle(self.im, centroids[0], 6, centroid_color, -1)
        cv2.circle(self.im, centroids[1], 6, centroid_color, -1)

    def visioneye(self, box, center_point, color=(235, 219, 11), pin_color=(255, 0, 255)):
        """
        Function for pinpoint human-vision eye mapping and plotting.

        Args:
            box (list): Bounding box coordinates
            center_point (tuple): center point for vision eye view
            color (tuple): object centroid and line color value
            pin_color (tuple): visioneye point color value
        """
        center_bbox = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
        cv2.circle(self.im, center_point, self.tf * 2, pin_color, -1)
        cv2.circle(self.im, center_bbox, self.tf * 2, color, -1)
        cv2.line(self.im, center_point, center_bbox, color, self.tf)

class TryExcept(contextlib.ContextDecorator):
    """
    Ultralytics TryExcept class. Use as @TryExcept() decorator or 'with TryExcept():' context manager.

    Examples:
        As a decorator:
        >>> @TryExcept(msg="Error occurred in func", verbose=True)
        >>> def func():
        >>>    # Function logic here
        >>>     pass

        As a context manager:
        >>> with TryExcept(msg="Error occurred in block", verbose=True):
        >>>     # Code block here
        >>>     pass
    """

    def __init__(self, msg="", verbose=True):
        """Initialize TryExcept class with optional message and verbosity settings."""
        self.msg = msg
        self.verbose = verbose

    def __enter__(self):
        """Executes when entering TryExcept context, initializes instance."""
        pass

    def __exit__(self, exc_type, value, traceback):
        """Defines behavior when exiting a 'with' block, prints error message if necessary."""
        if self.verbose and value:
            print((f"{self.msg}{': ' if self.msg else ''}{value}"))
        return True

def plt_settings(rcparams=None, backend="Agg"):
    """
    Decorator to temporarily set rc parameters and the backend for a plotting function.

    Example:
        decorator: @plt_settings({"font.size": 12})
        context manager: with plt_settings({"font.size": 12}):

    Args:
        rcparams (dict): Dictionary of rc parameters to set.
        backend (str, optional): Name of the backend to use. Defaults to 'Agg'.

    Returns:
        (Callable): Decorated function with temporarily set rc parameters and backend. This decorator can be
            applied to any function that needs to have specific matplotlib rc parameters and backend for its execution.
    """

    if rcparams is None:
        rcparams = {"font.size": 11}

    def decorator(func):
        """Decorator to apply temporary rc parameters and backend to a function."""

        def wrapper(*args, **kwargs):
            """Sets rc parameters and backend, calls the original function, and restores the settings."""
            original_backend = plt.get_backend()
            if backend.lower() != original_backend.lower():
                plt.close("all")  # auto-close()ing of figures upon backend switching is deprecated since 3.8
                plt.switch_backend(backend)

            with plt.rc_context(rcparams):
                result = func(*args, **kwargs)

            if backend != original_backend:
                plt.close("all")
                plt.switch_backend(original_backend)
            return result

        return wrapper

    return decorator

@TryExcept()  # known issue https://github.com/ultralytics/yolov5/issues/5395
@plt_settings()
def plot_labels(boxes, cls, names=(), save_dir=Path(""), on_plot=None):
    """Plot training labels including class histograms and box statistics."""
    import pandas  # scope for faster 'import ultralytics'
    import seaborn  # scope for faster 'import ultralytics'

    # Filter matplotlib>=3.7.2 warning and Seaborn use_inf and is_categorical FutureWarnings
    warnings.filterwarnings("ignore", category=UserWarning, message="The figure layout has changed to tight")
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Plot dataset labels
    nc = int(cls.max() + 1)  # number of classes
    boxes = boxes[:1000000]  # limit to 1M boxes
    x = pandas.DataFrame(boxes, columns=["x", "y", "width", "height"])

    # Seaborn correlogram
    seaborn.pairplot(x, corner=True, diag_kind="auto", kind="hist", diag_kws=dict(bins=50), plot_kws=dict(pmax=0.9))
    plt.savefig(save_dir / "labels_correlogram.jpg", dpi=200)
    plt.close()

    # Matplotlib labels
    ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)[1].ravel()
    y = ax[0].hist(cls, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    for i in range(nc):
        y[2].patches[i].set_color([x / 255 for x in colors(i)])
    ax[0].set_ylabel("instances")
    if 0 < len(names) < 30:
        ax[0].set_xticks(range(len(names)))
        ax[0].set_xticklabels(list(names.values()), rotation=90, fontsize=10)
    else:
        ax[0].set_xlabel("classes")
    seaborn.histplot(x, x="x", y="y", ax=ax[2], bins=50, pmax=0.9)
    seaborn.histplot(x, x="width", y="height", ax=ax[3], bins=50, pmax=0.9)

    # Rectangles
    boxes[:, 0:2] = 0.5  # center
    boxes = xywh2xyxy(boxes) * 1000
    img = Image.fromarray(np.ones((1000, 1000, 3), dtype=np.uint8) * 255)
    for cls, box in zip(cls[:500], boxes[:500]):
        ImageDraw.Draw(img).rectangle(box, width=1, outline=colors(cls))  # plot
    ax[1].imshow(img)
    ax[1].axis("off")

    for a in [0, 1, 2, 3]:
        for s in ["top", "right", "left", "bottom"]:
            ax[a].spines[s].set_visible(False)

    fname = save_dir / "labels.jpg"
    plt.savefig(fname, dpi=200)
    plt.close()
    if on_plot:
        on_plot(fname)

def increment_path(path, exist_ok=False, sep="", mkdir=False):
    """
    Increments a file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.

    If the path exists and exist_ok is not set to True, the path will be incremented by appending a number and sep to
    the end of the path. If the path is a file, the file extension will be preserved. If the path is a directory, the
    number will be appended directly to the end of the path. If mkdir is set to True, the path will be created as a
    directory if it does not already exist.

    Args:
        path (str, pathlib.Path): Path to increment.
        exist_ok (bool, optional): If True, the path will not be incremented and returned as-is. Defaults to False.
        sep (str, optional): Separator to use between the path and the incrementation number. Defaults to ''.
        mkdir (bool, optional): Create a directory if it does not exist. Defaults to False.

    Returns:
        (pathlib.Path): Incremented path.
    """
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")

        # Method 1
        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"  # increment path
            if not os.path.exists(p):
                break
        path = Path(p)

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path

def save_one_box(xyxy, im, file=Path("im.jpg"), gain=1.02, pad=10, square=False, BGR=False, save=True):
    """
    Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop.

    This function takes a bounding box and an image, and then saves a cropped portion of the image according
    to the bounding box. Optionally, the crop can be squared, and the function allows for gain and padding
    adjustments to the bounding box.

    Args:
        xyxy (torch.Tensor or list): A tensor or list representing the bounding box in xyxy format.
        im (numpy.ndarray): The input image.
        file (Path, optional): The path where the cropped image will be saved. Defaults to 'im.jpg'.
        gain (float, optional): A multiplicative factor to increase the size of the bounding box. Defaults to 1.02.
        pad (int, optional): The number of pixels to add to the width and height of the bounding box. Defaults to 10.
        square (bool, optional): If True, the bounding box will be transformed into a square. Defaults to False.
        BGR (bool, optional): If True, the image will be saved in BGR format, otherwise in RGB. Defaults to False.
        save (bool, optional): If True, the cropped image will be saved to disk. Defaults to True.

    Returns:
        (numpy.ndarray): The cropped image.

    Example:
        ```python
        from ultralytics.utils.plotting import save_one_box

        xyxy = [50, 50, 150, 150]
        im = cv2.imread('image.jpg')
        cropped_im = save_one_box(xyxy, im, file='cropped.jpg', square=True)
        ```
    """

    if not isinstance(xyxy, torch.Tensor):  # may be list
        xyxy = torch.stack(xyxy)
    b = xyxy2xywh(xyxy.view(-1, 4))  # boxes
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = xywh2xyxy(b).long()
    xyxy = clip_boxes(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]) : int(xyxy[0, 3]), int(xyxy[0, 0]) : int(xyxy[0, 2]), :: (1 if BGR else -1)]
    if save:
        file.parent.mkdir(parents=True, exist_ok=True)  # make directory
        f = str(increment_path(file).with_suffix(".jpg"))
        # cv2.imwrite(f, crop)  # save BGR, https://github.com/ultralytics/yolov5/issues/7007 chroma subsampling issue
        Image.fromarray(crop[..., ::-1]).save(f, quality=95, subsampling=0)  # save RGB
    return crop

import threading

def threaded(func):
    """
    Multi-threads a target function by default and returns the thread or function result.

    Use as @threaded decorator. The function runs in a separate thread unless 'threaded=False' is passed.
    """

    def wrapper(*args, **kwargs):
        """Multi-threads a given function based on 'threaded' kwarg and returns the thread or function result."""
        if kwargs.pop("threaded", True):  # run in thread
            thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
            thread.start()
            return thread
        else:
            return func(*args, **kwargs)

    return wrapper


@threaded
def plot_images(
    images,
    batch_idx,
    cls,
    bboxes=np.zeros(0, dtype=np.float32),
    confs=None,
    masks=np.zeros(0, dtype=np.uint8),
    kpts=np.zeros((0, 51), dtype=np.float32),
    paths=None,
    fname="images.jpg",
    names=None,
    on_plot=None,
    max_subplots=16,
    save=True,
    conf_thres=0.25,
):
    """Plot image grid with labels."""
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(cls, torch.Tensor):
        cls = cls.cpu().numpy()
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.cpu().numpy()
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy().astype(int)
    if isinstance(kpts, torch.Tensor):
        kpts = kpts.cpu().numpy()
    if isinstance(batch_idx, torch.Tensor):
        batch_idx = batch_idx.cpu().numpy()

    max_size = 1920  # max image size
    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs**0.5)  # number of subplots (square)
    if np.max(images[0]) <= 1:
        images *= 255  # de-normalise (optional)

    # Build Image
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
    for i in range(bs):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        mosaic[y : y + h, x : x + w, :] = images[i].transpose(1, 2, 0)

    # Resize (optional)
    scale = max_size / ns / max(h, w)
    if scale < 1:
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)
        mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))

    # Annotate
    fs = int((h + w) * ns * 0.01)  # font size
    annotator = Annotator(mosaic, line_width=round(fs / 10), font_size=fs, pil=True, example=names)
    for i in range(bs):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)  # borders
        if paths:
            annotator.text((x + 5, y + 5), text=Path(paths[i]).name[:40], txt_color=(220, 220, 220))  # filenames
        if len(cls) > 0:
            idx = batch_idx == i
            classes = cls[idx].astype("int")
            labels = confs is None

            if len(bboxes):
                boxes = bboxes[idx]
                conf = confs[idx] if confs is not None else None  # check for confidence presence (label vs pred)
                if len(boxes):
                    if boxes[:, :4].max() <= 1.1:  # if normalized with tolerance 0.1
                        boxes[..., [0, 2]] *= w  # scale to pixels
                        boxes[..., [1, 3]] *= h
                    elif scale < 1:  # absolute coords need scale if image scales
                        boxes[..., :4] *= scale
                boxes[..., 0] += x
                boxes[..., 1] += y
                is_obb = boxes.shape[-1] == 5  # xywhr
                boxes = xywhr2xyxyxyxy(boxes) if is_obb else xywh2xyxy(boxes)
                for j, box in enumerate(boxes.astype(np.int64).tolist()):
                    c = classes[j]
                    color = colors(c)
                    c = names.get(c, c) if names else c
                    if labels or conf[j] > conf_thres:
                        label = f"{c}" if labels else f"{c} {conf[j]:.1f}"
                        annotator.box_label(box, label, color=color, rotated=is_obb)

            elif len(classes):
                for c in classes:
                    color = colors(c)
                    c = names.get(c, c) if names else c
                    annotator.text((x, y), f"{c}", txt_color=color, box_style=True)

            # Plot keypoints
            if len(kpts):
                kpts_ = kpts[idx].copy()
                if len(kpts_):
                    if kpts_[..., 0].max() <= 1.01 or kpts_[..., 1].max() <= 1.01:  # if normalized with tolerance .01
                        kpts_[..., 0] *= w  # scale to pixels
                        kpts_[..., 1] *= h
                    elif scale < 1:  # absolute coords need scale if image scales
                        kpts_ *= scale
                kpts_[..., 0] += x
                kpts_[..., 1] += y
                for j in range(len(kpts_)):
                    if labels or conf[j] > conf_thres:
                        annotator.kpts(kpts_[j], conf_thres=conf_thres)

            # Plot masks
            if len(masks):
                if idx.shape[0] == masks.shape[0]:  # overlap_masks=False
                    image_masks = masks[idx]
                else:  # overlap_masks=True
                    image_masks = masks[[i]]  # (1, 640, 640)
                    nl = idx.sum()
                    index = np.arange(nl).reshape((nl, 1, 1)) + 1
                    image_masks = np.repeat(image_masks, nl, axis=0)
                    image_masks = np.where(image_masks == index, 1.0, 0.0)

                im = np.asarray(annotator.im).copy()
                for j in range(len(image_masks)):
                    if labels or conf[j] > conf_thres:
                        color = colors(classes[j])
                        mh, mw = image_masks[j].shape
                        if mh != h or mw != w:
                            mask = image_masks[j].astype(np.uint8)
                            mask = cv2.resize(mask, (w, h))
                            mask = mask.astype(bool)
                        else:
                            mask = image_masks[j].astype(bool)
                        with contextlib.suppress(Exception):
                            im[y : y + h, x : x + w, :][mask] = (
                                im[y : y + h, x : x + w, :][mask] * 0.4 + np.array(color) * 0.6
                            )
                annotator.fromarray(im)
    if not save:
        return np.asarray(annotator.im)
    annotator.im.save(fname)  # save
    if on_plot:
        on_plot(fname)


@plt_settings()
def plot_results(file="path/to/results.csv", dir="", segment=False, pose=False, classify=False, on_plot=None):
    """
    Plot training results from a results CSV file. The function supports various types of data including segmentation,
    pose estimation, and classification. Plots are saved as 'results.png' in the directory where the CSV is located.

    Args:
        file (str, optional): Path to the CSV file containing the training results. Defaults to 'path/to/results.csv'.
        dir (str, optional): Directory where the CSV file is located if 'file' is not provided. Defaults to ''.
        segment (bool, optional): Flag to indicate if the data is for segmentation. Defaults to False.
        pose (bool, optional): Flag to indicate if the data is for pose estimation. Defaults to False.
        classify (bool, optional): Flag to indicate if the data is for classification. Defaults to False.
        on_plot (callable, optional): Callback function to be executed after plotting. Takes filename as an argument.
            Defaults to None.

    Example:
        ```python
        from ultralytics.utils.plotting import plot_results

        plot_results('path/to/results.csv', segment=True)
        ```
    """
    import pandas as pd  # scope for faster 'import ultralytics'
    from scipy.ndimage import gaussian_filter1d

    save_dir = Path(file).parent if file else Path(dir)
    if classify:
        fig, ax = plt.subplots(2, 2, figsize=(6, 6), tight_layout=True)
        index = [1, 4, 2, 3]
    elif segment:
        fig, ax = plt.subplots(2, 8, figsize=(18, 6), tight_layout=True)
        index = [1, 2, 3, 4, 5, 6, 9, 10, 13, 14, 15, 16, 7, 8, 11, 12]
    elif pose:
        fig, ax = plt.subplots(2, 9, figsize=(21, 6), tight_layout=True)
        index = [1, 2, 3, 4, 5, 6, 7, 10, 11, 14, 15, 16, 17, 18, 8, 9, 12, 13]
    else:
        fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)
        index = [1, 2, 3, 4, 5, 8, 9, 10, 6, 7]
    ax = ax.ravel()
    files = list(save_dir.glob("results*.csv"))
    assert len(files), f"No results.csv files found in {save_dir.resolve()}, nothing to plot."
    for f in files:
        try:
            data = pd.read_csv(f)
            s = [x.strip() for x in data.columns]
            x = data.values[:, 0]
            for i, j in enumerate(index):
                y = data.values[:, j].astype("float")
                # y[y == 0] = np.nan  # don't show zero values
                ax[i].plot(x, y, marker=".", label=f.stem, linewidth=2, markersize=8)  # actual results
                ax[i].plot(x, gaussian_filter1d(y, sigma=3), ":", label="smooth", linewidth=2)  # smoothing line
                ax[i].set_title(s[j], fontsize=12)
                # if j in {8, 9, 10}:  # share train and val loss y axes
                #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
        except Exception as e:
            print(f"WARNING: Plotting error for {f}: {e}")
    ax[1].legend()
    fname = save_dir / "results.png"
    fig.savefig(fname, dpi=200)
    plt.close()
    if on_plot:
        on_plot(fname)


def plt_color_scatter(v, f, bins=20, cmap="viridis", alpha=0.8, edgecolors="none"):
    """
    Plots a scatter plot with points colored based on a 2D histogram.

    Args:
        v (array-like): Values for the x-axis.
        f (array-like): Values for the y-axis.
        bins (int, optional): Number of bins for the histogram. Defaults to 20.
        cmap (str, optional): Colormap for the scatter plot. Defaults to 'viridis'.
        alpha (float, optional): Alpha for the scatter plot. Defaults to 0.8.
        edgecolors (str, optional): Edge colors for the scatter plot. Defaults to 'none'.

    Examples:
        >>> v = np.random.rand(100)
        >>> f = np.random.rand(100)
        >>> plt_color_scatter(v, f)
    """

    # Calculate 2D histogram and corresponding colors
    hist, xedges, yedges = np.histogram2d(v, f, bins=bins)
    colors = [
        hist[
            min(np.digitize(v[i], xedges, right=True) - 1, hist.shape[0] - 1),
            min(np.digitize(f[i], yedges, right=True) - 1, hist.shape[1] - 1),
        ]
        for i in range(len(v))
    ]

    # Scatter plot
    plt.scatter(v, f, c=colors, cmap=cmap, alpha=alpha, edgecolors=edgecolors)


def plot_tune_results(csv_file="tune_results.csv"):
    """
    Plot the evolution results stored in an 'tune_results.csv' file. The function generates a scatter plot for each key
    in the CSV, color-coded based on fitness scores. The best-performing configurations are highlighted on the plots.

    Args:
        csv_file (str, optional): Path to the CSV file containing the tuning results. Defaults to 'tune_results.csv'.

    Examples:
        >>> plot_tune_results('path/to/tune_results.csv')
    """

    import pandas as pd  # scope for faster 'import ultralytics'
    from scipy.ndimage import gaussian_filter1d

    # Scatter plots for each hyperparameter
    csv_file = Path(csv_file)
    data = pd.read_csv(csv_file)
    num_metrics_columns = 1
    keys = [x.strip() for x in data.columns][num_metrics_columns:]
    x = data.values
    fitness = x[:, 0]  # fitness
    j = np.argmax(fitness)  # max fitness index
    n = math.ceil(len(keys) ** 0.5)  # columns and rows in plot
    plt.figure(figsize=(10, 10), tight_layout=True)
    for i, k in enumerate(keys):
        v = x[:, i + num_metrics_columns]
        mu = v[j]  # best single result
        plt.subplot(n, n, i + 1)
        plt_color_scatter(v, fitness, cmap="viridis", alpha=0.8, edgecolors="none")
        plt.plot(mu, fitness.max(), "k+", markersize=15)
        plt.title(f"{k} = {mu:.3g}", fontdict={"size": 9})  # limit to 40 characters
        plt.tick_params(axis="both", labelsize=8)  # Set axis label size to 8
        if i % n != 0:
            plt.yticks([])

    file = csv_file.with_name("tune_scatter_plots.png")  # filename
    plt.savefig(file, dpi=200)
    plt.close()

    # Fitness vs iteration
    x = range(1, len(fitness) + 1)
    plt.figure(figsize=(10, 6), tight_layout=True)
    plt.plot(x, fitness, marker="o", linestyle="none", label="fitness")
    plt.plot(x, gaussian_filter1d(fitness, sigma=3), ":", label="smoothed", linewidth=2)  # smoothing line
    plt.title("Fitness vs Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.grid(True)
    plt.legend()

    file = csv_file.with_name("tune_fitness.png")  # filename
    plt.savefig(file, dpi=200)
    plt.close()


def output_to_target(output, max_det=300):
    """Convert model output to target format [batch_id, class_id, x, y, w, h, conf] for plotting."""
    targets = []
    for i, o in enumerate(output):
        box, conf, cls = o[:max_det, :6].cpu().split((4, 1, 1), 1)
        j = torch.full((conf.shape[0], 1), i)
        targets.append(torch.cat((j, cls, xyxy2xywh(box), conf), 1))
    targets = torch.cat(targets, 0).numpy()
    return targets[:, 0], targets[:, 1], targets[:, 2:-1], targets[:, -1]


def output_to_rotated_target(output, max_det=300):
    """Convert model output to target format [batch_id, class_id, x, y, w, h, conf] for plotting."""
    targets = []
    for i, o in enumerate(output):
        box, conf, cls, angle = o[:max_det].cpu().split((4, 1, 1, 1), 1)
        j = torch.full((conf.shape[0], 1), i)
        targets.append(torch.cat((j, cls, box, angle, conf), 1))
    targets = torch.cat(targets, 0).numpy()
    return targets[:, 0], targets[:, 1], targets[:, 2:-1], targets[:, -1]


def feature_visualization(x, module_type, stage, n=32, save_dir=Path("runs/detect/exp")):
    """
    Visualize feature maps of a given model module during inference.

    Args:
        x (torch.Tensor): Features to be visualized.
        module_type (str): Module type.
        stage (int): Module stage within the model.
        n (int, optional): Maximum number of feature maps to plot. Defaults to 32.
        save_dir (Path, optional): Directory to save results. Defaults to Path('runs/detect/exp').
    """
    for m in {"Detect", "Segment", "Pose", "Classify", "OBB", "RTDETRDecoder"}:  # all model heads
        if m in module_type:
            return
    if isinstance(x, torch.Tensor):
        _, channels, height, width = x.shape  # batch, channels, height, width
        if height > 1 and width > 1:
            f = save_dir / f"stage{stage}_{module_type.split('.')[-1]}_features.png"  # filename

            blocks = torch.chunk(x[0].cpu(), channels, dim=0)  # select batch index 0, block by channels
            n = min(n, channels)  # number of plots
            _, ax = plt.subplots(math.ceil(n / 8), 8, tight_layout=True)  # 8 rows x n/8 cols
            ax = ax.ravel()
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            for i in range(n):
                ax[i].imshow(blocks[i].squeeze())  # cmap='gray'
                ax[i].axis("off")

            plt.savefig(f, dpi=300, bbox_inches="tight")
            plt.close()
            np.save(str(f.with_suffix(".npy")), x[0].cpu().numpy())  # npy save


class Profile(contextlib.ContextDecorator):
    """
    YOLOv8 Profile class. Use as a decorator with @Profile() or as a context manager with 'with Profile():'.

    Example:
        ```python
        from ultralytics.utils.ops import Profile

        with Profile(device=device) as dt:
            pass  # slow operation here

        print(dt)  # prints "Elapsed time is 9.5367431640625e-07 s"
        ```
    """

    def __init__(self, t=0.0, device: torch.device = None):
        """
        Initialize the Profile class.

        Args:
            t (float): Initial time. Defaults to 0.0.
            device (torch.device): Devices used for model inference. Defaults to None (cpu).
        """
        self.t = t
        self.device = device
        self.cuda = bool(device and str(device).startswith("cuda"))

    def __enter__(self):
        """Start timing."""
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):  # noqa
        """Stop timing."""
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt

    def __str__(self):
        """Returns a human-readable string representing the accumulated elapsed time in the profiler."""
        return f"Elapsed time is {self.t} s"

    def time(self):
        """Get current time."""
        if self.cuda:
            torch.cuda.synchronize(self.device)
        return time.time()


def segment2box(segment, width=640, height=640):
    """
    Convert 1 segment label to 1 box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy).

    Args:
        segment (torch.Tensor): the segment label
        width (int): the width of the image. Defaults to 640
        height (int): The height of the image. Defaults to 640

    Returns:
        (np.ndarray): the minimum and maximum x and y values of the segment.
    """
    x, y = segment.T  # segment xy
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x = x[inside]
    y = y[inside]
    return (
        np.array([x.min(), y.min(), x.max(), y.max()], dtype=segment.dtype)
        if any(x)
        else np.zeros(4, dtype=segment.dtype)
    )  # xyxy


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding=True, xywh=False):
    """
    Rescales bounding boxes (in the format of xyxy by default) from the shape of the image they were originally
    specified in (img1_shape) to the shape of a different image (img0_shape).

    Args:
        img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
        boxes (torch.Tensor): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
        img0_shape (tuple): the shape of the target image, in the format of (height, width).
        ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be
            calculated based on the size difference between the two images.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.
        xywh (bool): The box format is xywh or not, default=False.

    Returns:
        boxes (torch.Tensor): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (
            round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
            round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
        )  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        boxes[..., 0] -= pad[0]  # x padding
        boxes[..., 1] -= pad[1]  # y padding
        if not xywh:
            boxes[..., 2] -= pad[0]  # x padding
            boxes[..., 3] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    return clip_boxes(boxes, img0_shape)


def make_divisible(x, divisor):
    """
    Returns the nearest number that is divisible by the given divisor.

    Args:
        x (int): The number to make divisible.
        divisor (int | torch.Tensor): The divisor.

    Returns:
        (int): The nearest number divisible by the divisor.
    """
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nc=0,  # number of classes (optional)
    max_time_img=0.05,
    max_nms=30000,
    max_wh=7680,
    in_place=True,
    rotated=False,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

    Args:
        prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes, and all
            classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
            list contains the apriori labels for a given image. The list should be in the format
            output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int, optional): The number of classes output by the model. Any indices after this will be considered masks.
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes into torchvision.ops.nms().
        max_wh (int): The maximum box width and height in pixels.
        in_place (bool): If True, the input prediction tensor will be modified in place.

    Returns:
        (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """
    import torchvision  # scope for faster 'import ultralytics'

    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4  # number of masks
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 2.0 + max_time_img * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    if not rotated:
        if in_place:
            prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy
        else:
            prediction = torch.cat((xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), dim=-1)  # xywh to xyxy

    t = time.time()
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]) and not rotated:
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        scores = x[:, 4]  # scores
        boxes = x[:, :4] + c  # boxes (offset by class)
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections

        # # Experimental
        # merge = False  # use merge-NMS
        # if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
        #     # Update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
        #     from .metrics import box_iou
        #     iou = box_iou(boxes[i], boxes) > iou_thres  # IoU matrix
        #     weights = iou * scores[None]  # box weights
        #     x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
        #     redundant = True  # require redundant detections
        #     if redundant:
        #         i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


def clip_boxes(boxes, shape):
    """
    Takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the shape.

    Args:
        boxes (torch.Tensor): the bounding boxes to clip
        shape (tuple): the shape of the image

    Returns:
        (torch.Tensor | numpy.ndarray): Clipped boxes
    """
    if isinstance(boxes, torch.Tensor):  # faster individually (WARNING: inplace .clamp_() Apple MPS bug)
        boxes[..., 0] = boxes[..., 0].clamp(0, shape[1])  # x1
        boxes[..., 1] = boxes[..., 1].clamp(0, shape[0])  # y1
        boxes[..., 2] = boxes[..., 2].clamp(0, shape[1])  # x2
        boxes[..., 3] = boxes[..., 3].clamp(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
    return boxes


def clip_coords(coords, shape):
    """
    Clip line coordinates to the image boundaries.

    Args:
        coords (torch.Tensor | numpy.ndarray): A list of line coordinates.
        shape (tuple): A tuple of integers representing the size of the image in the format (height, width).

    Returns:
        (torch.Tensor | numpy.ndarray): Clipped coordinates
    """
    if isinstance(coords, torch.Tensor):  # faster individually (WARNING: inplace .clamp_() Apple MPS bug)
        coords[..., 0] = coords[..., 0].clamp(0, shape[1])  # x
        coords[..., 1] = coords[..., 1].clamp(0, shape[0])  # y
    else:  # np.array (faster grouped)
        coords[..., 0] = coords[..., 0].clip(0, shape[1])  # x
        coords[..., 1] = coords[..., 1].clip(0, shape[0])  # y
    return coords


def scale_image(masks, im0_shape, ratio_pad=None):
    """
    Takes a mask, and resizes it to the original image size.

    Args:
        masks (np.ndarray): resized and padded masks/images, [h, w, num]/[h, w, 3].
        im0_shape (tuple): the original image shape
        ratio_pad (tuple): the ratio of the padding to the original image.

    Returns:
        masks (torch.Tensor): The masks that are being returned.
    """
    # Rescale coordinates (xyxy) from im1_shape to im0_shape
    im1_shape = masks.shape
    if im1_shape[:2] == im0_shape[:2]:
        return masks
    if ratio_pad is None:  # calculate from im0_shape
        gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
        pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
    else:
        # gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    top, left = int(pad[1]), int(pad[0])  # y, x
    bottom, right = int(im1_shape[0] - pad[1]), int(im1_shape[1] - pad[0])

    if len(masks.shape) < 2:
        raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
    masks = masks[top:bottom, left:right]
    masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]))
    if len(masks.shape) == 2:
        masks = masks[:, :, None]

    return masks


def xyxy2xywh(x):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x1, y1, x2, y2) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x, y, width, height) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner. Note: ops per 2 channels faster than per channel.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    xy = x[..., :2]  # centers
    wh = x[..., 2:] / 2  # half width-height
    y[..., :2] = xy - wh  # top left xy
    y[..., 2:] = xy + wh  # bottom right xy
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    """
    Convert normalized bounding box coordinates to pixel coordinates.

    Args:
        x (np.ndarray | torch.Tensor): The bounding box coordinates.
        w (int): Width of the image. Defaults to 640
        h (int): Height of the image. Defaults to 640
        padw (int): Padding width. Defaults to 0
        padh (int): Padding height. Defaults to 0
    Returns:
        y (np.ndarray | torch.Tensor): The coordinates of the bounding box in the format [x1, y1, x2, y2] where
            x1,y1 is the top-left corner, x2,y2 is the bottom-right corner of the bounding box.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + padw  # top left x
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + padh  # top left y
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + padw  # bottom right x
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + padh  # bottom right y
    return y


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height, normalized) format. x, y,
    width and height are normalized to image dimensions.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x1, y1, x2, y2) format.
        w (int): The width of the image. Defaults to 640
        h (int): The height of the image. Defaults to 640
        clip (bool): If True, the boxes will be clipped to the image boundaries. Defaults to False
        eps (float): The minimum value of the box's width and height. Defaults to 0.0

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x, y, width, height, normalized) format
    """
    if clip:
        x = clip_boxes(x, (h - eps, w - eps))
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    y[..., 0] = ((x[..., 0] + x[..., 2]) / 2) / w  # x center
    y[..., 1] = ((x[..., 1] + x[..., 3]) / 2) / h  # y center
    y[..., 2] = (x[..., 2] - x[..., 0]) / w  # width
    y[..., 3] = (x[..., 3] - x[..., 1]) / h  # height
    return y


def xywh2ltwh(x):
    """
    Convert the bounding box format from [x, y, w, h] to [x1, y1, w, h], where x1, y1 are the top-left coordinates.

    Args:
        x (np.ndarray | torch.Tensor): The input tensor with the bounding box coordinates in the xywh format

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in the xyltwh format
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    return y


def xyxy2ltwh(x):
    """
    Convert nx4 bounding boxes from [x1, y1, x2, y2] to [x1, y1, w, h], where xy1=top-left, xy2=bottom-right.

    Args:
        x (np.ndarray | torch.Tensor): The input tensor with the bounding boxes coordinates in the xyxy format

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in the xyltwh format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def ltwh2xywh(x):
    """
    Convert nx4 boxes from [x1, y1, w, h] to [x, y, w, h] where xy1=top-left, xy=center.

    Args:
        x (torch.Tensor): the input tensor

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in the xywh format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] + x[..., 2] / 2  # center x
    y[..., 1] = x[..., 1] + x[..., 3] / 2  # center y
    return y


def xyxyxyxy2xywhr(x):
    """
    Convert batched Oriented Bounding Boxes (OBB) from [xy1, xy2, xy3, xy4] to [xywh, rotation]. Rotation values are
    expected in degrees from 0 to 90.

    Args:
        x (numpy.ndarray | torch.Tensor): Input box corners [xy1, xy2, xy3, xy4] of shape (n, 8).

    Returns:
        (numpy.ndarray | torch.Tensor): Converted data in [cx, cy, w, h, rotation] format of shape (n, 5).
    """
    is_torch = isinstance(x, torch.Tensor)
    points = x.cpu().numpy() if is_torch else x
    points = points.reshape(len(x), -1, 2)
    rboxes = []
    for pts in points:
        # NOTE: Use cv2.minAreaRect to get accurate xywhr,
        # especially some objects are cut off by augmentations in dataloader.
        (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
        rboxes.append([cx, cy, w, h, angle / 180 * np.pi])
    return torch.tensor(rboxes, device=x.device, dtype=x.dtype) if is_torch else np.asarray(rboxes)


def xywhr2xyxyxyxy(x):
    """
    Convert batched Oriented Bounding Boxes (OBB) from [xywh, rotation] to [xy1, xy2, xy3, xy4]. Rotation values should
    be in degrees from 0 to 90.

    Args:
        x (numpy.ndarray | torch.Tensor): Boxes in [cx, cy, w, h, rotation] format of shape (n, 5) or (b, n, 5).

    Returns:
        (numpy.ndarray | torch.Tensor): Converted corner points of shape (n, 4, 2) or (b, n, 4, 2).
    """
    cos, sin, cat, stack = (
        (torch.cos, torch.sin, torch.cat, torch.stack)
        if isinstance(x, torch.Tensor)
        else (np.cos, np.sin, np.concatenate, np.stack)
    )

    ctr = x[..., :2]
    w, h, angle = (x[..., i : i + 1] for i in range(2, 5))
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = cat(vec1, -1)
    vec2 = cat(vec2, -1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return stack([pt1, pt2, pt3, pt4], -2)


def ltwh2xyxy(x):
    """
    It converts the bounding box from [x1, y1, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right.

    Args:
        x (np.ndarray | torch.Tensor): the input image

    Returns:
        y (np.ndarray | torch.Tensor): the xyxy coordinates of the bounding boxes.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 2] = x[..., 2] + x[..., 0]  # width
    y[..., 3] = x[..., 3] + x[..., 1]  # height
    return y


def segments2boxes(segments):
    """
    It converts segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh)

    Args:
        segments (list): list of segments, each segment is a list of points, each point is a list of x, y coordinates

    Returns:
        (np.ndarray): the xywh coordinates of the bounding boxes.
    """
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return xyxy2xywh(np.array(boxes))  # cls, xywh


def resample_segments(segments, n=1000):
    """
    Inputs a list of segments (n,2) and returns a list of segments (n,2) up-sampled to n points each.

    Args:
        segments (list): a list of (n,2) arrays, where n is the number of points in the segment.
        n (int): number of points to resample the segment to. Defaults to 1000

    Returns:
        segments (list): the resampled segments.
    """
    for i, s in enumerate(segments):
        s = np.concatenate((s, s[0:1, :]), axis=0)
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = (
            np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)], dtype=np.float32).reshape(2, -1).T
        )  # segment xy
    return segments


def crop_mask(masks, boxes):
    """
    It takes a mask and a bounding box, and returns a mask that is cropped to the bounding box.

    Args:
        masks (torch.Tensor): [n, h, w] tensor of masks
        boxes (torch.Tensor): [n, 4] tensor of bbox coordinates in relative point form

    Returns:
        (torch.Tensor): The masks are being cropped to the bounding box.
    """
    _, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(n,1,1)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,1,w)
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(1,h,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def process_mask_upsample(protos, masks_in, bboxes, shape):
    """
    Takes the output of the mask head, and applies the mask to the bounding boxes. This produces masks of higher quality
    but is slower.

    Args:
        protos (torch.Tensor): [mask_dim, mask_h, mask_w]
        masks_in (torch.Tensor): [n, mask_dim], n is number of masks after nms
        bboxes (torch.Tensor): [n, 4], n is number of masks after nms
        shape (tuple): the size of the input image (h,w)

    Returns:
        (torch.Tensor): The upsampled masks.
    """
    c, mh, mw = protos.shape  # CHW
    masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)
    masks = F.interpolate(masks[None], shape, mode="bilinear", align_corners=False)[0]  # CHW
    masks = crop_mask(masks, bboxes)  # CHW
    return masks.gt_(0.5)


def process_mask(protos, masks_in, bboxes, shape, upsample=False):
    """
    Apply masks to bounding boxes using the output of the mask head.

    Args:
        protos (torch.Tensor): A tensor of shape [mask_dim, mask_h, mask_w].
        masks_in (torch.Tensor): A tensor of shape [n, mask_dim], where n is the number of masks after NMS.
        bboxes (torch.Tensor): A tensor of shape [n, 4], where n is the number of masks after NMS.
        shape (tuple): A tuple of integers representing the size of the input image in the format (h, w).
        upsample (bool): A flag to indicate whether to upsample the mask to the original image size. Default is False.

    Returns:
        (torch.Tensor): A binary mask tensor of shape [n, h, w], where n is the number of masks after NMS, and h and w
            are the height and width of the input image. The mask is applied to the bounding boxes.
    """

    c, mh, mw = protos.shape  # CHW
    ih, iw = shape
    masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)  # CHW
    width_ratio = mw / iw
    height_ratio = mh / ih

    downsampled_bboxes = bboxes.clone()
    downsampled_bboxes[:, 0] *= width_ratio
    downsampled_bboxes[:, 2] *= width_ratio
    downsampled_bboxes[:, 3] *= height_ratio
    downsampled_bboxes[:, 1] *= height_ratio

    masks = crop_mask(masks, downsampled_bboxes)  # CHW
    if upsample:
        masks = F.interpolate(masks[None], shape, mode="bilinear", align_corners=False)[0]  # CHW
    return masks.gt_(0.5)


def process_mask_native(protos, masks_in, bboxes, shape):
    """
    It takes the output of the mask head, and crops it after upsampling to the bounding boxes.

    Args:
        protos (torch.Tensor): [mask_dim, mask_h, mask_w]
        masks_in (torch.Tensor): [n, mask_dim], n is number of masks after nms
        bboxes (torch.Tensor): [n, 4], n is number of masks after nms
        shape (tuple): the size of the input image (h,w)

    Returns:
        masks (torch.Tensor): The returned masks with dimensions [h, w, n]
    """
    c, mh, mw = protos.shape  # CHW
    masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)
    masks = scale_masks(masks[None], shape)[0]  # CHW
    masks = crop_mask(masks, bboxes)  # CHW
    return masks.gt_(0.5)


def scale_masks(masks, shape, padding=True):
    """
    Rescale segment masks to shape.

    Args:
        masks (torch.Tensor): (N, C, H, W).
        shape (tuple): Height and width.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.
    """
    mh, mw = masks.shape[2:]
    gain = min(mh / shape[0], mw / shape[1])  # gain  = old / new
    pad = [mw - shape[1] * gain, mh - shape[0] * gain]  # wh padding
    if padding:
        pad[0] /= 2
        pad[1] /= 2
    top, left = (int(pad[1]), int(pad[0])) if padding else (0, 0)  # y, x
    bottom, right = (int(mh - pad[1]), int(mw - pad[0]))
    masks = masks[..., top:bottom, left:right]

    masks = F.interpolate(masks, shape, mode="bilinear", align_corners=False)  # NCHW
    return masks


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None, normalize=False, padding=True):
    """
    Rescale segment coordinates (xy) from img1_shape to img0_shape.

    Args:
        img1_shape (tuple): The shape of the image that the coords are from.
        coords (torch.Tensor): the coords to be scaled of shape n,2.
        img0_shape (tuple): the shape of the image that the segmentation is being applied to.
        ratio_pad (tuple): the ratio of the image size to the padded image size.
        normalize (bool): If True, the coordinates will be normalized to the range [0, 1]. Defaults to False.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.

    Returns:
        coords (torch.Tensor): The scaled coordinates.
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        coords[..., 0] -= pad[0]  # x padding
        coords[..., 1] -= pad[1]  # y padding
    coords[..., 0] /= gain
    coords[..., 1] /= gain
    coords = clip_coords(coords, img0_shape)
    if normalize:
        coords[..., 0] /= img0_shape[1]  # width
        coords[..., 1] /= img0_shape[0]  # height
    return coords


def regularize_rboxes(rboxes):
    """
    Regularize rotated boxes in range [0, pi/2].

    Args:
        rboxes (torch.Tensor): Input boxes of shape(N, 5) in xywhr format.

    Returns:
        (torch.Tensor): The regularized boxes.
    """
    x, y, w, h, t = rboxes.unbind(dim=-1)
    # Swap edge and angle if h >= w
    w_ = torch.where(w > h, w, h)
    h_ = torch.where(w > h, h, w)
    t = torch.where(w > h, t, t + math.pi / 2) % math.pi
    return torch.stack([x, y, w_, h_, t], dim=-1)  # regularized boxes


def masks2segments(masks, strategy="largest"):
    """
    It takes a list of masks(n,h,w) and returns a list of segments(n,xy)

    Args:
        masks (torch.Tensor): the output of the model, which is a tensor of shape (batch_size, 160, 160)
        strategy (str): 'concat' or 'largest'. Defaults to largest

    Returns:
        segments (List): list of segment masks
    """
    segments = []
    for x in masks.int().cpu().numpy().astype("uint8"):
        c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if c:
            if strategy == "concat":  # concatenate all segments
                c = np.concatenate([x.reshape(-1, 2) for x in c])
            elif strategy == "largest":  # select largest segment
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
        else:
            c = np.zeros((0, 2))  # no segments found
        segments.append(c.astype("float32"))
    return segments


def convert_torch2numpy_batch(batch: torch.Tensor) -> np.ndarray:
    """
    Convert a batch of FP32 torch tensors (0.0-1.0) to a NumPy uint8 array (0-255), changing from BCHW to BHWC layout.

    Args:
        batch (torch.Tensor): Input tensor batch of shape (Batch, Channels, Height, Width) and dtype torch.float32.

    Returns:
        (np.ndarray): Output NumPy array batch of shape (Batch, Height, Width, Channels) and dtype uint8.
    """
    return (batch.permute(0, 2, 3, 1).contiguous() * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()


def clean_str(s):
    """
    Cleans a string by replacing special characters with underscore _

    Args:
        s (str): a string needing special characters replaced

    Returns:
        (str): a string with special characters replaced by an underscore _
    """
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)



class BaseTensor:
    """Base tensor class with additional methods for easy manipulation and device handling."""

    def __init__(self, data, orig_shape) -> None:
        """
        Initialize BaseTensor with data and original shape.

        Args:
            data (torch.Tensor | np.ndarray): Predictions, such as bboxes, masks and keypoints.
            orig_shape (tuple): Original shape of image.
        """
        assert isinstance(data, (torch.Tensor, np.ndarray))
        self.data = data
        self.orig_shape = orig_shape

    @property
    def shape(self):
        """Return the shape of the data tensor."""
        return self.data.shape

    def cpu(self):
        """Return a copy of the tensor on CPU memory."""
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.cpu(), self.orig_shape)

    def numpy(self):
        """Return a copy of the tensor as a numpy array."""
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.numpy(), self.orig_shape)

    def cuda(self):
        """Return a copy of the tensor on GPU memory."""
        return self.__class__(torch.as_tensor(self.data).cuda(), self.orig_shape)

    def to(self, *args, **kwargs):
        """Return a copy of the tensor with the specified device and dtype."""
        return self.__class__(torch.as_tensor(self.data).to(*args, **kwargs), self.orig_shape)

    def __len__(self):  # override len(results)
        """Return the length of the data tensor."""
        return len(self.data)

    def __getitem__(self, idx):
        """Return a BaseTensor with the specified index of the data tensor."""
        return self.__class__(self.data[idx], self.orig_shape)


class Results:
    """
    A class for storing and manipulating inference results.

    Attributes:
        orig_img (numpy.ndarray): Original image as a numpy array.
        orig_shape (tuple): Original image shape in (height, width) format.
        boxes (Boxes, optional): Object containing detection bounding boxes.
        masks (Masks, optional): Object containing detection masks.
        probs (Probs, optional): Object containing class probabilities for classification tasks.
        keypoints (Keypoints, optional): Object containing detected keypoints for each object.
        speed (dict): Dictionary of preprocess, inference, and postprocess speeds (ms/image).
        names (dict): Dictionary of class names.
        path (str): Path to the image file.

    Methods:
        update(boxes=None, masks=None, probs=None, obb=None): Updates object attributes with new detection results.
        cpu(): Returns a copy of the Results object with all tensors on CPU memory.
        numpy(): Returns a copy of the Results object with all tensors as numpy arrays.
        cuda(): Returns a copy of the Results object with all tensors on GPU memory.
        to(*args, **kwargs): Returns a copy of the Results object with tensors on a specified device and dtype.
        new(): Returns a new Results object with the same image, path, and names.
        plot(...): Plots detection results on an input image, returning an annotated image.
        show(): Show annotated results to screen.
        save(filename): Save annotated results to file.
        verbose(): Returns a log string for each task, detailing detections and classifications.
        save_txt(txt_file, save_conf=False): Saves detection results to a text file.
        save_crop(save_dir, file_name=Path("im.jpg")): Saves cropped detection images.
        tojson(normalize=False): Converts detection results to JSON format.
    """

    def __init__(
        self, orig_img, path, names, boxes=None, masks=None, probs=None, keypoints=None, obb=None, speed=None
    ) -> None:
        """
        Initialize the Results class.

        Args:
            orig_img (numpy.ndarray): The original image as a numpy array.
            path (str): The path to the image file.
            names (dict): A dictionary of class names.
            boxes (torch.tensor, optional): A 2D tensor of bounding box coordinates for each detection.
            masks (torch.tensor, optional): A 3D tensor of detection masks, where each mask is a binary image.
            probs (torch.tensor, optional): A 1D tensor of probabilities of each class for classification task.
            keypoints (torch.tensor, optional): A 2D tensor of keypoint coordinates for each detection.
            obb (torch.tensor, optional): A 2D tensor of oriented bounding box coordinates for each detection.
        """
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]
        self.boxes = Boxes(boxes, self.orig_shape) if boxes is not None else None  # native size boxes
        self.probs = Probs(probs) if probs is not None else None
        self.speed = speed if speed is not None else {"preprocess": None, "inference": None, "postprocess": None}
        self.names = names
        self.path = path
        self.save_dir = None
        self._keys = "boxes", "masks", "probs", "keypoints", "obb"

    def __getitem__(self, idx):
        """Return a Results object for the specified index."""
        return self._apply("__getitem__", idx)

    def __len__(self):
        """Return the number of detections in the Results object."""
        for k in self._keys:
            v = getattr(self, k)
            if v is not None:
                return len(v)

    def update(self, boxes=None, masks=None, probs=None, obb=None):
        """Update the boxes, masks, and probs attributes of the Results object."""
        if boxes is not None:
            self.boxes = Boxes(clip_boxes(boxes, self.orig_shape), self.orig_shape)
        if probs is not None:
            self.probs = probs

    def _apply(self, fn, *args, **kwargs):
        """
        Applies a function to all non-empty attributes and returns a new Results object with modified attributes. This
        function is internally called by methods like .to(), .cuda(), .cpu(), etc.

        Args:
            fn (str): The name of the function to apply.
            *args: Variable length argument list to pass to the function.
            **kwargs: Arbitrary keyword arguments to pass to the function.

        Returns:
            Results: A new Results object with attributes modified by the applied function.
        """
        r = self.new()
        for k in self._keys:
            v = getattr(self, k)
            if v is not None:
                setattr(r, k, getattr(v, fn)(*args, **kwargs))
        return r

    def cpu(self):
        """Return a copy of the Results object with all tensors on CPU memory."""
        return self._apply("cpu")

    def numpy(self):
        """Return a copy of the Results object with all tensors as numpy arrays."""
        return self._apply("numpy")

    def cuda(self):
        """Return a copy of the Results object with all tensors on GPU memory."""
        return self._apply("cuda")

    def to(self, *args, **kwargs):
        """Return a copy of the Results object with tensors on the specified device and dtype."""
        return self._apply("to", *args, **kwargs)

    def new(self):
        """Return a new Results object with the same image, path, names and speed."""
        return Results(orig_img=self.orig_img, path=self.path, names=self.names, speed=self.speed)

    def plot(
        self,
        conf=True,
        line_width=None,
        font_size=None,
        font="Arial.ttf",
        pil=False,
        img=None,
        im_gpu=None,
        kpt_radius=5,
        kpt_line=True,
        labels=True,
        boxes=True,
        masks=True,
        probs=True,
        show=False,
        save=False,
        filename=None,
    ):
        """
        Plots the detection results on an input RGB image. Accepts a numpy array (cv2) or a PIL Image.

        Args:
            conf (bool): Whether to plot the detection confidence score.
            line_width (float, optional): The line width of the bounding boxes. If None, it is scaled to the image size.
            font_size (float, optional): The font size of the text. If None, it is scaled to the image size.
            font (str): The font to use for the text.
            pil (bool): Whether to return the image as a PIL Image.
            img (numpy.ndarray): Plot to another image. if not, plot to original image.
            im_gpu (torch.Tensor): Normalized image in gpu with shape (1, 3, 640, 640), for faster mask plotting.
            kpt_radius (int, optional): Radius of the drawn keypoints. Default is 5.
            kpt_line (bool): Whether to draw lines connecting keypoints.
            labels (bool): Whether to plot the label of bounding boxes.
            boxes (bool): Whether to plot the bounding boxes.
            masks (bool): Whether to plot the masks.
            probs (bool): Whether to plot classification probability
            show (bool): Whether to display the annotated image directly.
            save (bool): Whether to save the annotated image to `filename`.
            filename (str): Filename to save image to if save is True.

        Returns:
            (numpy.ndarray): A numpy array of the annotated image.

        Example:
            ```python
            from PIL import Image
            from ultralytics import YOLO

            model = YOLO('yolov8n.pt')
            results = model('bus.jpg')  # results list
            for r in results:
                im_array = r.plot()  # plot a BGR numpy array of predictions
                im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
                im.show()  # show image
                im.save('results.jpg')  # save image
            ```
        """
        if img is None and isinstance(self.orig_img, torch.Tensor):
            img = (self.orig_img[0].detach().permute(1, 2, 0).contiguous() * 255).to(torch.uint8).cpu().numpy()

        names = self.names
        pred_boxes, show_boxes = self.boxes, boxes
        pred_probs, show_probs = self.probs, probs
        annotator = Annotator(
            deepcopy(self.orig_img if img is None else img),
            line_width,
            font_size,
            font,
            pil or (pred_probs is not None and show_probs),  # Classify tasks default to pil=True
            example=names,
        )

        # Plot Detect results
        if pred_boxes is not None and show_boxes:
            for d in reversed(pred_boxes):
                c, conf, id = int(d.cls), float(d.conf) if conf else None, None if d.id is None else int(d.id.item())
                name = ("" if id is None else f"id:{id} ") + names[c]
                label = (f"{name} {conf:.2f}" if conf else name) if labels else None
                box = d.xyxy.squeeze()
                annotator.box_label(box, label, color=colors(c, True), rotated=False)

        # Plot Classify results
        if pred_probs is not None and show_probs:
            text = ",\n".join(f"{names[j] if names else j} {pred_probs.data[j]:.2f}" for j in pred_probs.top5)
            x = round(self.orig_shape[0] * 0.03)
            annotator.text([x, x], text, txt_color=(255, 255, 255))  # TODO: allow setting colors

        # Show results
        if show:
            annotator.show(self.path)

        # Save results
        if save:
            annotator.save(filename)

        return annotator.result()

    def show(self, *args, **kwargs):
        """Show annotated results image."""
        self.plot(show=True, *args, **kwargs)

    def save(self, filename=None, *args, **kwargs):
        """Save annotated results image."""
        if not filename:
            filename = f"results_{Path(self.path).name}"
        self.plot(save=True, filename=filename, *args, **kwargs)
        return filename

    def verbose(self):
        """Return log string for each task."""
        log_string = ""
        probs = self.probs
        boxes = self.boxes
        if len(self) == 0:
            return log_string if probs is not None else f"{log_string}(no detections), "
        if probs is not None:
            log_string += f"{', '.join(f'{self.names[j]} {probs.data[j]:.2f}' for j in probs.top5)}, "
        if boxes:
            for c in boxes.cls.unique():
                n = (boxes.cls == c).sum()  # detections per class
                log_string += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "
        return log_string

    def save_txt(self, txt_file, save_conf=False):
        """
        Save predictions into txt file.

        Args:
            txt_file (str): txt file path.
            save_conf (bool): save confidence score or not.
        """
        boxes = self.boxes
        probs = self.probs
        texts = []
        if probs is not None:
            # Classify
            [texts.append(f"{probs.data[j]:.2f} {self.names[j]}") for j in probs.top5]
        elif boxes:
            # Detect/segment/pose
            for j, d in enumerate(boxes):
                c, conf, id = int(d.cls), float(d.conf), None if d.id is None else int(d.id.item())
                line = (c, *(d.xywhn.view(-1)))
                line += (conf,) * save_conf + (() if id is None else (id,))
                texts.append(("%g " * len(line)).rstrip() % line)

        if texts:
            Path(txt_file).parent.mkdir(parents=True, exist_ok=True)  # make directory
            with open(txt_file, "a") as f:
                f.writelines(text + "\n" for text in texts)

    def save_crop(self, save_dir, file_name=Path("im.jpg")):
        """
        Save cropped predictions to `save_dir/cls/file_name.jpg`.

        Args:
            save_dir (str | pathlib.Path): Save path.
            file_name (str | pathlib.Path): File name.
        """
        for d in self.boxes:
            save_one_box(
                d.xyxy,
                self.orig_img.copy(),
                file=Path(save_dir) / self.names[int(d.cls)] / f"{Path(file_name)}.jpg",
                BGR=True,
            )

    def summary(self, normalize=False, decimals=5):
        """Convert the results to a summarized format."""
        # Create list of detection dictionaries
        results = []
        if self.probs is not None:
            class_id = self.probs.top1
            results.append(
                {
                    "name": self.names[class_id],
                    "class": class_id,
                    "confidence": round(self.probs.top1conf.item(), decimals),
                }
            )
            return results

        is_obb = self.obb is not None
        data = self.obb if is_obb else self.boxes
        h, w = self.orig_shape if normalize else (1, 1)
        for i, row in enumerate(data):  # xyxy, track_id if tracking, conf, class_id
            class_id, conf = int(row.cls), round(row.conf.item(), decimals)
            box = (row.xyxyxyxy if is_obb else row.xyxy).squeeze().reshape(-1, 2).tolist()
            xy = {}
            for j, b in enumerate(box):
                xy[f"x{j + 1}"] = round(b[0] / w, decimals)
                xy[f"y{j + 1}"] = round(b[1] / h, decimals)
            result = {"name": self.names[class_id], "class": class_id, "confidence": conf, "box": xy}
            if data.is_track:
                result["track_id"] = int(row.id.item())  # track ID
            if self.masks:
                result["segments"] = {
                    "x": (self.masks.xy[i][:, 0] / w).round(decimals).tolist(),
                    "y": (self.masks.xy[i][:, 1] / h).round(decimals).tolist(),
                }
            if self.keypoints is not None:
                x, y, visible = self.keypoints[i].data[0].cpu().unbind(dim=1)  # torch Tensor
                result["keypoints"] = {
                    "x": (x / w).numpy().round(decimals).tolist(),  # decimals named argument required
                    "y": (y / h).numpy().round(decimals).tolist(),
                    "visible": visible.numpy().round(decimals).tolist(),
                }
            results.append(result)

        return results

    def tojson(self, normalize=False, decimals=5):
        """Convert the results to JSON format."""
        import json

        return json.dumps(self.summary(normalize=normalize, decimals=decimals), indent=2)


class Boxes(BaseTensor):
    """
    Manages detection boxes, providing easy access and manipulation of box coordinates, confidence scores, class
    identifiers, and optional tracking IDs. Supports multiple formats for box coordinates, including both absolute and
    normalized forms.

    Attributes:
        data (torch.Tensor): The raw tensor containing detection boxes and their associated data.
        orig_shape (tuple): The original image size as a tuple (height, width), used for normalization.
        is_track (bool): Indicates whether tracking IDs are included in the box data.

    Properties:
        xyxy (torch.Tensor | numpy.ndarray): Boxes in [x1, y1, x2, y2] format.
        conf (torch.Tensor | numpy.ndarray): Confidence scores for each box.
        cls (torch.Tensor | numpy.ndarray): Class labels for each box.
        id (torch.Tensor | numpy.ndarray, optional): Tracking IDs for each box, if available.
        xywh (torch.Tensor | numpy.ndarray): Boxes in [x, y, width, height] format, calculated on demand.
        xyxyn (torch.Tensor | numpy.ndarray): Normalized [x1, y1, x2, y2] boxes, relative to `orig_shape`.
        xywhn (torch.Tensor | numpy.ndarray): Normalized [x, y, width, height] boxes, relative to `orig_shape`.

    Methods:
        cpu(): Moves the boxes to CPU memory.
        numpy(): Converts the boxes to a numpy array format.
        cuda(): Moves the boxes to CUDA (GPU) memory.
        to(device, dtype=None): Moves the boxes to the specified device.
    """

    def __init__(self, boxes, orig_shape) -> None:
        """
        Initialize the Boxes class.

        Args:
            boxes (torch.Tensor | numpy.ndarray): A tensor or numpy array containing the detection boxes, with
                shape (num_boxes, 6) or (num_boxes, 7). The last two columns contain confidence and class values.
                If present, the third last column contains track IDs.
            orig_shape (tuple): Original image size, in the format (height, width).
        """
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        n = boxes.shape[-1]
        assert n in {6, 7}, f"expected 6 or 7 values but got {n}"  # xyxy, track_id, conf, cls
        super().__init__(boxes, orig_shape)
        self.is_track = n == 7
        self.orig_shape = orig_shape

    @property
    def xyxy(self):
        """Return the boxes in xyxy format."""
        return self.data[:, :4]

    @property
    def conf(self):
        """Return the confidence values of the boxes."""
        return self.data[:, -2]

    @property
    def cls(self):
        """Return the class values of the boxes."""
        return self.data[:, -1]

    @property
    def id(self):
        """Return the track IDs of the boxes (if available)."""
        return self.data[:, -3] if self.is_track else None

    @property
    @lru_cache(maxsize=2)  # maxsize 1 should suffice
    def xywh(self):
        """Return the boxes in xywh format."""
        return xyxy2xywh(self.xyxy)

    @property
    @lru_cache(maxsize=2)
    def xyxyn(self):
        """Return the boxes in xyxy format normalized by original image size."""
        xyxy = self.xyxy.clone() if isinstance(self.xyxy, torch.Tensor) else np.copy(self.xyxy)
        xyxy[..., [0, 2]] /= self.orig_shape[1]
        xyxy[..., [1, 3]] /= self.orig_shape[0]
        return xyxy

    @property
    @lru_cache(maxsize=2)
    def xywhn(self):
        """Return the boxes in xywh format normalized by original image size."""
        xywh = xyxy2xywh(self.xyxy)
        xywh[..., [0, 2]] /= self.orig_shape[1]
        xywh[..., [1, 3]] /= self.orig_shape[0]
        return xywh


class Probs(BaseTensor):
    """
    A class for storing and manipulating classification predictions.

    Attributes:
        top1 (int): Index of the top 1 class.
        top5 (list[int]): Indices of the top 5 classes.
        top1conf (torch.Tensor): Confidence of the top 1 class.
        top5conf (torch.Tensor): Confidences of the top 5 classes.

    Methods:
        cpu(): Returns a copy of the probs tensor on CPU memory.
        numpy(): Returns a copy of the probs tensor as a numpy array.
        cuda(): Returns a copy of the probs tensor on GPU memory.
        to(): Returns a copy of the probs tensor with the specified device and dtype.
    """

    def __init__(self, probs, orig_shape=None) -> None:
        """Initialize the Probs class with classification probabilities and optional original shape of the image."""
        super().__init__(probs, orig_shape)

    @property
    @lru_cache(maxsize=1)
    def top1(self):
        """Return the index of top 1."""
        return int(self.data.argmax())

    @property
    @lru_cache(maxsize=1)
    def top5(self):
        """Return the indices of top 5."""
        return (-self.data).argsort(0)[:5].tolist()  # this way works with both torch and numpy.

    @property
    @lru_cache(maxsize=1)
    def top1conf(self):
        """Return the confidence of top 1."""
        return self.data[self.top1]

    @property
    @lru_cache(maxsize=1)
    def top5conf(self):
        """Return the confidences of top 5."""
        return self.data[self.top5]
