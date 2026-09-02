"""렌더 프레임 위아래에 띠를 붙이고 그 안에만 영어 자막을 넣는다.

3D 영역을 침범하지 않으므로 카메라가 어디로 가든 로봇과 사람을 가리지 않는다.
render_compare.py에서만 쓴다.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
MONO_B = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

BAND = (14, 16, 20)
WHITE = (255, 255, 255)
DIM = (170, 176, 186)
ORANGE = (255, 150, 40)
GREY = (205, 208, 214)

TOP = 110
BOTTOM = 130


class Overlay:
    """scene_h 높이의 3D 프레임을 받아 (TOP + scene_h + BOTTOM) 높이로 돌려준다."""

    def __init__(self, seq, fps, human_seg, robot_seg):
        self.seq = seq
        self.fps = fps
        self.h = human_seg
        self.r = robot_seg
        self.f_title = ImageFont.truetype(FONT_B, 34)
        self.f_sub = ImageFont.truetype(FONT, 20)
        self.f_key = ImageFont.truetype(MONO_B, 19)
        self.f_val = ImageFont.truetype(MONO, 19)
        self.f_big = ImageFont.truetype(MONO_B, 30)

    def draw(self, img, frame_idx, err=None):
        sh, sw = img.shape[0], img.shape[1]
        canvas = Image.new("RGB", (sw, TOP + sh + BOTTOM), BAND)
        canvas.paste(Image.fromarray(img), (0, TOP))
        d = ImageDraw.Draw(canvas)
        H = canvas.size[1]

        d.text((32, 20), "Human Mocap → Unitree G1 Retargeting",
               font=self.f_title, fill=WHITE)
        t = frame_idx / self.fps
        d.text((34, 66),
               f"LAFAN1   {self.seq}   {t:6.2f} s   frame {frame_idx}",
               font=self.f_sub, fill=DIM)

        lx = sw - 430
        d.rectangle((lx, 24, lx + 18, 42), fill=ORANGE)
        d.text((lx + 30, 22), "source human  (LAFAN1 skeleton)",
               font=self.f_sub, fill=DIM)
        d.rectangle((lx, 58, lx + 18, 76), fill=GREY)
        d.text((lx + 30, 56), "retargeted G1  (GMR)",
               font=self.f_sub, fill=DIM)

        y = H - BOTTOM + 20
        d.text((32, y), "EMBODIMENT", font=self.f_key, fill=WHITE)
        x = 190
        for label, key in (("thigh", "thigh"), ("shank", "shank"),
                           ("upper arm", "upper_arm")):
            d.text((x, y), f"{label} ", font=self.f_val, fill=DIM)
            x += len(label) * 12 + 14
            d.text((x, y), f"{self.h[key] * 100:.1f}", font=self.f_val,
                   fill=ORANGE)
            x += 54
            d.text((x, y), "→", font=self.f_val, fill=DIM)
            x += 24
            d.text((x, y), f"{self.r[key] * 100:.1f} cm", font=self.f_val,
                   fill=GREY)
            x += 116

        if err is not None:
            y2 = H - BOTTOM + 62
            m = err["mean"]
            d.text((32, y2 + 6), "IK TARGET ERROR", font=self.f_key,
                   fill=WHITE)
            d.text((225, y2),
                   f"feet {err['feet']:.1f} \u00b7 hands {err['hands']:.1f} cm",
                   font=self.f_big, fill=WHITE)
            d.text((680, y2 + 6),
                   f"seq mean {m['feet']:.1f} / {m['hands']:.1f} cm",
                   font=self.f_val, fill=DIM)
            d.text((32, y2 + 36),
                   "targets = LAFAN1 human scaled to G1 embodiment",
                   font=self.f_val, fill=DIM)

        return np.asarray(canvas)
