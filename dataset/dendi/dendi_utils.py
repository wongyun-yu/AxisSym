import os.path as osp
import numpy as np
import torch
import albumentations as A
from PIL import Image, ImageDraw


def draw_line(lines, size):
        w, h = size
        if lines.shape[0] > 0:
            axis = Image.new('L', size)
            draw = ImageDraw.Draw(axis)

            for idx, coords in enumerate(lines):
                if coords[0] > coords[2]:
                    coords = np.roll(coords, -2)
                _coords = [coords[0]*w, coords[1]*h, coords[2]*w, coords[3]*h]
                draw.line(_coords, fill=(idx + 1))

            axis = np.asarray(axis).astype(np.float32)
            axis = torch.from_numpy(axis).unsqueeze(0).unsqueeze(1)
            axis = (axis > 0).float()
        else:
            axis = torch.zeros(1, 1, h, w)
        return axis

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def intersect(A,B,C,D):
    # https://stackoverflow.com/a/9997374
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def clamp(num, min_value=0, max_value=1):
    min_value = torch.tensor(min_value)
    max_value = torch.tensor(max_value)
    return max(min(num, max_value), min_value)

def calibrate_lines(lines):
    x1, y1, x2, y2 = lines
    new_x1, new_y1, new_x2, new_y2 = x1, y1, x2, y2
    invalid = False
    
    cross_t = intersect((x1, y1), (x2, y2), (0, 0), (1, 0))
    cross_b = intersect((x1, y1), (x2, y2), (0, 1), (1, 1))
    cross_l = intersect((x1, y1), (x2, y2), (0, 0), (0, 1))
    cross_r = intersect((x1, y1), (x2, y2), (1, 0), (1, 1))
    
    if not cross_t + cross_b + cross_l + cross_r:
        mid_x, mid_y = (x1 + x2)/2, (y1 + y2) / 2
        valid = (mid_x > 0) & (mid_y > 0) & (mid_x < 1) & (mid_y < 1)
        invalid = not valid
        if invalid:
            return
        return new_x1, new_y1, new_x2, new_y2 #, cross_t, cross_b, cross_l, cross_r, invalid    
    
    if x1 == x2:
        invalid = (x1 < 0) or (x1 > 1)
        invalid = ((y1 > 1) and (y2 > 1)) or ((y1 < 0) and (y2 < 0))
        if not invalid:
            new_x1, new_y1, new_x2, new_y2 = \
                clamp(x1), clamp(y1), clamp(x2), clamp(y2)
    elif y1 == y2:
        invalid = (y1 < 0) or (y1 > 1)
        invalid = ((x1 > 1) and (x2 > 1)) or ((x1 < 0) and (x2 < 0))
        if not invalid:
            new_x1, new_y1, new_x2, new_y2 = \
                clamp(x1), clamp(y1), clamp(x2), clamp(y2)
    else:
        # y = mx + c
        m = (y2 - y1) / (x2 - x1)
        c = - m*x2 + (y2)
        # print(m, c)
        zero, one = torch.zeros(1,), torch.ones(1,)
        m, c = torch.tensor([m]), torch.tensor([c])
        inter_t = -c/m, zero
        inter_b = (1-c)/m, one
        inter_l = zero, c
        inter_r = one, m+c
        inter = [inter_t, inter_b, inter_l, inter_r]
        if cross_t + cross_b + cross_l + cross_r == 2:
            cross = [cross_t, cross_b, cross_l, cross_r]
            inter = [inter[i] for i, _cross in enumerate(cross) if _cross]
            new_x1, new_y1 = inter[0]
            new_x2, new_y2 = inter[1]
        else:
            if cross_t:
                if y2 > y1:
                    new_x1, new_y1 = inter_t
                else:
                    new_x2, new_y2 = inter_t
            elif cross_b:
                if y2 < y1:
                    new_x1, new_y1 = inter_b
                else:
                    new_x2, new_y2 = inter_b
            elif cross_l:
                if x2 > x1:
                    new_x1, new_y1 = inter_l
                else:
                    new_x2, new_y2 = inter_l
            elif cross_r:
                if x2 < x1:
                    new_x1, new_y1 = inter_r
                else:
                    new_x2, new_y2 = inter_r

    return new_x1, new_y1, new_x2, new_y2#, cross_t, cross_b, cross_l, cross_r, invalid
