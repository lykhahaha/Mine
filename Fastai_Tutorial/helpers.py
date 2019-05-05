from fastai import vision
import numpy as np

#------------------------------------------for lesson3-headpose.py----------------------------------------------
def convert_biwi(coords, cal):
    c1 = coords[0] * cal[0][0]/coords[2] + cal[0][2]
    c2 = coords[1] * cal[1][1]/coords[2] + cal[1][2]
    return vision.tensor([c2, c1])

def get_center_pt(image_path, cal, image_path2text_file_name_fn):
    center_pt = np.genfromtxt(image_path2text_file_name_fn(image_path), skip_header=3)
    return convert_biwi(center_pt, cal)

def get_image_pt(image, points):
    return vision.ImagePoints(vision.FlowField(image.size, points), scale=True)