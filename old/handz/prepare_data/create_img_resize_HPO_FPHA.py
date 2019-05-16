import os
from pathlib import Path
from PIL import Image
import numpy as np

import utils.FPHA_utils as DATASET_DIR
from utils.image_utils import *

video_path = os.path.join(FPHA.DIR, "Video_files")
result = list(Path(video_path).rglob("*.[jJ][pP][eE][gG]"))

#make directories
def make_directories(size=416):
    directories = [x[0] for x in os.walk(video_path)]
    for dr in directories:
        new_dr = os.path.join(FPHA.DIR, "Video_files_{size}", dr[53:])
        os.mkdir(new_dr)
        print("MKDIR:", new_dr)

def write_imgs(size=(416,416))
for i, path in enumerate(result):
    img_path = str(path)[53:]
    img_save_path = os.path.join(FPHA.DIR, "Video_files_416", img_path)
    if os.path.isfile(img_save_path):
        print("ALREADY EXISTS:", img_save_path, "%i/%i" %(i, len(result)))
        continue
    img = np.asarray(Image.open(path))
    img = resize_img(img, size)
    img = Image.fromarray(img)
    img.save(img_save_path)
    print("saved", img_save_path, "%i/%i" %(i, len(result)))
    
if __name__ == "__main__":       
    make_directories()
    write_imgs()