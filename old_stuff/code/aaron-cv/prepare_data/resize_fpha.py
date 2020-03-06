import sys
import os
from pathlib import Path
from PIL import Image
import numpy as np

sys.path.append(str(Path(Path(__file__).resolve()).parents[1]))
from src.utils import IMG, DATA_DIR, FPHA

video_path = os.path.join(DATA_DIR, 'First_Person_Action_Benchmark', 
                          "Video_files")
result = list(Path(video_path).rglob("*.[jJ][pP][eE][gG]"))
size = (FPHA.ORI_WIDTH//2, FPHA.ORI_HEIGHT//2)

#make directories
def make_directories():
    directories = [x[0] for x in os.walk(video_path)]
    new_video_dir = os.path.join(DATA_DIR, 'First_Person_Action_Benchmark', 
                              "Video_files_rsz")
    for dr in directories:
        new_dr = os.path.join(new_video_dir, dr[(43+len(DATA_DIR)):])
        
        os.mkdir(new_dr)
        print("MKDIR:", new_dr)

def write_imgs():
    for i, path in enumerate(result):
        img_path = str(path)[(43+len(DATA_DIR)):]
        img_save_path = os.path.join(DATA_DIR, 'First_Person_Action_Benchmark', 
                                     "Video_files_rsz", img_path)
        if os.path.isfile(img_save_path):
            print("ALREADY EXISTS:", img_save_path, "%i/%i" %(i, len(result)))
            continue
        img = np.asarray(Image.open(path))
        img = IMG.resize_img(img, size)
        img = Image.fromarray(img)
        img.save(img_save_path)
        print("saved", img_save_path, "%i/%i" %(i, len(result)))
    
if __name__ == "__main__":
    # make_directories()
    write_imgs()