import os
from pathlib import Path
from PIL import Image
import numpy as np
from skimage.transform import resize
from utils.directory import DATASET_DIR

fpha_path = os.path.join(DATASET_DIR, 'First_Person_Action_Benchmark')
video_path = os.path.join(fpha_path, 'Video_files')

result = list(Path(video_path).rglob("*.[jJ][pP][eE][gG]"))

# directories = [x[0] for x in os.walk(video_path)]
# for dr in directories:
	# new_dr = os.path.join(fpha_path, 'Video_files_416', dr[53:])
	# os.mkdir(new_dr)
	# print('made', new_dr)

# for i, path in enumerate(result):
#     img_path = str(path)[53:]
#     img_save_path = os.path.join(fpha_path, 'Video_files_416', img_path)
#     if os.path.isfile(img_save_path):
#         print('exists', img_save_path, '%i/%i' %(i, len(result)))
#         continue
#     img = np.asarray(Image.open(path))
#     img = resize(img, (416, 416), order=3, preserve_range=True).astype('uint8')
#     img = Image.fromarray(img)
#     img.save(img_save_path)
#     print('saved', img_save_path, '%i/%i' %(i, len(result)))

img_path = '/4TB/aaron/First_Person_Action_Benchmark/Video_files/Subject_2/receive_coin/3/color/color_0039.jpeg'
img_save_path = '/4TB/aaron/First_Person_Action_Benchmark/Video_files_416/Subject_2/receive_coin/3/color/color_0039.jpeg'
img = np.asarray(Image.open(img_path))
img = resize(img, (416, 416), order=3, preserve_range=True).astype('uint8')
img = Image.fromarray(img)
img.save(img_save_path)
print('saved', img_save_path)
