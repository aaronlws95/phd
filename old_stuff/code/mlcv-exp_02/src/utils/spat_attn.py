import numpy as np
from pathlib import Path

from src.utils import *
from src.datasets.transforms import *
from src import ROOT

def abs_sum(attn_map):
    abs_attn_map = np.abs(attn_map)
    return np.sum(abs_attn_map, axis=0)

def abs_mean(attn_map):
    abs_attn_map = np.abs(attn_map)
    return np.mean(abs_attn_map, axis=0)

def abs_sum_p(p):
    def _abs_sum_p(attn_map):
        abs_attn_map = np.abs(attn_map)**p
        return np.sum(abs_attn_map, axis=0)
    return _abs_sum_p

def max_sum_p(p):
    def _max_sum_p(attn_map):
        abs_attn_map = np.abs(attn_map)**p
        return np.amax(abs_attn_map, axis=0)
    return _max_sum_p

def visualize_spat_attn_video(model, seq_path, block_id, layer_id, func, seq_name, fps=12, img_size=416, model_info=''):
    from moviepy.editor import ImageSequenceClip
    from tqdm import tqdm
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    from IPython.display import Image as IPythonImage
    from src.datasets.transforms import ImgToNumpy

    seq = [x for x in sorted(seq_path.glob('*')) if x.is_file()]

    frames = []
    for f in tqdm(seq):
        img = get_img_dataloader(str(f), img_size)
        img = img.unsqueeze(0).cuda()
        spatial_attn_map = model.net.get_spatial_attn_map(img)
        proc_attn_map = [[x[0].detach().cpu().numpy() for x in block] for block in spatial_attn_map]
        proc_attn_map = [[func(x) for x in block] for block in proc_attn_map]
        block = proc_attn_map[block_id][layer_id]
        img = ImgToNumpy()(img.cpu())[0]
        block = cv2.resize(block[:, :, np.newaxis], img.shape[:2])

        fig, ax = plt.subplots()
        ax = fig.gca()
        ax.axis('off')
        ax.imshow(block, cmap='jet')
        ax.imshow(img, alpha=0.5)

        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(data)

        plt.close()

    segment_clip = ImageSequenceClip(frames, fps=fps)
    name = str(Path(ROOT)/'mlcv-exp/data/saved'/'{}_{}_{}_{}.gif'.format(seq_name.replace('/', '_'), block_id, layer_id, model_info))
    segment_clip.write_gif(name, fps=fps)

    with open(name, 'rb') as f:
        display(IPythonImage(data=f.read(), format='png'))