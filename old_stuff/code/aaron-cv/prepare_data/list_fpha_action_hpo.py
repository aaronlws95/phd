import sys
import numpy as np
from tqdm import tqdm
from pathlib import Path

sys.path.append(str(Path(Path(__file__).resolve()).parents[1]))
from src.utils import FPHA, DATA_DIR

with open(Path(DATA_DIR)/'First_Person_Action_Benchmark'/'bad_imgs.txt', 'r') as f:
    bad_imgs_list = f.readlines()
bad_imgs_list = [i.rstrip() for i in bad_imgs_list]

action_dict = {}
verb = []
noun = []
with open(Path(DATA_DIR)/'First_Person_Action_Benchmark/action_object_info.txt', 'r') as f:
        lines = f.readlines()[1:]
        for l in lines:
            l = l.split(' ')
            verb.append(l[1].split('_')[0])
            noun.append(l[2])
            action_dict[int(l[0]) - 1] = (l[1].split('_')[0], l[2])
verb = np.unique(verb)
noun = np.unique(noun)
noun = {k: v for v, k in enumerate(noun)}

def write_data(file_name, xyz_gt, label, split):
    img_f = open(Path(DATA_DIR)/'First_Person_Action_Benchmark'/'{}_fpha_hpo_img.txt'.format(split), 'w')
    xyz_save = []
    for file, xyz, lab in tqdm(zip(file_name, xyz_gt, label)):
        lab = int(lab)
        if file not in bad_imgs_list:
            seq_info = file + ' ' + str(lab) + ' ' + str(noun[action_dict[lab][1]])
            img_f.write(seq_info + '\n')
            xyz_save.append(xyz)
        # else:
        #     print('Omitting bad img')
    np.savetxt(Path(DATA_DIR)/'First_Person_Action_Benchmark'/'{}_fpha_hpo_xyz.txt'.format(split),
               np.reshape(xyz_save, (-1, 63)))
    
if __name__ == "__main__":
    train_file_name, test_file_name, train_xyz_gt, test_xyz_gt, train_label, test_label \
     = FPHA.get_train_test_pairs_with_action('color', 
                                 Path(DATA_DIR)/'First_Person_Action_Benchmark')
    write_data(train_file_name, train_xyz_gt, train_label, 'train')
    write_data(test_file_name, test_xyz_gt, test_label, 'test')
