import egohand_utils as ego
import hof_utils as hof
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default=None, help='name of dataset used', required=False)
args = parser.parse_args()

PATH_TO_SAVE = '/media/aaron/DATA/ubuntu/hand-seg/%s' %args.dataset

if args.dataset == 'egohand':

    video_id_list = ego.get_video_ids()

    test_idx = [0, 10, 12, 19, 30, 33, 37, 43]
    valid_idx = [6, 13, 27, 46]
    train_idx = [1, 2, 3, 4, 5, 7, 8, 9, 11, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 28, 29, 31, 32, 34, 35, 36, 38, 39, 40, 41, 42, 44, 45, 47]

    def write_split(split_list, save_path):
        print('Writing data split into %s' %save_path)
        with open(save_path, 'w') as f:
            for idx in split_list:
                f.write(video_id_list[idx] + '\n')

    write_split(test_idx, os.path.join(PATH_TO_SAVE, 'data_test.txt'))
    write_split(valid_idx, os.path.join(PATH_TO_SAVE, 'data_val.txt'))
    write_split(train_idx, os.path.join(PATH_TO_SAVE, 'data_train.txt'))
    ego.save_mask_data_split('train')
    ego.save_mask_data_split('test')
    ego.save_mask_data_split('val')

if args.dataset == 'hand_over_face':

    idx_range = list(range(0+1, 302+1))
    idx_range.remove(166)
    idx_range.remove(133)
    test_idx = [1, 6, 8, 9, 13, 27, 34, 38, 44 ,50, 55, 63, 64, 68, 69, 71, 81, 98, 104, 107, 121, 122, 125, 127, 128, 129, 131, 134, 137, 140, 141, 142, 156, 159, 169, 170, 174, 177, 180, 186, 195, 197, 201, 202, 205, 207, 212, 223, 225, 228, 231, 234, 239, 246, 258, 273, 274, 278, 280, 283, 285, 299]
    train_idx = [x for x in idx_range if x not in test_idx]
    valid_idx = train_idx[0:20]
    train_idx = train_idx[20::]

    def write_split(split_list, save_path):
        print('Writing data split into %s' %save_path)
        with open(save_path, 'w') as f:
            for idx in split_list:
                f.write(str(idx) + '\n')

    write_split(test_idx, os.path.join(PATH_TO_SAVE, 'data_test.txt'))
    write_split(valid_idx, os.path.join(PATH_TO_SAVE, 'data_val.txt'))
    write_split(train_idx, os.path.join(PATH_TO_SAVE, 'data_train.txt'))
    hof.save_mask_data_split('train')
    hof.save_mask_data_split('test')
    hof.save_mask_data_split('val')
