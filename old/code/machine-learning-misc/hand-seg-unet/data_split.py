import read_meta as rm
import os

root_path = '/media/aaron/DATA/ubuntu/egohands_data'
video_path = os.path.join(root_path, '_LABELLED_SAMPLES')
data_path = os.path.join(root_path, 'data')

video_id_list = rm.get_video_ids()

test_idx = [0, 10, 12, 19, 30, 33, 37, 43]
valid_idx = [6, 13, 27, 46]
train_idx = [1, 2, 3, 4, 5, 7, 8, 9, 11, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 28, 29, 31, 32, 34, 35, 36, 38, 39, 40, 41, 42, 44, 45, 47]

def write_split(split_list, save_path):
    split = []
    for idx in split_list:
        split.append(video_id_list[idx])

    with open(save_path, 'w') as f:
        for video_id in split:
            f.write(video_id + '\n')

write_split(test_idx, os.path.join(data_path, 'data_test.txt'))
write_split(valid_idx, os.path.join(data_path, 'data_val.txt'))
write_split(train_idx, os.path.join(data_path, 'data_train.txt'))
