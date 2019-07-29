import os
import numpy as np
from collections import Counter

root_path = '/media/aaron/DATA/ubuntu/fpa-benchmark/'
video_path = os.path.join(root_path, 'Video_files')
data_path = os.path.join(root_path, 'data')

actions = []
idx = []
with open(os.path.join(data_path, 'actions.txt')) as f:
    lines = f.readlines()
    for line in lines:
        words = line.split()
        actions.append(words[0])
        idx.append(words[1])

action_dict = dict(zip(actions, idx))

def get_data_dir():
    print('Getting data_dir...')
    directories = []
    subject_action_count = []
    for subject in os.listdir(video_path):
        subject_path = os.path.join(video_path, subject)
        for action in os.listdir(subject_path):
            action_path = os.path.join(subject_path, action)
            for number in os.listdir(action_path):
                number_path = os.path.join(action_path, number)
                dir_to_save = os.path.join(subject, action)
                dir_to_save = os.path.join(dir_to_save, number)
                subject_action_count.append((subject + ' ' + action_dict[action], action_dict[action]))
                directories.append((dir_to_save, action_dict[action]))

    with open(os.path.join(data_path, 'data_dir.txt'), 'w') as f:
        for data_dir, idx in directories:
            f.write(data_dir + ' ' + idx + '\n')

    # counts = Counter(x[0] for x in subject_action_count)
    # print(counts)

def get_data_custom(name, number_list, sort=False):
    print('Getting', name, '...')

    def getKey(item):
        return int(item[1])

    directories = []
    for subject in os.listdir(video_path):
        subject_path = os.path.join(video_path, subject)
        for action in os.listdir(subject_path):
            action_path = os.path.join(subject_path, action)
            for number in os.listdir(action_path):
                if(int(number) in number_list):
                    number_path = os.path.join(action_path, number)
                    dir_to_save = os.path.join(subject, action)
                    dir_to_save = os.path.join(dir_to_save, number)
                    directories.append((dir_to_save, action_dict[action]))
    if sort:
        directories = sorted(directories, key=getKey)
    with open(os.path.join(data_path, name), 'w') as f:
        for data_dir, idx in directories:
            f.write(data_dir + ' ' + idx + '\n')

get_data_dir()
get_data_custom('data_train.txt', [2, 4])
get_data_custom('data_test.txt', [3, 5, 6 , 7, 8, 9])
get_data_custom('data_val.txt', [1])
