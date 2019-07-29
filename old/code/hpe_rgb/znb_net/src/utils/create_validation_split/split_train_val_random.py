import sys
import os
import random

with open('training_list.txt') as f:
    lines = f.readlines()
    random.shuffle(lines)
    lines_val = lines[:120]
    lines_train = lines[120:]

with open('data_split_action_recognition_with_val_random.txt', "w") as f:
        f.write('Validation %i\n' %len(lines_val))
        for lines in lines_val:
            f.write(lines)
        f.write('Training %i\n' %len(lines_train))
        for lines in lines_train:
            f.write(lines)

