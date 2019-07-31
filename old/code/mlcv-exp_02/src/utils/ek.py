import numpy as np
import pandas as pd
from pathlib import Path

from src import ROOT

def _rev_lin_id(unique_ids, type):
    """ Get dict to find actual id """
    lin_id = dict()
    for i, id in enumerate(np.sort(unique_ids)):
        lin_id[i] = get_class_name(id, type)
    return lin_id

def get_class_name(id, type):
    epic_root       = Path(ROOT)/'EPIC_KITCHENS_2018'
    csv_file        = 'EPIC_{}_classes.csv'.format(type)
    csv_file        = epic_root/'annotations'/csv_file
    classes         = pd.read_csv(csv_file)
    name            = classes.loc[classes['{}_id'.format(type)] == id].to_numpy()[0, 1]
    return name

def get_verb_dict():
    epic_root = Path(ROOT)/'EPIC_KITCHENS_2018'
    epic_data = str(epic_root/'annotations'/'EPIC_train_action_labels.csv')
    epic_data = pd.read_csv(epic_data)
    verb_class = np.sort(epic_data.verb_class.unique())
    verb_dict = _rev_lin_id(verb_class, 'verb')
    return verb_dict

def get_noun_dict():
    epic_root = Path(ROOT)/'EPIC_KITCHENS_2018'
    epic_data = str(epic_root/'annotations'/'EPIC_train_action_labels.csv')
    epic_data = pd.read_csv(epic_data)
    noun_class = np.sort(epic_data.noun_class.unique())
    noun_dict = _rev_lin_id(noun_class, 'noun')
    return noun_dict

