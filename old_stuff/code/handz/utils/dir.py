import os

from utils.json_utils import parse

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dirjson = os.path.join(root, 'config' '/dir.json')
dir_dict = parse(dirjson)