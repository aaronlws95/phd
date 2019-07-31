from pathlib import Path
with open(Path(__file__).absolute().parents[1]/'data/root.txt') as f:
    ROOT = f.readlines()[0]
    
def parse(path):
    """ Parse the cfg file """
    with open(path, 'r') as f:
        lines = f.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    # Get rid of whitespaces
    lines = [x.rstrip().lstrip() for x in lines]
    data_cfg = {}
    for line in lines:
        key, value = line.split('=')
        value = value.strip()
        value = None if value.lower() == 'false' else value
        data_cfg[key.rstrip()] = value
    return data_cfg