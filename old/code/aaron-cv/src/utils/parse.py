def parse_data_cfg(path):
    """ Parse the cfg file and returns data """
    with open(path, 'r') as f:
        lines = f.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    # Get rid of fringe whitespaces
    lines = [x.rstrip().lstrip() for x in lines]  
    data_cfg = {}
    for line in lines:
        key, value = line.split("=")
        value = value.strip()
        value = None if value.lower() == 'false' else value
        data_cfg[key.rstrip()] = value
    return data_cfg

def parse_model_cfg(path):
    """ Parse the cfg file and returns module definitions """
    with open(path, 'r') as f:
        lines = f.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    # Get rid of fringe whitespaces
    lines = [x.rstrip().lstrip() for x in lines]  
    module_defs = []
    for line in lines:
        # This marks the start of a new block
        if line.startswith('['):  
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()
    return module_defs