import json

def parse(conf_file):
    with open(conf_file, 'r') as f:
        json_str = f.read()
        conf = json.loads(json_str)
    return conf