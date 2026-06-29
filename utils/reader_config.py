import json
from dotmap import DotMap

def config_reader(path_to_json_conf: str):
    """Load parameters from configuration file.

    Args:
        path_to_json_conf (str): path to the configuration file

    Returns:
        DotMap: variable with parameters
    """   
    with open(path_to_json_conf, 'r') as config_file:
        config_dict = json.load(config_file)
 
    return DotMap(config_dict, _dynamic=False)