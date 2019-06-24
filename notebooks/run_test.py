import pytest
import os
import sys
sys.path.append('..')
os.environ["ABSCO_PATH"] = "/Users/smyth/absco/v5.0.0"
from refractor.factory import process_config
from refractor import framework as rf
from config import oco_config
from pprint import pprint

def test_load_example_config():
    data_dir = os.path.realpath('../test/in')
    l1b_file = os.path.join(data_dir, "oco2_L1bScND_16094a_170711_B7302r_171102090317-selected_ids.h5")
    met_file = os.path.join(data_dir, "oco2_L2MetND_16094a_170711_B8000r_171017214714-selected_ids.h5")

    sounding_id = "2017071110541471"
    config_def = oco_config.retrieval_config_definition(l1b_file, met_file, sounding_id)
    config_inst = process_config(config_def)
    pprint(config_inst, indent=2)
    fm = config_inst.forward_model
    atm = config_inst.atmosphere
    sv = config_inst.retrieval.state_vector
    solver = config_inst.retrieval.solver
    
