import os
from refractor.config import refractor_config
from .retrieval_config import retrieval_config_definition


@refractor_config
def config(**kwargs):
    test_in_dir = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) +"/../test/in")
    l1b_file = os.path.join(test_in_dir, "oco2_L1bScND_16094a_170711_B7302r_171102090317-selected_ids.h5")
    met_file = os.path.join(test_in_dir, "oco2_L2MetND_16094a_170711_B8000r_171017214714-selected_ids.h5")
    sounding_id = "2017071110541471"
    config_def = retrieval_config_definition(l1b_file, met_file, sounding_id)
    return config_def





