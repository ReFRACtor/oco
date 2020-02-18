import os
from refractor.config import refractor_config
from .rtr_comparison_base import rtr_comparison_base_config, initial_guess_values

@refractor_config
def config(**kwargs):
    config_def = rtr_comparison_base_config(**kwargs)

    config_def['atmosphere']['aerosol']['aerosols'] = []

    config_def['retrieval']['initial_guess'] = initial_guess_values(use_aerosols=False)

    return config_def
