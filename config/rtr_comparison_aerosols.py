import os

import numpy as np

from refractor.config import refractor_config
from refractor.factory import creator, param
from refractor import framework as rf

from .rtr_comparison_base import rtr_comparison_base_config, initial_guess_values
from .base_config import aerosol_prop_file

@refractor_config
def config(**kwargs):
    config_def = rtr_comparison_base_config(**kwargs)

    # Use aerosol types as done in expected test data
    aerosol_setup = { 
        "SO": "SO" ,
        "DU": "DU",
        "Ice": "ice_cloud_MODIS6_deltaM_1000",
        "Water": "wc_008",
        "ST": "strat",
    }
    config_def['atmosphere']['aerosol'] = {
        'creator': creator.aerosol.AerosolOptical,
        'aerosols': ["SO", "DU", "Ice", "Water", "ST"],
    }

    for aer_name, prop_name in aerosol_setup.items():
        config_def['atmosphere']['aerosol'][aer_name] = {
            'creator': creator.aerosol.AerosolDefinition,
            'extinction': {
                'creator': creator.aerosol.AerosolShapeGaussian,
                # Initialize when reading expected SV
                'shape_params': np.array([0, 0, 0]),
            },
            'properties': {
                'creator': creator.aerosol.AerosolPropertyHdf,
                'filename': aerosol_prop_file,
                'prop_name': prop_name,
            },
        }

    # Add covariances for all new aerosol types
    class AddAerosolCov(creator.value.LoadValuesFromHDF):
        met = param.InstanceOf(rf.Meteorology)
        
        def create(self, **kwargs):
            cov_dict = super().create(**kwargs)
            
            simple_cov = np.zeros((3,3))
            np.fill_diagonal(simple_cov, 1.0)
            
            for aer_name in config_def['atmosphere']['aerosol']['aerosols']:
                cov_dict[f'aerosol_extinction/gaussian_log/{aer_name}'] = simple_cov
            
            return cov_dict

    config_def['retrieval']['covariance']['values']['creator'] = AddAerosolCov

    config_def['retrieval']['initial_guess'] = initial_guess_values(use_aerosols=True)

    return config_def
