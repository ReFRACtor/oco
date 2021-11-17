import numpy as np

from refractor.framework.factory import creator
from refractor import framework as rf

from oco import Level1bOco, OcoMetFile, OcoSoundingId, OcoNoiseModel

from .base_config import base_config_definition

def oco_level1b(hdf_obj, observation_id):
    max_ms = np.array([ 7.00e20, 2.45e20, 1.25e20 ])

    l1b = Level1bOco(hdf_obj, observation_id)

    noise = OcoNoiseModel(hdf_obj, observation_id, max_ms)
    l1b.noise_model = noise

    return l1b

def oco_meteorology(met_file, observation_id):
    return OcoMetFile(met_file, observation_id)

def oco_bad_sample_mask(hdf_obj, observation_id):
    sounding_idx = observation_id.sounding_number 

    # Try to get bad sample list from a dedicated dataset
    if hdf_obj.has_object("/InstrumentHeader/bad_sample_list"):
        return hdf_obj.read_double_3d("/InstrumentHeader/bad_sample_list")[:, sounding_idx, :].astype(bool)
    else:
        # Else try and get bad sample list from snr_coef dataset for older versions of the OCO-2 product
        return hdf_obj.read_double_4d("/InstrumentHeader/snr_coef")[:, sounding_idx, :, 2].astype(bool)

# Return ILS information for the observation sounding position
def ils_delta_lambda(hdf_obj, observation_id):
    sounding_idx = observation_id.sounding_number 
    return hdf_obj.read_double_4d("/InstrumentHeader/ils_delta_lambda")[:, sounding_idx, :, :]

def ils_response(hdf_obj, observation_id):
    sounding_idx = observation_id.sounding_number 
    return hdf_obj.read_double_4d("/InstrumentHeader/ils_relative_response")[:, sounding_idx, :, :]

def retrieval_config_definition(l1b_file, met_file, sounding_id, **kwargs):
    l1b_obj = rf.HdfFile(l1b_file)
    observation_id = OcoSoundingId(l1b_obj, sounding_id)

    config_def = base_config_definition(**kwargs)

    config_def['input'] = {
        'creator': creator.base.SaveToCommon,
        'l1b': oco_level1b(l1b_obj, observation_id),
        'met': oco_meteorology(met_file, observation_id),
        'sounding_number': observation_id.sounding_number,
    }

    config_def['scenario'] = {
        'creator': creator.scenario.ScenarioFromL1b,
    }

    config_def['spec_win']['bad_sample_mask'] = oco_bad_sample_mask(l1b_obj, observation_id)

    # Instrument values
    config_def['instrument']['ils_function']['delta_lambda'] = ils_delta_lambda(l1b_obj, observation_id)
    config_def['instrument']['ils_function']['response'] = ils_response(l1b_obj, observation_id)

    # Fill in values required for computing the dispersion from L1B values
    config_def['instrument']['dispersion']['number_samples'] = {
        'creator': creator.l1b.ValueFromLevel1b,
        'field': 'number_sample',
    }

    config_def['instrument']['dispersion']['spectral_variable'] = {
        'creator': creator.l1b.ValueFromLevel1b,
        'field': 'spectral_variable',
    }

    # Reuse creator set up for determining albedo from the continuum level
    albedo_cont_level = {
        'creator': creator.ground.AlbedoFromSignalLevel,
        'signal_level': {
            'creator': creator.l1b.ValueFromLevel1b,
            'field': "signal",
        },
        'solar_strength': np.array([4.87e21, 2.096e21, 1.15e21]),
        'solar_distance': { 'creator': creator.l1b.SolarDistanceFromL1b },
    } 

    # Lambertian value is directly the albedo level from the continuum
    config_def['atmosphere']['ground']['lambertian']['polynomial_coeffs'] = albedo_cont_level

    # BRDF values is a modification of albedo level based on computed BRDF weight
    orig_brdf_params = config_def['atmosphere']['ground']['brdf']['brdf_parameters']
    config_def['atmosphere']['ground']['brdf']['brdf_parameters'] = {
        'creator': creator.ground.BrdfWeightFromContinuum,
        'continuum_albedo': albedo_cont_level,
        'brdf_parameters': orig_brdf_params,
        'brdf_type': config_def['atmosphere']['ground']['brdf']['brdf_type'],
    }
 
    return config_def
