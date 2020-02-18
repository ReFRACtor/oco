import numpy as np

import refractor.factory.creator as creator
from refractor import framework as rf

from oco import Level1bOco, OcoMetFile, OcoSoundingId, OcoNoiseModel

from base_config import base_config_definition

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
        return hdf_obj.read_double_3d("/InstrumentHeader/bad_sample_list")[:, sounding_idx, :]
    else:
        # Else try and get bad sample list from snr_coef dataset for older versions of the OCO-2 product
        return hdf_obj.read_double_4d("/InstrumentHeader/snr_coef")[:, sounding_idx, :, 2]

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
    config_def['atmosphere']['ground']['lambertian']['value'] = albedo_cont_level

    # BRDF values is a modification of albedo level based on computed BRDF weight
    orig_brdf_params = config_def['atmosphere']['ground']['brdf']['value']
    config_def['atmosphere']['ground']['brdf']['value'] = {
        'creator': creator.ground.BrdfWeightFromContinuum,
        'continuum_albedo': albedo_cont_level,
        'brdf_parameters': orig_brdf_params,
        'brdf_type': config_def['atmosphere']['ground']['brdf']['brdf_type'],
    }
 
    return config_def
