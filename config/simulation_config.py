import re

import numpy as np

import refractor.factory.creator as creator
from refractor import framework as rf
from refractor.config import refractor_config

from base_config import base_config_definition, aerosol_prop_file

from simulation_file import SimulationFile

@refractor_config
def simulation_config_definition(sim_file, sim_index, channel_index=None, **kwargs):

    config_def = base_config_definition(**kwargs)

    # Load simulation file
    sim_data = SimulationFile(sim_file, sim_index)

    # Load scenario values
    for val_name in config_def['scenario'].keys():
        if val_name != "creator":
            config_def['scenario'][val_name] = getattr(sim_data.scenario, val_name)

    config_def['scenario']['sounding_number'] = int(str(sim_data.scenario.observation_id[...])[-1])

    # Instrument values
    config_def['instrument']['ils_function']['delta_lambda'] = sim_data.instrument.ils_delta_lambda
    config_def['instrument']['ils_function']['response'] = sim_data.instrument.ils_response

    # Channels
    if channel_index is not None:
        # Zero out other channels
        win_ranges = config_def['spec_win']['window_ranges']['value']
        for c_i in range(win_ranges.shape[0]):
            if c_i != channel_index:
                win_ranges[c_i, :, :] = 0

    # Empirical Orthogonal Functions
    rad_uncert = []
    for chan_idx in range(sim_data.instrument.radiance_uncertainty.rows):
        rad_uncert.append(sim_data.instrument.radiance_uncertainty[chan_idx, :])

    ic_config = config_def['instrument']['instrument_correction']
    for ic_name in ic_config['corrections']:
        if re.search('eof_', ic_name):
            # Set EOF scaling term
            ic_config[ic_name]['value'] = getattr(sim_data.instrument, ic_name)

            # Set uncertainty
            ic_config[ic_name]['uncertainty'] = rad_uncert

    # Atmosphere
    config_def['atmosphere']['pressure'] = {
        'creator': creator.atmosphere.PressureGrid,
        'pressure_levels': sim_data.atmosphere.pressure_levels,
        'value': sim_data.atmosphere.surface_pressure,
    }

    config_def['atmosphere']['temperature'] = {
        'creator': creator.atmosphere.TemperatureLevelOffset,
        'temperature_levels': sim_data.atmosphere.temperature,
        'value': np.array([0.0]),
    }

    # Absorber values
    config_def['atmosphere']['gases'] = sim_data.absorber.molecule_names

    for name in sim_data.absorber.molecule_names:
        config_def['atmosphere']['absorber'][name]["vmr"] = {
            'creator': creator.absorber.AbsorberVmrLevel,
            'value': sim_data.absorber.vmr(name),
        }

    # Aerosol values, reset to start with an empty aerosol configuration
    config_def['atmosphere']['aerosol'] = {
        'creator': creator.aerosol.AerosolOptical,
        'aerosols': sim_data.aerosol.particle_names,
    }

    for name in sim_data.aerosol.particle_names:
        config_def['atmosphere']['aerosol'][name] = {
            'creator': creator.aerosol.AerosolDefinition,
            'extinction': {
                'creator': creator.aerosol.AerosolShapeGaussian,
                'value': sim_data.aerosol.gaussian_param(name),
            },
            'properties': {
                'creator': creator.aerosol.AerosolPropertyHdf,
                'filename': aerosol_prop_file,
                'prop_name': sim_data.aerosol.property_name(name),
            },
        }

    # Ground
    if sim_data.ground.type == "lambertian":
        config_def['atmosphere']['ground']['child'] = 'lambertian'
        config_def['atmosphere']['ground']['lambertian']['value'] = sim_data.ground.lambertian_albedo
    elif sim_data.ground.type == "brdf":
        config_def['atmosphere']['ground']['child'] = 'brdf'
        config_def['atmosphere']['ground']['brdf']['value'] = sim_data.ground.brdf_parameters
    else:
        raise param.ParamError("Could not determine ground type")

    # Fluorescence
    spec_eff_config = config_def['forward_model']['spectrum_effect']
    if 'fluorescence_effect' in spec_eff_config:
        spec_eff_config['fluorescence_effect']['value'] = sim_data.atmosphere.fluorescence[:]

    # Require only the state vector to be set up, this gives us jacobians 
    # without worrying about satisfying solver requirements
    config_def['retrieval']['creator'] = creator.retrieval.RetrievalBaseCreator

    return config_def
