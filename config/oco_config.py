import os
import h5py
import logging
from enum import Enum

import numpy as np

import refractor.factory.creator as creator
import refractor.factory.param as param
from refractor.factory import process_config
from refractor import framework as rf

from oco import Level1bOco, OcoMetFile, OcoSoundingId, OcoNoiseModel

from .simulation import SimulationFile

absco_base_path = os.environ['ABSCO_PATH']

config_dir = os.path.dirname(__file__)

static_input_file = os.path.join(config_dir, "static_input.h5")

solar_file = os.path.join(config_dir, "oco_solar_model.h5")
eof_file = os.path.join(config_dir, "oco_eof.h5")

aerosol_prop_file = os.path.join(os.environ["REFRACTOR_INPUTS"], "l2_aerosol_combined.h5")
reference_atm_file =  os.path.join(os.environ["REFRACTOR_INPUTS"], "reference_atmosphere.h5")
covariance_file = os.path.join(config_dir, "retrieval_covariance.h5")

class AbscoType(Enum):
    Legacy = 1
    AER = 2

# Helpers to abstract away getting data out of the static input file
def static_value(dataset, dtype=None):
    with h5py.File(static_input_file, "r") as static_input:
        return np.array(static_input[dataset][:], dtype=dtype)

def static_units(dataset):
    with h5py.File(static_input_file, "r") as static_input:
        return static_input[dataset].attrs['Units'][0].decode('UTF8') 

def static_spectral_domain(dataset):
    return rf.SpectralDomain(static_value(dataset), rf.Unit(static_units(dataset)))

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

# Common configuration defintion shared amonst retrieval and simulation types of configuration
def common_config_definition(absco_type=AbscoType.Legacy):

    absorber_legacy = {
        'creator': creator.absorber.AbsorberAbsco,
        'gases': ['CO2', 'H2O', 'O2'],
        'CO2': {
            'creator': creator.absorber.AbsorberGasDefinition,
            'vmr': {
                'creator': creator.absorber.AbsorberVmrLevel,
                'value': {
                    'creator': creator.absorber.GasVmrAprioriMetL1b,
                    'reference_atm_file': reference_atm_file,
                },
            },
            'absorption': {
                'creator': creator.absorber.AbscoLegacy,
                'table_scale': [1.0, 1.0, 1.004],
                'filename': "{absco_base_path}/co2_devi2015_wco2scale-nist_sco2scale-unity.h5",
            },
        },
        'H2O': {
            'creator': creator.absorber.AbsorberGasDefinition,
            'vmr': {
                'creator': creator.absorber.AbsorberVmrMet,
                'value': np.array([1.0]),
            },
            'absorption': {
                'creator': creator.absorber.AbscoLegacy,
                'table_scale': 1.0,
                'filename': "{absco_base_path}/h2o_hitran12.h5",
            },
        },
        'O2': {
            'creator': creator.absorber.AbsorberGasDefinition,
            'vmr': {
                'creator': creator.absorber.AbsorberVmrLevel,
                'value': {
                    'creator': creator.atmosphere.ConstantForAllLevels,
                    'value': static_value("Gas/O2/average_mole_fraction")[0],
                },
                'retrieved': False,
            },
            'absorption': {
                'creator': creator.absorber.AbscoLegacy,
                'table_scale': 1.0,
                'filename': "{absco_base_path}/o2_v151005_cia_mlawer_v151005r1_narrow.h5",
             },
        },
    }

    absorber_aer = {
        'creator': creator.absorber.AbsorberAbsco,
        'gases': ['CO2', 'H2O', 'O2'],
        'CO2': {
            'creator': creator.absorber.AbsorberGasDefinition,
            'vmr': {
                'creator': creator.absorber.AbsorberVmrLevel,
                'value': {
                    'creator': creator.absorber.GasVmrAprioriMetL1b,
                    'reference_atm_file': reference_atm_file,
                },
            },
            'absorption': {
                'creator': creator.absorber.AbscoAer,
                'table_scale': [1.0, 1.0, 1.004],
                'filename': "{absco_base_path}/{gas!u}_04760-06300_v0.0_init.nc",
            },
        },
        'H2O': {
            'creator': creator.absorber.AbsorberGasDefinition,
            'vmr': {
                'creator': creator.absorber.AbsorberVmrMet,
                'value': np.array([1.0]),
            },
            'absorption': {
                'creator': creator.absorber.AbscoAer,
                'table_scale': 1.0,
                'filename': "{absco_base_path}/H2O_04760-13230_v0.0_init.nc",
            },
        },
        'O2': {
            'creator': creator.absorber.AbsorberGasDefinition,
            'vmr': {
                'creator': creator.absorber.AbsorberVmrLevel,
                'value': {
                    'creator': creator.atmosphere.ConstantForAllLevels,
                    'value': static_value("Gas/O2/average_mole_fraction")[0],
                },
                'retrieved': False,
            },
            'absorption': {
                'creator': creator.absorber.AbscoAer,
                'table_scale': 1.0,
                'filename': "{absco_base_path}/O2_06140-13230_v0.0_init.nc",
             },
        },
    }

    config_def = {
        'creator': creator.base.SaveToCommon,
        'order': ['input', 'common', 'scenario', 'spec_win', 'spectrum_sampling', 'instrument', 'atmosphere', 'radiative_transfer', 'forward_model' , 'retrieval'],
        'input': {
            # Filled in by derived config, not required
        },
        'common': {
            'creator': creator.base.SaveToCommon,
            'desc_band_name': static_value("Common/desc_band_name", dtype=str),
            'hdf_band_name': static_value("Common/hdf_band_name", dtype=str),
            'band_reference': {
                'creator': creator.value.ArrayWithUnit,
                'value': static_value("Common/band_reference_point"),
                'units': static_units("Common/band_reference_point"),
            },
            'num_channels': 3,
            'absco_base_path': absco_base_path,
            'constants': {
                'creator': creator.common.DefaultConstants,
            },
        },
        # Wrap the common temporal/spatial values into the scenario block which will
        # be exposed to other creators
        'scenario': {
            # Place holders for the value required, must be filled in by derived config
            'creator': creator.base.SaveToCommon,
            'time': None,
            'latitude': None,
            'longitude': None,
            'surface_height': None,
            'solar_distance': None,
            'solar_zenith': None,
            'solar_azimuth': None,
            'observation_zenith': None,
            'observation_azimuth': None,
            'relative_velocity': None,
            'spectral_coefficient': None,
            'stokes_coefficient': None,
        },
        'spec_win': {
            'creator': creator.forward_model.SpectralWindowRange,
            'window_ranges': {
                'creator': creator.value.ArrayWithUnit,
                'value': static_value("/Spectral_Window/microwindow"),
                'units': static_units("/Spectral_Window/microwindow"),
            },
        },
        'spectrum_sampling': {
            'creator': creator.forward_model.NonuniformSpectrumSampling,
            'high_res_spacing': rf.DoubleWithUnit(0.01, "cm^-1"), 
            'channel_domains': [ static_spectral_domain("/Spectrum_Sampling/nonuniform_grid_1"), 
                                 static_spectral_domain("/Spectrum_Sampling/nonuniform_grid_2"),
                                 static_spectral_domain("/Spectrum_Sampling/nonuniform_grid_3"), ]
        },
        'instrument': {
            'creator': creator.instrument.IlsInstrument,
            'ils_half_width': {
                'creator': creator.value.ArrayWithUnit,
                'value': np.array([4.09e-04, 1.08e-03, 1.40e-03]),
                'units': "um",
            },
            'dispersion': {
                'creator': creator.instrument.DispersionPolynomial,
                'value': {
                    'creator': creator.value.NamedCommonValue,
                    'name': 'spectral_coefficient',
                },
                'number_samples': static_value("Instrument/Dispersion/number_pixel"),
                'is_one_based': True,
                'num_parameters': 2,
            },
            'ils_function': {
                'creator': creator.instrument.IlsTable,
                # These need to be supplied by the derived config definition
                'delta_lambda': None,
                'response': None, 
            },
            'instrument_correction': {
                'creator': creator.instrument.InstrumentCorrectionList,
                'corrections': ['eof_1', 'eof_2', 'eof_3'],
                'eof_1': {
                    'creator': creator.instrument.EmpiricalOrthogonalFunction,
                    'value': np.array([0, 0, 0], dtype=float),
                    'order': 1,
                    'scale_uncertainty': True,
                    'scale_to_stddev': 1e19,
                    'eof_file': eof_file,
                    'hdf_group': "Instrument/EmpiricalOrthogonalFunction/Glint",
                },
                'eof_2': {
                    'creator': creator.instrument.EmpiricalOrthogonalFunction,
                    'value': np.array([0, 0, 0], dtype=float),
                    'order': 2,
                    'scale_uncertainty': True,
                    'scale_to_stddev': 1e19,
                    'eof_file': eof_file,
                    'hdf_group': "Instrument/EmpiricalOrthogonalFunction/Glint",
                },
                'eof_3': {
                    'creator': creator.instrument.EmpiricalOrthogonalFunction,
                    'value': np.array([0, 0, 0], dtype=float),
                    'order': 3,
                    'scale_uncertainty': True,
                    'scale_to_stddev': 1e19,
                    'eof_file': eof_file,
                    'hdf_group': "Instrument/EmpiricalOrthogonalFunction/Glint",
                },
                'eof_4': {
                    'creator': creator.instrument.EmpiricalOrthogonalFunction,
                    'value': np.array([0, 0, 0], dtype=float),
                    'order': 4,
                    'scale_uncertainty': True,
                    'scale_to_stddev': 1e19,
                    'eof_file': eof_file,
                    'hdf_group': "Instrument/EmpiricalOrthogonalFunction/Glint",
                },
            },
        },
        'atmosphere': {
            'creator': creator.atmosphere.AtmosphereCreator,
            'pressure': {
                'creator': creator.atmosphere.PressureSigma,
                'value': {
                    'creator': creator.met.ValueFromMet,
                    'field': "surface_pressure",
                },
                'a_coeff': static_value("Pressure/Pressure_sigma_a"),
                'b_coeff': static_value("Pressure/Pressure_sigma_b"),
            },
            'temperature': {
                'creator': creator.atmosphere.TemperatureMet,
                'value': static_value("Temperature/Offset/a_priori")
            },
            'altitude': { 
                'creator': creator.atmosphere.AltitudeHydrostatic,
            },
            'absorber': None, # Determined by switch to config
            'aerosol': {
                'creator': creator.aerosol.AerosolOptical,
                'aerosols': [ "kahn_2b", "kahn_3b", "water", "ice" ],
                'kahn_2b': {
                    'creator': creator.aerosol.AerosolDefinition,
                    'extinction': {
                        'creator': creator.aerosol.AerosolShapeGaussian,
                        'value': np.array([-4.38203, 1, 0.2]),
                    },
                    'properties': {
                        'creator': creator.aerosol.AerosolPropertyHdf,
                        'filename': aerosol_prop_file,
                    },
                },
                'kahn_3b': {
                    'creator': creator.aerosol.AerosolDefinition,
                    'extinction': {
                        'creator': creator.aerosol.AerosolShapeGaussian,
                        'value': np.array([-4.38203, 1, 0.2]),
                    },
                    'properties': {
                        'creator': creator.aerosol.AerosolPropertyHdf,
                        'filename': aerosol_prop_file,
                    },
                },
                'water': {
                    'creator': creator.aerosol.AerosolDefinition,
                    'extinction': {
                        'creator': creator.aerosol.AerosolShapeGaussian,
                        'value': np.array([-4.38203, 0.75, 0.1]),
                    },
                    'properties': {
                        'creator': creator.aerosol.AerosolPropertyHdf,
                        'filename': aerosol_prop_file,
                        'prop_name': "wc_008",
                    },
                },
                'ice': {
                    'creator': creator.aerosol.AerosolDefinition,
                    'extinction': {
                        'creator': creator.aerosol.AerosolShapeGaussian,
                        'value': np.array([-4.38203, 0.3, 0.04]),
                    },
                    'properties': {
                        'creator': creator.aerosol.AerosolPropertyHdf,
                        'filename': aerosol_prop_file,
                        'prop_name': "ice_cloud_MODIS6_deltaM_1000",
                    },
                },

            },
            'relative_humidity': {
                'creator': creator.atmosphere.RelativeHumidity,
            },
            'ground': {
                'creator': creator.base.PickChild,
                'child': 'lambertian',
                'lambertian': {
                    'creator': creator.ground.GroundLambertian,
                    # Filled in by derived creator
                    'value': None,
                },
            },
        },
        'radiative_transfer': {
            'creator': creator.rt.LsiRt,
            'num_low_streams': 1,
            'num_high_streams': 8,
            'lsi_config_file': static_input_file,
        },
        'forward_model': {
            'creator': creator.forward_model.ForwardModel,
            'spectrum_effect': {
                'creator': creator.forward_model.SpectrumEffectList,
                'effects': ["solar_model", "instrument_doppler", "fluorescence_effect"],
                'solar_model': {
                    'creator': creator.solar_model.SolarAbsorptionAndContinuum,
                    'doppler': {
                        'creator': creator.solar_model.SolarDopplerShiftPolynomial,
                    },
                    'absorption': {
                        'creator': creator.solar_model.SolarAbsorptionTable,
                        'solar_data_file': solar_file,
                    },
                    'continuum': {
                        'creator': creator.solar_model.SolarContinuumTable,
                        'solar_data_file': solar_file,
                    },
                },
                'instrument_doppler': {
                    'creator': creator.instrument.InstrumentDoppler,
                    'value': {
                        'creator': creator.value.NamedCommonValue,
                        'name': 'relative_velocity',
                    },
                },
                'fluorescence_effect':{
                    'creator': creator.forward_model.FluorescenceEffect,
                    'reference_point': {
                        'creator': creator.value.ArrayWithUnit,
                        'value': np.array([0.757]),
                        'units':'micron',
                    },
                    'value': np.array([0.0, 0.0018]),
                    'cov_unit': rf.Unit("W/cm^2/sr/cm^-1"),
                    'spec_index': np.array([0]),
                  },
            },
        },
        'retrieval': {
            'creator': creator.retrieval.NLLSRetrieval,
            'retrieval_components': {
                'creator': creator.retrieval.SVObserverComponents,
                'exclude': ['absorber_levels/O2', 'instrument_doppler'],
                # Match order tradtionally used in old system
                'order': ['CO2', 'H2O', 'surface_pressure', 'temperature_offset', 'aerosol_shape', 'ground', 'dispersion'],
            },
            'state_vector': {
                'creator': creator.retrieval.StateVector,
            },
            'initial_guess': {
                'creator': creator.retrieval.InitialGuessFromSV,
            },
            'a_priori': {
                'creator': creator.retrieval.AprioriFromIG,
            },
            'covariance': {
                'creator': creator.retrieval.CovarianceByComponent,
                'values': {
                    'creator': creator.value.LoadValuesFromHDF,
                    'filename': covariance_file,
                }
            },
            'solver': {
                'creator': creator.retrieval.NLLSSolverLM,
                'max_iteration': 5,
            },
            'solver_nlls_gsl': {
                'creator': creator.retrieval.NLLSSolverGSLLMSDER,
                'max_cost_function_calls': 10,
                'dx_tol_abs': 1e-5,
                'dx_tol_rel': 1e-5, 
                'g_tol_abs': 1e-5,
            },
            'solver_connor': {
                'creator': creator.retrieval.ConnorSolverMAP,
                'max_cost_function_calls': 14,
                'threshold': 2.0,
                'max_iteration': 7,
                'max_divergence': 2,
                'max_chisq': 1.4,
                'gamma_initial': 10.0,
            },
        },
    }

    if absco_type == AbscoType.Legacy:
        config_def['atmosphere']['absorber'] = absorber_legacy
    elif absco_type == AbscoType.AER:
        config_def['atmosphere']['absorber'] = absorber_aer
    else:
        raise param.ParamError("Invalid absco type: {}".format(absco_type))

    return config_def

def retrieval_config_definition(l1b_file, met_file, sounding_id, **kwargs):
    l1b_obj = rf.HdfFile(l1b_file)
    observation_id = OcoSoundingId(l1b_obj, sounding_id)

    config_def = common_config_definition(**kwargs)

    config_def['input'] = {
        'creator': creator.base.SaveToCommon,
        'l1b': oco_level1b(l1b_obj, observation_id),
        'met': oco_meteorology(met_file, observation_id),
        'sounding_number': observation_id.sounding_number,
    }

    # Set up scenario values that have the same name as they are named in the L1B
    l1b_value_names = ['time', 'latitude', 'longitude', 'solar_zenith', 'solar_azimuth', 
        "relative_velocity", "spectral_coefficient", "stokes_coefficient"] 
    values_from_l1b = { n:n for n in l1b_value_names }

    # Set up scenario values with different names
    values_from_l1b.update({
        "surface_height": "altitude", 
        "observation_zenith": "sounding_zenith"
    })

    for config_name, l1b_name in values_from_l1b.items():
        config_def['scenario'][config_name] = {
            'creator': creator.l1b.ValueFromLevel1b,
            'field': l1b_name,
        }
    
    # Set up scenario values with creators
    config_def['scenario']['solar_distance'] = {
        'creator': creator.l1b.SolarDistanceFromL1b,
    }

    config_def['scenario']['observation_azimuth'] = {
        'creator': creator.l1b.RelativeAzimuthFromLevel1b,
    }

    config_def['spec_win']['bad_sample_mask'] = oco_bad_sample_mask(l1b_obj, observation_id)

    # Instrument values
    config_def['instrument']['ils_function']['delta_lambda'] = ils_delta_lambda(l1b_obj, observation_id)
    config_def['instrument']['ils_function']['response'] = ils_response(l1b_obj, observation_id)

    # Ground albedo from radiance continuum level
    config_def['atmosphere']['ground']['lambertian']['value'] = {
        'creator': creator.ground.AlbedoFromSignalLevel,
        'signal_level': {
            'creator': creator.l1b.ValueFromLevel1b,
            'field': "signal",
        },
        'solar_strength': np.array([4.87e21, 2.096e21, 1.15e21]),
    } 
 
    return config_def

def simulation_config_definition(sim_file, sim_index, **kwargs):

    config_def = common_config_definition(**kwargs)

    # Load simulation file
    sim_data = SimulationFile(sim_file, sim_index)

    # Load scenario values
    for val_name in config_def['scenario'].keys():
        if val_name != "creator":
            config_def['scenario'][val_name] = getattr(sim_data.scenario, val_name)

    # Instrument values
    config_def['instrument']['ils_function']['delta_lambda'] = sim_data.instrument.ils_delta_lambda
    config_def['instrument']['ils_function']['response'] = sim_data.instrument.ils_response

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

    # Ground lambertian
    config_def['atmosphere']['ground']['lambertian']['value'] = sim_data.ground.lambertian_albedo
 
    # Require only the state vector to be set up, this gives us jacobians 
    # without worrying about satisfying solver requirements
    config_def['retrieval']['creator'] = creator.retrieval.RetrievalBaseCreator


    return config_def

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    data_dir = os.path.realpath(os.path.join(config_dir, '../test/in'))
    l1b_file = os.path.join(data_dir, "oco2_L1bScND_16094a_170711_B7302r_171102090317-selected_ids.h5")
    met_file = os.path.join(data_dir, "oco2_L2MetND_16094a_170711_B8000r_171017214714-selected_ids.h5")

    sounding_id = "2017071110541471"

    config_def = retrieval_config_definition(l1b_file, met_file, sounding_id)
    config_inst = process_config(config_def)

    from pprint import pprint
    pprint(config_inst, indent=4)
