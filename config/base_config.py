import os
import h5py
from enum import Enum

import numpy as np

import refractor.factory.creator as creator
import refractor.factory.param as param
from refractor import framework as rf

absco_base_path = os.environ['ABSCO_PATH']

config_dir = os.path.dirname(__file__)

static_input_file = os.path.join(config_dir, "static_input.h5")

solar_file = os.path.join(config_dir, "oco_solar_model.h5")
eof_file = os.path.join(config_dir, "oco_eof.h5")

aerosol_prop_file = os.path.join(os.environ["REFRACTOR_INPUTS"], "l2_aerosol_combined.h5")
reference_atm_file =  os.path.join(os.environ["REFRACTOR_INPUTS"], "reference_atmosphere.h5")
covariance_file = os.path.join(config_dir, "retrieval_covariance.h5")

# OCO has 3 channels (bands)
# O2A, WCO2, SCO2
num_channels = 3

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

# Common configuration defintion shared amonst retrieval and simulation types of configuration
def base_config_definition(absco_type=AbscoType.Legacy, **kwargs):

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
            'num_channels': num_channels,
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
            'solar_zenith': None,
            'solar_azimuth': None,
            'observation_zenith': None,
            'observation_azimuth': None,
            'relative_azimuth': None,
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
            'creator': creator.instrument.IlsGratingInstrument,
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
                'spectral_variable': [ np.arange(1, static_value("Instrument/Dispersion/number_pixel")[0]),
                                       np.arange(1, static_value("Instrument/Dispersion/number_pixel")[1]),
                                       np.arange(1, static_value("Instrument/Dispersion/number_pixel")[1]), ],
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
                    'uncertainty': creator.l1b.UncertaintyFromL1b,
                    'scale_to_stddev': 1e19,
                    'eof_file': eof_file,
                    'hdf_group': "Instrument/EmpiricalOrthogonalFunction/Glint",
                },
                'eof_2': {
                    'creator': creator.instrument.EmpiricalOrthogonalFunction,
                    'value': np.array([0, 0, 0], dtype=float),
                    'order': 2,
                    'scale_uncertainty': True,
                    'uncertainty': creator.l1b.UncertaintyFromL1b,
                    'scale_to_stddev': 1e19,
                    'eof_file': eof_file,
                    'hdf_group': "Instrument/EmpiricalOrthogonalFunction/Glint",
                },
                'eof_3': {
                    'creator': creator.instrument.EmpiricalOrthogonalFunction,
                    'value': np.array([0, 0, 0], dtype=float),
                    'order': 3,
                    'scale_uncertainty': True,
                    'uncertainty': creator.l1b.UncertaintyFromL1b,
                    'scale_to_stddev': 1e19,
                    'eof_file': eof_file,
                    'hdf_group': "Instrument/EmpiricalOrthogonalFunction/Glint",
                },
                'eof_4': {
                    'creator': creator.instrument.EmpiricalOrthogonalFunction,
                    'value': np.array([0, 0, 0], dtype=float),
                    'order': 4,
                    'scale_uncertainty': True,
                    'uncertainty': creator.l1b.UncertaintyFromL1b,
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
            'rayleigh': {
                'creator': creator.rayleigh.RayleighYoung,
            },
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
                    'value': np.full((num_channels, 1), 1.0),
                },
                'brdf': {
                    'creator': creator.ground.GroundBrdf,
                    'value': static_value("/Ground/Brdf/a_priori"),
                    'brdf_type': creator.ground.BrdfTypeOption.soil,
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
                    'cov_unit': rf.Unit("ph / s / m^2 / micron sr^-1"),
                    'which_channels': np.array([0]),
                  },
            },
        },
        'retrieval': {
            'creator': creator.retrieval.NLLSRetrieval,
            'retrieval_components': {
                'creator': creator.retrieval.SVObserverComponents,
                'exclude': ['absorber_levels/linear/O2', 'instrument_doppler'],
                # Match order tradtionally used in old system
                'order': ['CO2', 'H2O', 'surface_pressure', 'temperature_offset', 'aerosol_extinction/gaussian_log', 'ground', 'dispersion'],
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
            'solver_new': {
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
            'solver': {
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
