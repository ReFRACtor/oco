#!/usr/bin/env python

import os
import re
import sys
import logging
from enum import Enum
from collections import OrderedDict
from itertools import zip_longest

import h5py
import netCDF4
import numpy as np

# Find where the code repository is located relative to this file
oco_repo_path = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))

# Add the path to the configuration so it can be imported
sys.path.append(os.path.join(oco_repo_path))

# Import ReFRACtor framework
from refractor.framework.factory import process_config
from refractor import framework as rf

# Import configuration module
from config.retrieval_config import retrieval_config_definition

logger = logging.getLogger()

class GroundType(Enum):
    lambertian = "lambertian"
    lambertian_piecewise = "lambertian_piecewise"
    coxmunk = "coxmunk"
    coxmunk_lambertian = "coxmunk_lambertian"
    brdf = "brdf"

class SimulationWriter(object): 

    def __init__(self, l1b_file, met_file, sounding_id_list, max_name_len=25, albedo_degree=4, diag_file=None, ground_type=GroundType.lambertian, enable_cloud_3d=False):
        
        logging.debug("Creating simulation file using L1B: %s, Met: %s" % (l1b_file, met_file))

        self.l1b_file = l1b_file
        self.met_file = met_file
        self.sounding_id_list = sounding_id_list

        if diag_file is not None:
            logger.debug(f"Loading L2 diagnostic file for converged results: {diag_file}")
            self.diag_file = h5py.File(diag_file, "r")
            self.diag_sounding_ids = self.diag_file['/RetrievalHeader/sounding_id'][:].astype(str)
            logger.debug(f"Loaded {len(self.diag_sounding_ids)} sounding ids from diagnostic file")
        else:
            self.diag_file = None

        self.max_name_len = 80
        self.albedo_degree = albedo_degree

        logger.debug("Using ground type: {}".format(ground_type))
        self.ground_type = GroundType(ground_type)

        self.enable_cloud_3d = enable_cloud_3d

    def config(self, sounding_id):

        logging.debug("Loading configuration for sounding: %s" % sounding_id)

        config_def = retrieval_config_definition(self.l1b_file, self.met_file, sounding_id)

        config_def['atmosphere']['ground']['child'] = self.ground_type.value

        config_inst = process_config(config_def)
        config_inst.config_def = config_def

        return config_inst

    def _create_dims(self, output_file):

        logging.debug("Setting up file dimensions")

        # Find max of values to be used in dimensioning
        max_level = 0
        max_gas = 0
        max_aer = 0
        num_channel = 0
        num_spec_coeff = 0
        num_stokes_coeff = 0
        num_samples = 0
        num_ils_values = 0

        for sid in self.sounding_id_list:
            snd_config = self.config(sid)

            atm = snd_config.atmosphere
            max_level = max(max_level, atm.pressure.number_level)
            max_gas = max(max_gas, atm.absorber.number_species)
            if atm.aerosol:
                max_aer = max(max_aer, atm.aerosol.number_particle)

            inst = snd_config.instrument
            num_channel = inst.number_spectrometer
            num_samples = inst.ils(0).ils_function.response.shape[0]
            num_ils_values = inst.ils(0).ils_function.response.shape[1]

            l1b = snd_config.input.l1b
            num_spec_coeff = l1b.spectral_coefficient(0).value.shape[0]
            num_stokes_coeff = l1b.stokes_coefficient(0).shape[0]

        # Create dimensions for the maximum number of 
        self.snd_id_dim = output_file.createDimension('n_sounding', len(self.sounding_id_list))
        self.channel_dim = output_file.createDimension('n_channel', num_channel)
        self.lev_dim = output_file.createDimension('n_level', max_level)
        self.gas_dim = output_file.createDimension('n_absorber', max_gas)
        self.aer_dim = output_file.createDimension('n_aerosol', max_aer)

        self.spec_coeff_dim = output_file.createDimension('n_spectral_coefficient', num_spec_coeff)
        self.stokes_coeff_dim = output_file.createDimension('n_stokes_coefficient', num_stokes_coeff)

        self.samp_dim = output_file.createDimension('n_samples', num_samples)
        self.ils_dim = output_file.createDimension('n_ils', num_ils_values)
        self.index_dim = output_file.createDimension('n_indexes', None)

        # Length of names of gas and aerosols
        self.name_len = output_file.createDimension('name_length', self.max_name_len)

        # Ground dimensions conditional on the ground type used
        if self.ground_type == GroundType.lambertian:
            self.albedo_poly_dim = output_file.createDimension('n_albedo_poly', self.albedo_degree + 1)
        elif self.ground_type == GroundType.lambertian_piecewise:
            self.albedo_piecewise_dim = output_file.createDimension('n_albedo_grid', None)
        elif self.ground_type == GroundType.brdf:
            # Weight offset, slope + 5 BRDF kernel params
            self.brdf_params_dim = output_file.createDimension('n_brdf_params', 7)
        elif self.ground_type == GroundType.coxmunk_lambertian:
            # 2 params for each channel
            self.coxmunk_albedo_dim = output_file.createDimension('n_coxmunk_albedo', 2)

        # Number of aerosol parameters
        self.aer_param_dim = output_file.createDimension('n_aerosol_parameters', 3)

        # Number of fluorescence parameters
        self.fluor_param_dim = output_file.createDimension('n_fluorescence_parameters', 2)

        if self.enable_cloud_3d:
            self.cloud_3d_dim = output_file.createDimension('n_cloud_3d_parameters', 2)

    def _create_datasets(self, output_file):

        logger.debug("Creating file datasets")

        # Scenario
        self.scenario_group = output_file.createGroup('Scenario')
        self.obs_id = self.scenario_group.createVariable('observation_id', np.int64, (self.snd_id_dim.name,))
        self.time = self.scenario_group.createVariable('time', float, (self.snd_id_dim.name, self.channel_dim.name))
        self.latitude = self.scenario_group.createVariable('latitude', float, (self.snd_id_dim.name, self.channel_dim.name))
        self.longitude = self.scenario_group.createVariable('longitude', float, (self.snd_id_dim.name, self.channel_dim.name))
        self.surface_height = self.scenario_group.createVariable('surface_height', float, (self.snd_id_dim.name, self.channel_dim.name))
        self.solar_zenith = self.scenario_group.createVariable('solar_zenith', float, (self.snd_id_dim.name, self.channel_dim.name))
        self.solar_azimuth = self.scenario_group.createVariable('solar_azimuth', float, (self.snd_id_dim.name, self.channel_dim.name))
        self.observation_zenith = self.scenario_group.createVariable('observation_zenith', float, (self.snd_id_dim.name, self.channel_dim.name))
        self.observation_azimuth = self.scenario_group.createVariable('observation_azimuth', float, (self.snd_id_dim.name, self.channel_dim.name))
        self.relative_velocity = self.scenario_group.createVariable('relative_velocity', float, (self.snd_id_dim.name, self.channel_dim.name))

        self.solar_distance = self.scenario_group.createVariable('solar_distance', float, (self.snd_id_dim.name,))
        self.solar_velocity = self.scenario_group.createVariable('solar_velocity', float, (self.snd_id_dim.name,))

        self.spectral_coefficient = self.scenario_group.createVariable('spectral_coefficient', float, (self.snd_id_dim.name, self.channel_dim.name, self.spec_coeff_dim.name))
        self.stokes_coefficient = self.scenario_group.createVariable('stokes_coefficient', float, (self.snd_id_dim.name, self.channel_dim.name, self.stokes_coeff_dim.name))
        
        # Instrument
        self.instrument_group = output_file.createGroup('Instrument')

        self.sample_indexes = self.instrument_group.createVariable('sample_indexes', np.int32, (self.snd_id_dim.name, self.channel_dim.name, self.index_dim.name))

        self.ils_delta_lambda = self.instrument_group.createVariable('ils_delta_lambda', float, (self.snd_id_dim.name, self.channel_dim.name, self.samp_dim.name, self.ils_dim.name))
        self.ils_response = self.instrument_group.createVariable('ils_response', float, (self.snd_id_dim.name, self.channel_dim.name, self.samp_dim.name, self.ils_dim.name))
        self.rad_uncertainty = self.instrument_group.createVariable('radiance_uncertainty', float, (self.snd_id_dim.name, self.channel_dim.name, self.samp_dim.name))

        self.eofs = {}
        for eof_idx in range(1,4):
            eof_name = f'eof_{eof_idx}'
            eof_order = self.instrument_group.createVariable(eof_name, float, (self.snd_id_dim.name, self.channel_dim.name))
            self.eofs[eof_name] = eof_order
        self.eof_type_var = self.instrument_group.createVariable('eof_type', 'S1', (self.snd_id_dim.name, self.name_len.name))

        # Atmosphere
        self.atmosphere_group = output_file.createGroup('Atmosphere')

        self.surface_pressure = self.atmosphere_group.createVariable('surface_pressure', float, (self.snd_id_dim.name))
        self.pressure_levels = self.atmosphere_group.createVariable('pressure_levels', float, (self.snd_id_dim.name, self.lev_dim.name))
        self.temperature = self.atmosphere_group.createVariable('temperature', float, (self.snd_id_dim.name, self.lev_dim.name))
        
        # Absorbers
        self.absorber_group = self.atmosphere_group.createGroup('Absorber')
        self.gas_name = self.absorber_group.createVariable('name', 'S1', (self.snd_id_dim.name, self.gas_dim.name, self.name_len.name))
        self.gas_vmr = self.absorber_group.createVariable('vmr', float, (self.snd_id_dim.name, self.gas_dim.name, self.lev_dim.name))

        # Aerosols
        self.aerosol_group = self.atmosphere_group.createGroup('Aerosol')
        self.aer_name = self.aerosol_group.createVariable('name', 'S1', (self.snd_id_dim.name, self.aer_dim.name, self.name_len.name))
        self.aer_prop_name = self.aerosol_group.createVariable('property_name', 'S1', (self.snd_id_dim.name, self.aer_dim.name, self.name_len.name))
        self.aer_param = self.aerosol_group.createVariable('gaussian_params', float, (self.snd_id_dim.name, self.aer_dim.name, self.aer_param_dim.name))

        # Ground
        self.ground_group = self.atmosphere_group.createGroup('Ground')
        self.ground_type_var = self.ground_group.createVariable('type', 'S1', (self.snd_id_dim.name, self.name_len.name))

        if self.ground_type == GroundType.lambertian:
            self.albedo = self.ground_group.createVariable('lambertian_albedo', float, (self.snd_id_dim.name, self.channel_dim.name, self.albedo_poly_dim.name))
        elif self.ground_type == GroundType.lambertian_piecewise:
            self.albedo_grid = self.ground_group.createVariable('lambertian_albedo_grid', float, (self.snd_id_dim.name, self.albedo_piecewise_dim.name))
            self.albedo_points = self.ground_group.createVariable('lambertian_albedo_points', float, (self.snd_id_dim.name, self.albedo_piecewise_dim.name))
        elif self.ground_type == GroundType.brdf:
            self.brdf = self.ground_group.createVariable('brdf_parameters', float, (self.snd_id_dim.name, self.channel_dim.name, self.brdf_params_dim.name))
        elif self.ground_type == GroundType.coxmunk:
            self.windspeed = self.ground_group.createVariable('windspeed', float, (self.snd_id_dim.name,))
        elif self.ground_type == GroundType.coxmunk_lambertian:
            self.windspeed = self.ground_group.createVariable('windspeed', float, (self.snd_id_dim.name,))
            self.coxmunk_albedo = self.ground_group.createVariable('coxmunk_albedo', float, (self.snd_id_dim.name, self.channel_dim.name, self.coxmunk_albedo_dim.name))

        # Fluorescence
        self.fluorescence = self.atmosphere_group.createVariable('fluorescence', float, (self.snd_id_dim.name, self.fluor_param_dim.name))

        if self.enable_cloud_3d:
            self.cloud_3d = self.atmosphere_group.createVariable('cloud_3d', float, (self.snd_id_dim.name, self.channel_dim.name, self.fluor_param_dim.name))

    def _set_converged_state_vector(self, sounding_id, state_vector):

        logger.debug("Updating state vector to converged state")

        w_sid = np.where(self.diag_sounding_ids == str(sounding_id))

        if w_sid[0].size == 0:
            raise Exception(f"Could not find sounding id: {sounding_id} in L2 diagnostic file: {self.diag_file.filename}")
        
        snd_index = w_sid[0][0]

        logger.debug(f"Found {sounding_id} in L2 diagnostic file at index {snd_index}")

        ret_sv_names = [ n.decode('UTF-8') for n in self.diag_file['/RetrievedStateVector/state_vector_names'][snd_index, :] ]
        ret_sv_values = self.diag_file['/RetrievedStateVector/state_vector_result'][snd_index, :]

        sv_names_fmt = "{:40s} <- {:40s}"
        logger.debug("-" * 84)
        logger.debug("{:^84s}".format("State Vector Names"))
        logger.debug("-" * 84)
        logger.debug(sv_names_fmt.format("Config SV", "Retrieved SV"))
        logger.debug("-" * 84)

        for config_name, ret_name in zip_longest(state_vector.state_vector_name, ret_sv_names, fillvalue=""):
            logger.debug(sv_names_fmt.format(config_name, ret_name))

        if len(ret_sv_values) != len(state_vector.state):
            raise Exception(f"Retrieved state vector size: {len(ret_sv_values)} does not match configuration state vector size: {len(state_vector.state)}")

        logger.debug("Updating configuration state vector with retrieved values. Please manually verify that the above printed state vector name mapping is correct.")

        state_vector.update_state(ret_sv_values)
        
        logger.debug(f"Updated state vector:\n{state_vector}")

    def _fill_datasets(self, output_file):

        logger.debug("Filling datasets with values from configuration")

        for snd_idx, sid in enumerate(self.sounding_id_list):
            snd_config = self.config(sid)

            if self.diag_file is not None:
                self._set_converged_state_vector(sid, snd_config.retrieval.state_vector)

            self.obs_id[snd_idx] = int(sid)
 
            # Scenario data from L1B reader
            # Copy per channel L1B values 
            l1b = snd_config.input.l1b
            for chan_idx in range(l1b.number_spectrometer()):
                self.time[snd_idx] = l1b.time(chan_idx).pgs_time
                self.time.units = "Seconds since 1993-01-01"

                for val_name in ['latitude', 'longitude', 'solar_zenith', 'solar_azimuth', 'relative_velocity']:
                    logger.debug("Copying L1B value: %s" % val_name)
                    getattr(self, val_name)[snd_idx, chan_idx] = getattr(l1b, val_name)(chan_idx).value
                    getattr(self, val_name).units = getattr(l1b, val_name)(chan_idx).units.name

                for nc_name, l1b_name in {'surface_height': 'altitude', 'observation_zenith': 'sounding_zenith', 'observation_azimuth': 'sounding_azimuth'}.items():
                    logger.debug("Copying L1B value: %s" % l1b_name)
                    getattr(self, nc_name)[snd_idx, chan_idx] = getattr(l1b, l1b_name)(chan_idx).value
                    getattr(self, nc_name).units = getattr(l1b, l1b_name)(chan_idx).units.name

                logger.debug("Copying L1B value: spectral_coefficient")
                self.spectral_coefficient[snd_idx, chan_idx, :] = l1b.spectral_coefficient(chan_idx).value
                self.spectral_coefficient.units = l1b.spectral_coefficient(chan_idx).units.name

                logger.debug("Copying L1B value: stokes_coefficient")
                self.stokes_coefficient[snd_idx, chan_idx, :] = l1b.stokes_coefficient(chan_idx)

            logger.debug("Copying L1B value: solar_distance")
            self.solar_distance[snd_idx] = l1b.solar_distance.value
            self.solar_distance.units = l1b.solar_distance.units.name

            logger.debug("Copying L1B value: solar_velocity")
            self.solar_velocity[snd_idx] = l1b.solar_velocity.value
            self.solar_velocity.units = l1b.solar_velocity.units.name

            # Sample indexes
            logger.debug("Setting sample indexes:")
            for chan_idx in range(l1b.number_spectrometer()):
                chan_samples = snd_config.forward_model.spectral_grid.pixel_list(chan_idx)
                self.sample_indexes[snd_idx, chan_idx, :len(chan_samples)] = chan_samples
                logger.debug("%d samples in channel %d" % (len(chan_samples), chan_idx+1))

            # ILS
            inst = snd_config.instrument
            for chan_idx in range(l1b.number_spectrometer()):
                logger.debug("Copying ils_delta_lambda for channel %d" % (chan_idx + 1))
                self.ils_delta_lambda[snd_idx, chan_idx, :, :] = inst.ils(chan_idx).ils_function.delta_lambda

                logger.debug("Copying ils_response for channel %d" % (chan_idx + 1))
                self.ils_response[snd_idx, chan_idx, :, :] = inst.ils(chan_idx).ils_function.response

            # Radiance uncertainty
            logger.debug("Copying radiance uncertainty")
            for chan_idx in range(l1b.number_spectrometer()):
                self.rad_uncertainty[snd_idx, chan_idx, :] = l1b.radiance(chan_idx).uncertainty
                self.rad_uncertainty.units = l1b.radiance(chan_idx).units.name

            # EOF
            eof_type_str = "N/A"
            for chan_idx in range(l1b.number_spectrometer()):
                for corr_obj in inst.instrument_correction(chan_idx):

                    eof_obj = rf.EmpiricalOrthogonalFunction.convert_from_instrument_correction(corr_obj)

                    if eof_obj is not None:
                        logger.debug(f"Copying EOF {eof_obj.order} for channel {chan_idx+1}")

                        self.eofs[f"eof_{eof_obj.order}"][snd_idx, chan_idx] = eof_obj.scale

                        # Use HDF group name for determining type string, last part of HDF path lower cased
                        eof_type_str = eof_obj.hdf_group_name.split("/")[-1].lower()

            self.eof_type_var[snd_idx, :] = netCDF4.stringtochar(np.array([eof_type_str], 'S%d' % self.max_name_len))

            # Atmosphere
            logger.debug("Copying pressure")
            atm = snd_config.atmosphere
            self.surface_pressure[snd_idx] = atm.pressure.surface_pressure.value.value
            self.surface_pressure.units = atm.pressure.surface_pressure.units.name
            
            self.pressure_levels[snd_idx, :] = atm.pressure.pressure_grid().value.value
            self.pressure_levels.units = atm.pressure.pressure_grid().units.name

            logger.debug("Copying temperature")
            temp_adwu = atm.temperature.temperature_grid(atm.pressure)
            self.temperature[snd_idx, :] = temp_adwu.value.value
            self.temperature.units = temp_adwu.units.name

            # Absorber
            self.gas_vmr.units = "VMR"
            for gas_index in range(atm.absorber.number_species):
                gas_name = atm.absorber.gas_name(gas_index)
                logger.debug("Copying absorber: %s" % gas_name)
                self.gas_name[snd_idx, gas_index, :] = netCDF4.stringtochar(np.array([gas_name], 'S%d' % self.max_name_len))
                self.gas_vmr[snd_idx, gas_index, :] = atm.absorber.absorber_vmr(gas_name).vmr_grid(atm.pressure).value

            # Aerosol
            for aer_index in range(atm.aerosol.number_particle):
                aer_name = atm.aerosol.aerosol_name[aer_index]
                logger.debug("Copying aerosol: %s" % aer_name)
                self.aer_name[snd_idx, aer_index, :] = netCDF4.stringtochar(np.array([aer_name], 'S%d' % self.max_name_len))
                self.aer_param[snd_idx, aer_index, :] = atm.aerosol.aerosol_extinction(aer_index).aerosol_parameter
                aer_config = snd_config.config_def['atmosphere']['aerosol'].get(aer_name, None)

                # Try and get aerosol property name from configuration falling back on the aerosol name itself as that of the property
                if aer_config is not None:
                    prop_name = aer_config['properties'].get('prop_name', aer_name) 
                else:
                    prop_name = aer_name

                self.aer_prop_name[snd_idx, aer_index, :] = netCDF4.stringtochar(np.array([prop_name], 'S%d' % self.max_name_len))

            # Ground
            ground_type_str = self.ground_type.name
            self.ground_type_var[snd_idx, :] = netCDF4.stringtochar(np.array([ground_type_str], 'S%d' % self.max_name_len))

            for chan_idx in range(l1b.number_spectrometer()):
                if self.ground_type == GroundType.lambertian:
                    logger.debug("Copying ground albedo parameters, channel {}".format(chan_idx))
                    ref_point = atm.ground.reference_point(chan_idx)
                    self.albedo[snd_idx, chan_idx, 0] = atm.ground.albedo(ref_point, chan_idx).value
                    self.albedo[snd_idx, chan_idx, 1:] = 0.0
                elif self.ground_type == GroundType.brdf:
                    logger.debug("Copying ground brdf parameters, channel {}".format(chan_idx))

                    self.brdf[snd_idx, chan_idx, atm.ground.RAHMAN_KERNEL_FACTOR_INDEX] = atm.ground.rahman_factor(chan_idx).value
                    self.brdf[snd_idx, chan_idx, atm.ground.RAHMAN_OVERALL_AMPLITUDE_INDEX] = atm.ground.hotspot_parameter(chan_idx).value
                    self.brdf[snd_idx, chan_idx, atm.ground.RAHMAN_ASYMMETRY_FACTOR_INDEX] = atm.ground.asymmetry_parameter(chan_idx).value
                    self.brdf[snd_idx, chan_idx, atm.ground.RAHMAN_GEOMETRIC_FACTOR_INDEX] = atm.ground.anisotropy_parameter(chan_idx).value
                    self.brdf[snd_idx, chan_idx, atm.ground.BREON_KERNEL_FACTOR_INDEX] = atm.ground.breon_factor(chan_idx).value
                    self.brdf[snd_idx, chan_idx, atm.ground.BRDF_WEIGHT_INTERCEPT_INDEX] = atm.ground.weight_intercept(chan_idx).value
                    self.brdf[snd_idx, chan_idx, atm.ground.BRDF_WEIGHT_SLOPE_INDEX] = atm.ground.weight_slope(chan_idx).value
                elif self.ground_type == GroundType.coxmunk_lambertian:
                    logger.debug("Copying ground coxmunk lambertian parameters, channel {}".format(chan_idx))
                    ref_point = atm.ground.lambertian.reference_point(chan_idx)
                    offset = chan_idx*2
                    self.coxmunk_albedo[snd_idx, chan_idx, :] = atm.ground.lambertian.sub_state_vector_values.value[offset:offset+2]

            # Lambertian piecewise
            if self.ground_type == GroundType.lambertian_piecewise:
                albedo_grid = atm.ground.spectral_points()
                
                albedo_values = np.zeros(albedo_grid.value.shape[0])
                for idx in range(albedo_grid.value.shape[0]):
                    val_at_point = atm.ground.value_at_point(albedo_grid[idx])
                    albedo_values[idx] = val_at_point.value

                self.albedo_grid[snd_idx, :] = albedo_grid.value
                self.albedo_grid.units = albedo_grid.units.name

                self.albedo_points[snd_idx, :] = albedo_values

            # Windspeed
            if self.ground_type == GroundType.coxmunk:
                self.windspeed[snd_idx] = atm.ground.windspeed().value
            elif self.ground_type == GroundType.coxmunk_lambertian:
                self.windspeed[snd_idx] = atm.ground.coxmunk.windspeed().value
 
            # Extra data annotations
            if self.ground_type == GroundType.lambertian:
                self.albedo.parameter_names = "0: Albedo offset\n 1: Albedo intercept..."
            elif self.ground_type == GroundType.brdf:
                self.brdf.parameter_names = f"""{atm.ground.RAHMAN_KERNEL_FACTOR_INDEX}: Rahman kernel factor
{atm.ground.RAHMAN_OVERALL_AMPLITUDE_INDEX}: Rahman hotspot parameter
{atm.ground.RAHMAN_ASYMMETRY_FACTOR_INDEX}: Rahman asymmetry factor
{atm.ground.RAHMAN_GEOMETRIC_FACTOR_INDEX}: Rahman anisotropy parameter
{atm.ground.BREON_KERNEL_FACTOR_INDEX}: Breon kernel factor
{atm.ground.BRDF_WEIGHT_INTERCEPT_INDEX}: BRDF overall weight intercept
{atm.ground.BRDF_WEIGHT_SLOPE_INDEX}: BRDF overall weight slope"""
            elif self.ground_type == GroundType.coxmunk_lambertian:
                self.coxmunk_albedo.parameter_names = "0: Albedo offset\n 1: Albedo intercept..."
 
            # Fluorescence
            spectrum_effects = snd_config.forward_model.spectrum_effect

            # Flurouescene effect is only in the first channel's spectrum effects
            for spec_eff_obj in spectrum_effects[0]:
                if isinstance(spec_eff_obj, rf.FluorescenceEffect):
                    logger.debug("Copying fluorescence parameters")
                    self.fluorescence[snd_idx, :] = [ spec_eff_obj.fluorescence_at_reference,
                                                      spec_eff_obj.fluorescence_slope ]
                    break

            # Cloud 3D hard coded values as examples
            if self.enable_cloud_3d:
                logger.debug("Creating cloud 3D values")
                self.cloud_3d[snd_idx, :, 0] = 0.2
                self.cloud_3d[snd_idx, :, 1] = 0.0002

    def save(self, output_file):

        logger.debug("Writing to file: %s" % output_file.filepath())

        # Create output file dimension objects
        self._create_dims(output_file)

        # Create datasets to fill information from soundings
        self._create_datasets(output_file)

        # Fill datasets with information from configurations
        self._fill_datasets(output_file)

def main():
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="Create a simulation file from OCO L1B and Met data using the retrieval set up")
    
    parser.add_argument("l1b_file", 
        help="Path to L1B file")

    parser.add_argument("met_file",
        help="Path to Meteorology file")

    parser.add_argument("output_file", 
        help="Output h5 filename")

    parser.add_argument("-s", "--sounding_ids_file", metavar="FILE", required=True,
        help="File with list of sounding ids to simulate")

    parser.add_argument("-d", "--diag_file", metavar="FILE", required=True,
        help="Path to the L2 Diagnostic file to optionally use to prime state vector with converged results")

    parser.add_argument("--cloud_3d", action="store_true",
        help="Add Cloud 3D effect values to simulation file")

    ground_types = [ v.value for v in list(GroundType) ]
    parser.add_argument("-g", "--ground_type", choices=ground_types, default=GroundType.lambertian.value)

    parser.add_argument("-v", "--verbose", action="store_true",
        help="Turn on verbose logging")

    args = parser.parse_args()

    logging.basicConfig(level=args.verbose and logging.DEBUG or logging.INFO, format="%(message)s", stream=sys.stdout)

    with netCDF4.Dataset(args.output_file, "w") as output_file:

        output_file.set_auto_mask(True)

        with open(args.sounding_ids_file) as sounding_id_file:
            sounding_id_list = []
            for sounding_id_line in sounding_id_file:
                sounding_id_list.append( sounding_id_line.strip() )

        sim_file = SimulationWriter(args.l1b_file, args.met_file, sounding_id_list, diag_file=args.diag_file, ground_type=args.ground_type, enable_cloud_3d=args.cloud_3d)
        sim_file.save(output_file)

if __name__ == "__main__":
    main()
