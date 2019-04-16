#!/usr/bin/env python

import os
import sys
import logging
from collections import OrderedDict

import netCDF4
import numpy as np

# Find where the code repository is located relative to this file
oco_repo_path = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))

# Add the path to the configuration so it can be imported
sys.path.append(os.path.join(oco_repo_path))

# Import ReFRACtor framework
from refractor.factory import process_config
from refractor import framework as rf

# Import configuration module
from config import oco_config

logger = logging.getLogger()

class SimulationWriter(object): 

    def __init__(self, l1b_file, met_file, sounding_id_list, max_name_len=25, albedo_degree=4):
        
        logging.debug("Creating simulation file using L1B: %s, Met: %s" % (l1b_file, met_file))

        self.l1b_file = l1b_file
        self.met_file = met_file
        self.sounding_id_list = sounding_id_list

        self.max_name_len = 80
        self.albedo_degree = albedo_degree

    def config(self, sounding_id):

        logging.debug("Loading configuration for sounding: %s" % sounding_id)

        config_def = oco_config.retrieval_config_definition(self.l1b_file, self.met_file, sounding_id)
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
            num_channel = inst.number_spectrometer()
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

        # Length of names of gas and aerosols
        self.name_len = output_file.createDimension('name_length', self.max_name_len)

        self.albedo_poly_dim = output_file.createDimension('n_albedo_poly', self.albedo_degree + 1)

        # Number of aerosol parameters
        self.aer_param_dim = output_file.createDimension('n_aerosol_parameters', 3)

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
        self.solar_distance = self.scenario_group.createVariable('solar_distance', float, (self.snd_id_dim.name, self.channel_dim.name))
        self.relative_velocity = self.scenario_group.createVariable('relative_velocity', float, (self.snd_id_dim.name, self.channel_dim.name))

        self.spectral_coefficient = self.scenario_group.createVariable('spectral_coefficient', float, (self.snd_id_dim.name, self.channel_dim.name, self.spec_coeff_dim.name))
        self.stokes_coefficient = self.scenario_group.createVariable('stokes_coefficient', float, (self.snd_id_dim.name, self.channel_dim.name, self.stokes_coeff_dim.name))
        
        # ILS
        self.instrument_group = output_file.createGroup('Instrument')
        self.ils_delta_lambda = self.instrument_group.createVariable('ils_delta_lambda', float, (self.snd_id_dim.name, self.channel_dim.name, self.samp_dim.name, self.ils_dim.name))
        self.ils_response = self.instrument_group.createVariable('ils_response', float, (self.snd_id_dim.name, self.channel_dim.name, self.samp_dim.name, self.ils_dim.name))

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
        self.albedo = self.ground_group.createVariable('lambertian_albedo', float, (self.snd_id_dim.name, self.channel_dim.name, self.albedo_poly_dim.name))

    def _fill_datasets(self, output_file):

        logger.debug("Filling datasets with values from configuration")

        for snd_idx, sid in enumerate(self.sounding_id_list):
            snd_config = self.config(sid)

            self.obs_id[snd_idx] = int(sid)
 
            # Scenario data from L1B reader
            # Copy per channel L1B values 
            l1b = snd_config.input.l1b
            for chan_idx in range(l1b.number_spectrometer()):
                self.time[snd_idx] = l1b.time(chan_idx).pgs_time
                self.time.units = "Seconds since 1993-01-01"

                for val_name in ['latitude', 'longitude', 'solar_zenith', 'solar_azimuth', 'relative_velocity']:
                    logger.debug("Copying L1B value: %s" % val_name)
                    getattr(self, val_name)[snd_idx] = getattr(l1b, val_name)(chan_idx).value
                    getattr(self, val_name).units = getattr(l1b, val_name)(chan_idx).units.name

                for nc_name, l1b_name in {'surface_height': 'altitude', 'observation_zenith': 'sounding_zenith', 'observation_azimuth': 'sounding_azimuth'}.items():
                    logger.debug("Copying L1B value: %s" % l1b_name)
                    getattr(self, nc_name)[snd_idx] = getattr(l1b, l1b_name)(chan_idx).value
                    getattr(self, nc_name).units = getattr(l1b, l1b_name)(chan_idx).units.name

                    logger.debug("Copying L1B value: spectral_coefficient")
                    self.spectral_coefficient[snd_idx, chan_idx, :] = l1b.spectral_coefficient(chan_idx).value
                    self.spectral_coefficient.units = l1b.spectral_coefficient(chan_idx).units.name

                    logger.debug("Copying L1B value: stokes_coefficient")
                    self.stokes_coefficient[snd_idx, chan_idx, :] = l1b.stokes_coefficient(chan_idx)

            logger.debug("Copying L1B value: solar_distance")
            self.solar_distance[snd_idx] = l1b.solar_distance.value
            self.solar_distance.units = l1b.solar_distance.units.name

            # ILS
            inst = snd_config.instrument
            for chan_idx in range(l1b.number_spectrometer()):
                logger.debug("Copying ils_delta_lambda for channel %d" % (chan_idx + 1))
                self.ils_delta_lambda[snd_idx, chan_idx, :, :] = inst.ils(chan_idx).ils_function.delta_lambda

                logger.debug("Copying ils_response for channel %d" % (chan_idx + 1))
                self.ils_response[snd_idx, chan_idx, :, :] = inst.ils(chan_idx).ils_function.response

            # Atmosphere
            logger.debug("Copying pressure")
            atm = snd_config.atmosphere
            self.surface_pressure[snd_idx] = atm.pressure.surface_pressure.value.value
            self.surface_pressure.units = atm.pressure.surface_pressure.units.name
            
            self.pressure_levels[snd_idx, :] = atm.pressure.pressure_grid.value.value
            self.pressure_levels.units = atm.pressure.pressure_grid.units.name

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
                prop_name = snd_config.config_def['atmosphere']['aerosol'][aer_name]['properties'].get('prop_name', aer_name)
                self.aer_prop_name[snd_idx, aer_index, :] = netCDF4.stringtochar(np.array([prop_name], 'S%d' % self.max_name_len))

            # Ground
            logger.debug("Copying ground albedo")
            for chan_idx in range(l1b.number_spectrometer()):
                ref_point = atm.ground.reference_point(0)
                self.albedo[snd_idx, chan_idx, 0] = atm.ground.albedo(ref_point, chan_idx).value
                self.albedo[snd_idx, chan_idx, 1:] = 0.0

    def save(self, output_file):

        logger.debug("Writing to file: %s" % output_file.filepath)

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

        sim_file = SimulationWriter(args.l1b_file, args.met_file, sounding_id_list)
        sim_file.save(output_file)

if __name__ == "__main__":
    main()
