#!/usr/bin/env python

import os
import re
import sys
import logging

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

class RtSimulation(object):

    def __init__(self, simulation_file, simulation_indexes):
        self.sim_file = simulation_file
        self.sim_indexes = simulation_indexes

        with netCDF4.Dataset(simulation_file) as sim_contents:
            self.all_obs_ids = sim_contents['/Scenario/observation_id'][:]

    def config(self, index):

        logging.debug("Loading configuration for #%d" % (index+1))

        config_def = oco_config.simulation_config_definition(self.sim_file, index)
        config_inst = process_config(config_def)

        # Necessary for jacobians to work
        sv = config_inst.retrieval.state_vector
        sv.update_state(config_inst.retrieval.initial_guess)

        return config_inst

    def radiances(self, index, config):
        fm = config.forward_model

        logger.debug("Calculating radiances for #%d"  % (index+1))

        radiances = []
        for channel_idx in range(config.common.num_channels):
            radiances.append(fm.radiance(channel_idx))

        return radiances

    def _create_dims(self, output_file):

        logging.debug("Setting up file dimensions")

        self.sim_dim = output_file.createDimension('n_simulation', len(self.sim_indexes))

        max_sv_len = 0
        max_sv_name = 0
        num_channels = 0
        for sim_index in self.sim_indexes:
            sim_config = self.config(sim_index)

            sv = sim_config.retrieval.state_vector

            max_sv_len = max(max_sv_len, sv.state.shape[0])
            max_config_name_len = np.max([ len(n) for n in sv.state_vector_name ])
            max_sv_name = max(max_sv_name, max_config_name_len)

            num_channels = sim_config.common.num_channels

        self.sv_dim = output_file.createDimension('n_state_vector', max_sv_len)

        self.sv_name_dim = output_file.createDimension('n_sv_name', max_sv_name)

        self.channel_dim = output_file.createDimension('n_channel', num_channels)

        self.grid_dim = output_file.createDimension('n_radiance')


    def _create_datasets(self, output_file):
        
        logger.debug("Creating file datasets")

        self.sim_index = output_file.createVariable("simulation_index", float, (self.sim_dim.name,))
        self.obs_id = output_file.createVariable("observation_id", int, (self.sim_dim.name,))

        self.sv_val = output_file.createVariable("state_vector_values", float, (self.sim_dim.name, self.sv_dim.name,))
        self.sv_name = output_file.createVariable("state_vector_names", 'S1', (self.sim_dim.name, self.sv_dim.name, self.sv_name_dim.name))

        self.spectral_domain = output_file.createVariable("spectral_domain", float, (self.sim_dim.name, self.channel_dim.name, self.grid_dim.name))
        self.radiance = output_file.createVariable("radiance", float, (self.sim_dim.name, self.channel_dim.name, self.grid_dim.name))
        self.jacobian = output_file.createVariable("jacobian", float, (self.sim_dim.name, self.channel_dim.name, self.grid_dim.name, self.sv_dim.name))

    def _fill_datasets(self, output_file):

        for cfg_idx, sim_index in enumerate(self.sim_indexes):
            sim_config = self.config(sim_index)

            sim_index = self.sim_indexes[cfg_idx]
            self.sim_index[cfg_idx] = sim_index

            self.obs_id[cfg_idx] = self.all_obs_ids[sim_index]

            sv = sim_config.retrieval.state_vector
            num_sv = sv.state.shape[0]

            self.sv_val[cfg_idx, :num_sv] = sv.state
            self.sv_name[cfg_idx, :num_sv] = netCDF4.stringtochar(np.array(sv.state_vector_name, 'S%d' % self.sv_name_dim.size))

            for chan_idx, chan_rad in enumerate(self.radiances(sim_index, sim_config)):
                num_rad = chan_rad.spectral_domain.data.shape[0]

                self.spectral_domain[cfg_idx, chan_idx, :num_rad] = chan_rad.spectral_domain.data
                self.spectral_domain.units = chan_rad.spectral_domain.units.name

                self.radiance[cfg_idx, chan_idx, :num_rad] = chan_rad.spectral_range.data_ad.value
                self.radiance.units = chan_rad.spectral_range.units.name

                self.jacobian[cfg_idx, chan_idx, :num_rad, :num_sv] = chan_rad.spectral_range.data_ad.jacobian

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
    
    parser = ArgumentParser(description="Run OCO-2 Radiance and Jacobian using the descriptions in a simulation file")
    
    parser.add_argument("simulation_file", 
        help="Path to simulation file")

    parser.add_argument("output_file", 
        help="Output h5 filename")

    parser.add_argument("-i", "--index", metavar="INT", action="append",
        help="Index or range of indexes to model, default is all indexes in the simulation file")

    parser.add_argument("-v", "--verbose", action="store_true",
        help="Turn on verbose logging")

    args = parser.parse_args()

    logging.basicConfig(level=args.verbose and logging.DEBUG or logging.INFO, format="%(message)s", stream=sys.stdout)

    indexes = []
    if args.index is None:
        with netCDF4.Dataset(args.simulation_file) as sim_file:
            indexes = range(sim_file.dimensions['n_sounding'].size)
    else:
        for arg_idx in args.index:
            if "-" in arg_idx:
                beg, end = re.split("\s*-\s*", arg_idx)
                for sim_index in range(int(beg), int(end)+1):
                    indexes.append(sim_index)
            else:
                indexes.append(int(arg_idx))

    with netCDF4.Dataset(args.output_file, "w") as output_file:

        output_file.set_auto_mask(True)

        logger.debug("Simulating %d radiances" % len(indexes))

        sim = RtSimulation(args.simulation_file, indexes)
        sim.save(output_file)

if __name__ == "__main__":
    main()
