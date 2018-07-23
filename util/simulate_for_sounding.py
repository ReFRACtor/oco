#!/usr/bin/env python

import os
import sys
from functools import lru_cache

import netCDF4
import numpy as np

# Find where the code repository is located relative to this file
oco_repo_path = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))

# Add the path to the configuration so it can be imported
sys.path.append(os.path.join(oco_repo_path, 'config'))

# Import ReFRACtor framework
from refractor.factory import process_config
from refractor import framework as rf

# Import configuration module
import oco_config

class RtSimulation(object):

    def __init__(self, l1b_file, met_file, sounding_id):
        self.l1b_file = l1b_file
        self.met_file = met_file
        self.sounding_id = sounding_id

    @property
    @lru_cache()
    def config(self):

        config_def = oco_config.config_definition(self.l1b_file, self.met_file, self.sounding_id)
        config_inst = process_config(config_def)

        #from pprint import pprint
        #pprint(config_inst, indent=2)

        return config_inst

    @property
    @lru_cache()
    def radiances(self):
        fm = self.config.forward_model

        print("Calculating radiances for: " + self.sounding_id)
        radiances = []
        for channel_idx in range(self.config.common.num_channels):
            radiances.append(fm.radiance(channel_idx))

        return radiances

    def save(self, output_file):

        sounding_group = output_file.createGroup(self.sounding_id)

        sv = self.config.retrieval.state_vector

        sv_state_dim = output_file.createDimension('n_%s_sv' % (self.sounding_id), sv.state.shape[0])

        sv_val = sounding_group.createVariable("state_vector_values", float, (sv_state_dim.name,))
        sv_val[:] = sv.state

        max_name_len = np.max([ len(n) for n in sv.state_vector_name ])
        sv_chars_dim = output_file.createDimension('n_%d_sv_char', max_name_len)
        
        sv_name = sounding_group.createVariable("state_vector_names", 'S1', (sv_state_dim.name, sv_chars_dim.name))
        sv_name[:] = netCDF4.stringtochar(np.array(sv.state_vector_name, 'S%d' % max_name_len))

        for chan_idx, chan_rad in enumerate(self.radiances):
            chan_group = sounding_group.createGroup("Channel_%d" % (chan_idx + 1))
            
            chan_dim = output_file.createDimension('n_%s_rad_%d' % (self.sounding_id, chan_idx+1), chan_rad.spectral_domain.data.shape[0])

            sd_ds = chan_group.createVariable("spectral_domain", float, (chan_dim.name,))
            sd_ds[:] = chan_rad.spectral_domain.data
            sd_ds.units = chan_rad.spectral_domain.units.name

            rad_ds = chan_group.createVariable("radiance", float, (chan_dim.name,))
            rad_ds[:] = chan_rad.spectral_range.data_ad.value
            rad_ds.units = chan_rad.spectral_range.units.name

            jac_ds = chan_group.createVariable("jacobians", float, (chan_dim.name, sv_state_dim.name))
            jac_ds[:] = chan_rad.spectral_range.data_ad.jacobian

def main():
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="Run OCO-2 Radiance and Jacobian for a set of sounding ids")
    
    parser.add_argument("l1b_file", 
        help="Path to L1B file")

    parser.add_argument("met_file",
        help="Path to Meteorology file")

    parser.add_argument("output_file", 
        help="Output h5 filename")

    parser.add_argument("-s", "--sounding_ids_file", metavar="FILE", required=True,
        help="File with list of sounding ids to simulate")

    args = parser.parse_args()

    with netCDF4.Dataset(args.output_file, "w") as output_file:

        with open(args.sounding_ids_file) as sounding_id_list:
            for sounding_id_line in sounding_id_list:
                sounding_id = sounding_id_line.strip()

                sim = RtSimulation(args.l1b_file, args.met_file, sounding_id)
                sim.save(output_file)

if __name__ == "__main__":
    main()
