#!/usr/bin/env python

import os
import sys
from functools import lru_cache

import h5py
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

        sounding_group = output_file.create_group(self.sounding_id)

        for chan_idx, chan_rad in enumerate(self.radiances):
            chan_group = sounding_group.create_group("Channel_%d" % (chan_idx + 1))

            sd_ds = chan_group.create_dataset("spectral_domain", data=chan_rad.spectral_domain.data)
            sd_ds.attrs['Units'] = chan_rad.spectral_domain.units.name

            rad_ds = chan_group.create_dataset("radiance", data=chan_rad.spectral_range.data_ad.value)
            rad_ds.attrs['Units'] = chan_rad.spectral_range.units.name

            jac_ds = chan_group.create_dataset("jacobians", data=chan_rad.spectral_range.data_ad.jacobian)

        sv = self.config.retrieval.state_vector
        sounding_group.create_dataset("state_vector_values", data=sv.state)
        sounding_group.create_dataset("state_vector_names", data=np.string_(sv.state_vector_name))

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

    with h5py.File(args.output_file, "w") as output_file:
        with open(args.sounding_ids_file) as sounding_id_list:
            for sounding_id_line in sounding_id_list:
                sounding_id = sounding_id_line.strip()

                sim = RtSimulation(args.l1b_file, args.met_file, sounding_id)
                sim.save(output_file)

if __name__ == "__main__":
    main()
