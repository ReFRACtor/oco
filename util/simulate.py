#!/usr/bin/env python

import os
import sys
import logging

import netCDF4

from refractor.executor import StrategyExecutor
from refractor.output.base import OutputBase

logger = logging.getLogger(__name__)

class ObservationIdOutput(OutputBase):

    def __init__(self, output, step_index, observation_id):
        base_group_name = self.iter_step_group_name(step_index)

        obs_group = output.createGroup(base_group_name)
    
        obs_id = obs_group.createVariable("observation_id", int)

        logger.debug("Simulating for observation id: {}".format(observation_id))
        obs_id[...] = observation_id

class SimulationExecutor(StrategyExecutor):

    config_filename = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "config/simulation_config.py"))

    def __init__(self, simulation_file, observation_indexes, channel_index=None, output_filename=None):
        
        strategy_list = []
        for idx in observation_indexes:
            strategy_list.append( { "sim_file": simulation_file, "sim_index": idx, "channel_index": channel_index } )

        super().__init__(self.config_filename, output_filename, strategy_list=strategy_list)

        with netCDF4.Dataset(simulation_file) as sim_contents:
            self.all_obs_ids = sim_contents['/Scenario/observation_id'][:]

        self.obs_indexes = observation_indexes

    def attach_output(self, config_inst, step_index=0):
        super().attach_output(config_inst, step_index)

        obs_id = self.all_obs_ids[self.obs_indexes[step_index]]
        obs_id_out = ObservationIdOutput(self.output, step_index, obs_id)

def main():
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="Run OCO-2 Radiance and Jacobian using the descriptions in a simulation file")
    
    parser.add_argument("simulation_file", 
        help="Path to simulation file")

    parser.add_argument("output_file", 
        help="Output h5 filename")

    parser.add_argument("-i", "--observation_index", metavar="INT", action="append",
        help="Index or range of indexes to simulate, default is all indexes in the simulation file")

    parser.add_argument("-c", "--channel_index", metavar="INT", type=int,
        help="Index of channel (band) to simulate instead of all")

    parser.add_argument("-v", "--verbose", action="store_true",
        help="Turn on verbose logging")

    args = parser.parse_args()

    logging.basicConfig(level=args.verbose and logging.DEBUG or logging.INFO, format="%(message)s", stream=sys.stdout)

    obs_indexes = []
    if args.observation_index is None:
        with netCDF4.Dataset(args.simulation_file) as sim_file:
            obs_indexes = range(sim_file.dimensions['n_sounding'].size)
    else:
        for arg_idx in args.observation_index:
            if "-" in arg_idx:
                beg, end = re.split("\s*-\s*", arg_idx)
                for sim_index in range(int(beg), int(end)+1):
                    obs_indexes.append(sim_index)
            else:
                obs_indexes.append(int(arg_idx))

    exc = SimulationExecutor(args.simulation_file, obs_indexes, args.channel_index, output_filename=args.output_file)
    
    logger.debug("Simulating %d radiances" % len(obs_indexes))
    exc.execute_simulation()

if __name__ == "__main__":
    main()
