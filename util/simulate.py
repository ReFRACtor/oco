#!/usr/bin/env python

import os
import sys
import logging

import netCDF4

from refractor.executor import StrategyExecutor

logger = logging.getLogger(__name__)

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
    
    config_filename = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "config/simulation_config.py"))

    strategy_list = []
    for idx in indexes:
        strategy_list.append( { "sim_file": args.simulation_file, "sim_index": idx } )

    exc = StrategyExecutor(config_filename, output_filename=args.output_file, strategy_list=strategy_list)
    
    logger.debug("Simulating %d radiances" % len(indexes))
    exc.execute_simulation()

if __name__ == "__main__":
    main()
