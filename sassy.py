#!/usr/bin/env python

from __future__ import print_function

from performance_fitness import FitnessFunction

import sys
import argparse


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle_file")

    args = parser.parse_args()

    ff = FitnessFunction(bundle_file=args.bundle_file)
    fitness = ff.evaluate_fitness()
    print(fitness)

if __name__ == '__main__':
    sys.exit(main(sys.argv))