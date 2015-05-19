#!/usr/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
parameterize-gbsa.py

Parameterize the GBSA model on hydration free energies of small molecules using Bayesian inference
via Markov chain Monte Carlo (MCMC).

AUTHORS

John Chodera <jchodera@berkeley.edu>, University of California, Berkeley

The AtomTyper class is based on 'patty' by Pat Walters, Vertex Pharmaceuticals.

"""
#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import os
import os.path
import time
import math

import model

from optparse import OptionParser # For parsing of command line arguments

import numpy as np

import simtk.unit as units

import pymc

import utils



#=============================================================================================
# MAIN
#=============================================================================================

if __name__=="__main__":

    # Create command-line argument options.
    usage_string = """\
    usage: %prog --types typefile --parameters paramfile --database database --iterations MCMC_iterations --mcmcout MCMC_db_name

    example: %prog --types parameters/gbsa-amber-mbondi2.types --parameters parameters/gbsa-amber-mbondi2.parameters --database datasets/FreeSolv/FreeSolv/database.pickle --iterations 500 --mcmcout MCMC --verbose --mol2 datasets/FreeSolv/FreeSolv/tripos_mol2 --subset 10

    """
    version_string = "%prog %__version__"
    parser = OptionParser(usage=usage_string, version=version_string)

    parser.add_option("-t", "--types", metavar='TYPES',
                      action="store", type="string", dest='atomtypes_filename', default='',
                      help="Filename defining atomtypes as SMARTS atom matches.")

    parser.add_option("-p", "--parameters", metavar='PARAMETERS',
                      action="store", type="string", dest='parameters_filename', default='',
                      help="File containing initial parameter set.")

    parser.add_option("-d", "--database", metavar='DATABASE',
                      action="store", type="string", dest='database_filename', default='',
                      help="Python pickle file of database with molecule names, SMILES strings, hydration free energies, and experimental uncertainties (FreeSolv format).")

    parser.add_option("-m", "--mol2", metavar='MOL2',
                      action="store", type="string", dest='mol2_directory', default='',
                      help="Directory containing charged mol2 files (optional).")

    parser.add_option("-i", "--iterations", metavar='ITERATIONS',
                      action="store", type="int", dest='iterations', default=150,
                      help="MCMC iterations.")

    parser.add_option("-o", "--mcmcout", metavar='MCMCOUT',
                      action="store", type="string", dest='mcmcout', default='MCMC',
                      help="MCMC output database name.")

    parser.add_option("-s", "--subset", metavar='SUBSET',
                      action="store", type="int", dest='subset_size', default=None,
                      help="Size of subset to consider (for testing).")

    parser.add_option("-v", "--verbose", metavar='VERBOSE',
                      action="store_true", dest='verbose', default=False,
                      help="Verbosity flag.")

    # Parse command-line arguments.
    (options,args) = parser.parse_args()

    # Ensure all required options have been specified.
    if options.atomtypes_filename=='' or options.parameters_filename=='' or options.database_filename=='':
        parser.print_help()
        parser.error("All input files must be specified.")

    # Read GBSA parameters.
    parameters = utils.read_gbsa_parameters(options.parameters_filename)
    print parameters

    mcmcIterations = options.iterations
    mcmcDbName     = os.path.abspath(options.mcmcout)

    # Open database.
    import pickle
    database = pickle.load(open(options.database_filename, 'r'))

    # DEBUG: Create a small subset.
    if options.subset_size:
        subset_size = options.subset_size
        cid_list = database.keys()
        database = dict((k, database[k]) for k in cid_list[0:subset_size])

    # Prepare the database for calculations.
    utils.prepare_database(database, options.atomtypes_filename, parameters, mol2_directory=options.mol2_directory, verbose=options.verbose)

    # Compute energies with all molecules.
    print "Computing all energies..."
    start_time = time.time()
    energies = utils.compute_hydration_energies(database, parameters)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print "%.3f s elapsed" % elapsed_time

    # Print comparison.
    signed_errors = np.zeros([len(database.keys())], np.float64)
    for (i, (cid, entry)) in enumerate(database.items()):
        # Get metadata.
        molecule = entry['molecule']
        name = molecule.GetTitle()
        dg_exp           = float(entry['expt']) * units.kilocalories_per_mole
        ddg_exp          = float(entry['d_expt']) * units.kilocalories_per_mole
        signed_errors[i] = energies[molecule] / units.kilocalories_per_mole - dg_exp / units.kilocalories_per_mole

        # Form output.
        outstring = "%64s %8.3f %8.3f %8.3f" % (name, dg_exp / units.kilocalories_per_mole, ddg_exp / units.kilocalories_per_mole, energies[molecule] / units.kilocalories_per_mole)

        print outstring

    print "Initial RMS error %8.3f kcal/mol" % (signed_errors.std())

    # Create MCMC model.
    gbnmodel = model.GBFFGBnModel(database, parameters, utils.hydration_energy_factory)

    # Sample models.
    sampler = pymc.MCMC(gbnmodel.pymc_model, db='hdf5', dbname=mcmcDbName)
    #sampler.isample(iter=mcmcIterations, burn=0, save_interval=1, verbose=options.verbose)
    sampler.sample(iter=mcmcIterations, burn=0, verbose=True, progress_bar=True)
    sampler.db.close()
