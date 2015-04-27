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

from optparse import OptionParser # For parsing of command line arguments

import numpy

import simtk.unit as units

import pymc

import utils

#=============================================================================================
# PyMC model
#=============================================================================================


def create_model(database, initial_parameters):

    # Define priors for parameters.
    model = dict()
    parameters = dict() # just the parameters
    for (key, value) in initial_parameters.iteritems():
        (atomtype, parameter_name) = key.split('_')
        if parameter_name == 'scalingFactor':
            stochastic = pymc.Uniform(key, value=value, lower=+0.01, upper=+1.0)
        elif parameter_name == 'radius':
            stochastic = pymc.Uniform(key, value=value, lower=0.5, upper=3.5)
        else:
            raise Exception("Unrecognized parameter name: %s" % parameter_name)
        model[key] = stochastic
        parameters[key] = stochastic

    # Define prior on GB models.
    ngbmodels = 3 # number of GB models
    #all_models = numpy.ones([ngbmodels], numpy.float64) / float(ngbmodels)
    model['gbmodel_dir'] = pymc.Dirichlet('gbmodel_dir', numpy.ones([ngbmodels]))
    model['gbmodel_prior'] = pymc.CompletedDirichlet('gbmodel_prior', model['gbmodel_dir'])
    model['gbmodel'] = pymc.Categorical('gbmodel', p=model['gbmodel_prior'])

    # Define deterministic functions for hydration free energies.
    cid_list = database.keys()
    for (molecule_index, cid) in enumerate(cid_list):
        entry = database[cid]
        molecule = entry['molecule']

        molecule_name = molecule.GetTitle()
        variable_name = "dg_gbsa_%s" % cid
        # Determine which parameters are involved in this molecule to limit number of parents for caching.
        parents = dict()
        for atom in molecule.GetAtoms():
            atomtype = atom.GetStringData("gbsa_type") # GBSA atomtype
            for parameter_name in ['scalingFactor', 'radius']:
                stochastic_name = '%s_%s' % (atomtype,parameter_name)
                parents[stochastic_name] = parameters[stochastic_name]
        parents['gbmodel'] = model['gbmodel'] # add GB model choice
        print "%s : " % molecule_name,
        print parents.keys()
        hydration_energy_parameters = {}
        hydration_energy_parameters['gbmodel'] = model['gbmodel']
        # Create deterministic variable for computed hydration free energy.
        function = utils.hydration_energy_factory(entry,hydration_energy_parameters)
        model[variable_name] = pymc.Deterministic(eval=function,
                                                  name=variable_name,
                                                  parents=parents,
                                                  doc=cid,
                                                  trace=True,
                                                  verbose=1,
                                                  dtype=float,
                                                  plot=False,
                                                  cache_depth=2)

    # Define error model
    log_sigma_min              = math.log(0.01) # kcal/mol
    log_sigma_max              = math.log(10.0) # kcal/mol
    log_sigma_guess            = math.log(0.2) # kcal/mol
    model['log_sigma']         = pymc.Uniform('log_sigma', lower=log_sigma_min, upper=log_sigma_max, value=log_sigma_guess)
    model['sigma']             = pymc.Lambda('sigma', lambda log_sigma=model['log_sigma'] : math.exp(log_sigma) )
    model['tau']               = pymc.Lambda('tau', lambda sigma=model['sigma'] : sigma**(-2) )
    cid_list = database.keys()
    for (molecule_index, cid) in enumerate(cid_list):
        entry = database[cid]
        molecule = entry['molecule']

        molecule_name          = molecule.GetTitle()
        variable_name          = "dg_exp_%s" % cid
        dg_exp                 = float(molecule.GetData('expt')) # observed hydration free energy in kcal/mol
        ddg_exp                 = float(molecule.GetData('d_expt')) # observed hydration free energy uncertainty in kcal/mol
        #model[variable_name]   = pymc.Normal(variable_name, mu=model['dg_gbsa_%08d' % molecule_index], tau=model['tau'], value=expt, observed=True)
        model['tau_%s' % cid] = pymc.Lambda('tau_%s' % cid, lambda sigma=model['sigma'] : 1.0 / (sigma**2 + ddg_exp**2) ) # Include model error
        #model['tau_%s' % cid] = pymc.Lambda('tau_%s' % cid, lambda sigma=model['sigma'] : 1.0 / (ddg_exp**2) ) # Do not include model error.
        model[variable_name]   = pymc.Normal(variable_name, mu=model['dg_gbsa_%s' % cid], tau=model['tau_%s' % cid], value=dg_exp, observed=True)

    # Define convenience functions.
    parents = {'dg_gbsa_%s'%cid : model['dg_gbsa_%s' % cid] for cid in cid_list }
    def RMSE(**args):
        nmolecules = len(cid_list)
        error = numpy.zeros([nmolecules], numpy.float64)
        for (molecule_index, cid) in enumerate(cid_list):
            entry = database[cid]
            molecule = entry['molecule']
            error[molecule_index] = args['dg_gbsa_%s' % cid] - float(molecule.GetData('expt'))
        mse = numpy.mean((error - numpy.mean(error))**2)
        return numpy.sqrt(mse)

    model['RMSE'] = pymc.Deterministic(eval=RMSE,
                                       name='RMSE',
                                       parents=parents,
                                       doc='RMSE',
                                       trace=True,
                                       verbose=1,
                                       dtype=float,
                                       plot=True,
                                       cache_depth=2)

    return model

def print_file(filename):
    infile = open(filename, 'r')
    print infile.read()
    infile.close()

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
    signed_errors = numpy.zeros([len(database.keys())], numpy.float64)
    for (i, cid) in enumerate(database.keys()):
        # Get metadata.
        entry = database[cid]
        molecule = entry['molecule']
        name = molecule.GetTitle()
        dg_exp           = float(molecule.GetData('expt')) * units.kilocalories_per_mole
        ddg_exp          = float(molecule.GetData('d_expt')) * units.kilocalories_per_mole
        signed_errors[i] = energies[molecule] / units.kilocalories_per_mole - dg_exp / units.kilocalories_per_mole

        # Form output.
        outstring = "%64s %8.3f %8.3f %8.3f" % (name, dg_exp / units.kilocalories_per_mole, ddg_exp / units.kilocalories_per_mole, energies[molecule] / units.kilocalories_per_mole)

        print outstring

    print "Initial RMS error %8.3f kcal/mol" % (signed_errors.std())

    # Create MCMC model.
    model = create_model(database, parameters)

    # Sample models.
    from pymc import MCMC
    sampler = MCMC(model, db='hdf5', dbname=mcmcDbName)
    #sampler.isample(iter=mcmcIterations, burn=0, save_interval=1, verbose=options.verbose)
    sampler.sample(iter=mcmcIterations, burn=0, verbose=True, progress_bar=True)
    sampler.db.close()
