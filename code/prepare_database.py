__author__ = 'Patrick B. Grinaway'

import os
import os.path
import time
import copy
import celery

import simtk.openmm as openmm
import simtk.unit as units
import simtk.openmm.app as app

import openeye.oechem

from openeye import oechem
import numpy as np
import numpy.linalg as linalg

#=============================================================================================
# Constants
#=============================================================================================

kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA # Boltzmann constant



def read_gbsa_parameters(filename):
        """
        Read a GBSA parameter set from a file.

        ARGUMENTS

        filename (string) - the filename to read parameters from

        RETURNS

        parameters (dict) - parameters[(atomtype,parameter_name)] contains the dimensionless parameter

        TODO

        * Replace this with a standard format?

        """

        parameters = dict()

        infile = open(filename, 'r')
        for line in infile:
            # Strip trailing comments
            index = line.find('%')
            if index != -1:
                line = line[0:index]

            # Parse parameters
            elements = line.split()
            print("the length of the elements is %d" % len(elements))
            if len(elements) == 3:
                [atomtype, radius, scalingFactor] = elements
                parameters['%s_%s' % (atomtype,'radius')] = float(radius)
                parameters['%s_%s' % (atomtype,'scalingFactor')] = float(scalingFactor)
            elif len(elements) == 6:
                [atomtype, radius, scalingFactor, alpha, beta, gamma] = elements
                parameters['%s_%s' % (atomtype,'radius')] = float(radius)
                parameters['%s_%s' % (atomtype,'scalingFactor')] = float(scalingFactor)
                parameters['%s_%s' % (atomtype,'alpha')] = float(alpha)
                parameters['%s_%s' % (atomtype,'beta')] = float(beta)
                parameters['%s_%s' % (atomtype,'gamma')] = float(gamma)


        return parameters

#=============================================================================================
# Generate simulation data.
#=============================================================================================

def generate_simulation_data(database, parameters):
    """
    Regenerate simulation data for given parameters.

    ARGUMENTS

    database (dict) - database of molecules
    parameters (dict) - dictionary of GBSA parameters keyed on GBSA atom types

    """

    platform = openmm.Platform.getPlatformByName("Reference")

    from pymbar import timeseries

    for cid in database.keys():
        entry = database[cid]
        molecule = entry['molecule']
        iupac_name = entry['iupac']

        # Retrieve vacuum system.
        vacuum_system = copy.deepcopy(entry['system'])

        # Retrieve OpenMM System.
        solvent_system = copy.deepcopy(entry['system'])

        # Get nonbonded force.
        forces = { solvent_system.getForce(index).__class__.__name__ : solvent_system.getForce(index) for index in range(solvent_system.getNumForces()) }
        nonbonded_force = forces['NonbondedForce']

        # Add GBSA term
        gbsa_force = openmm.GBSAOBCForce()
        gbsa_force.setNonbondedMethod(openmm.GBSAOBCForce.NoCutoff) # set no cutoff
        gbsa_force.setSoluteDielectric(1)
        gbsa_force.setSolventDielectric(78)

        # Build indexable list of atoms.
        atoms = [atom for atom in molecule.GetAtoms()]
        natoms = len(atoms)

        # Assign GBSA parameters.
        for (atom_index, atom) in enumerate(atoms):
            [charge, sigma, epsilon] = nonbonded_force.getParticleParameters(atom_index)
            atomtype = atom.GetStringData("gbsa_type") # GBSA atomtype
            radius = parameters['%s_%s' % (atomtype, 'radius')] * units.angstroms
            scalingFactor = parameters['%s_%s' % (atomtype, 'scalingFactor')]
            gbsa_force.addParticle(charge, radius, scalingFactor)

        # Add the force to the system.
        solvent_system.addForce(gbsa_force)

        # Create context for solvent system.
        timestep = 2.0 * units.femtosecond
        collision_rate = 20.0 / units.picoseconds
        temperature = entry['temperature']
        integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
        context = openmm.Context(vacuum_system, integrator, platform)

        # Set the coordinates.
        positions = entry['positions']
        context.setPositions(positions)

        # Minimize.
        openmm.LocalEnergyMinimizer.minimize(context)

        # Simulate, saving periodic snapshots of configurations.
        kT = kB * temperature
        beta = 1.0 / kT

        initial_time = time.time()
        nsteps_per_iteration = 2500
        niterations = 200
        x_n = np.zeros([niterations,natoms,3], np.float32) # positions, in nm
        u_n = np.zeros([niterations], np.float64) # energy differences, in kT
        for iteration in range(niterations):
            integrator.step(nsteps_per_iteration)
            state = context.getState(getEnergy=True, getPositions=True)
            x_n[iteration,:,:] = state.getPositions(asNumpy=True) / units.nanometers
            u_n[iteration] = beta * state.getPotentialEnergy()

        if np.any(np.isnan(u_n)):
            raise Exception("Encountered NaN for molecule %s | %s" % (cid, iupac_name))

        final_time = time.time()
        elapsed_time = final_time - initial_time

        # Clean up.
        del context, integrator

        # Discard initial transient to equilibration.
        [t0, g, Neff_max] = timeseries.detectEquilibration(u_n)
        x_n = x_n[t0:,:,:]
        u_n = u_n[t0:]

        # Subsample to remove correlation.
        indices = timeseries.subsampleCorrelatedData(u_n, g=g)
        x_n = x_n[indices,:,:]
        u_n = u_n[indices]

        # Store data.
        entry['x_n'] = x_n
        entry['u_n'] = u_n

        print "%48s | %64s | simulation %12.3f s | %5d samples discarded | %5d independent samples remain" % (cid, iupac_name, elapsed_time, t0, len(indices))

    return


#=============================================================================================
# Prepare the FreeSolv-format database for calculations.
#=============================================================================================

def prepare_database(database, atomtypes_filename,parameters,  mol2_directory, verbose=False):
    """
    Wrapper function to prepare the database for sampling

    """
    #TODO: fix this. Right now it inserts the path for the parent directory to access atomtyping.py
    cwd = os.getcwd()
    sys.path.insert(0,os.path.dirname(cwd))
    from atomtyping import type_atoms
    database_prepped = load_database(database, mol2_directory, verbose=verbose)
    database_with_systems = create_openmm_systems(database_prepped, verbose=verbose)
    database_atomtyped = type_atoms(database_with_systems, atomtypes_filename, verbose=verbose)
    database_simulated = generate_simulation_data(database_atomtyped, parameters)
    return database_simulated

def load_database(database, mol2_directory, verbose=False):
    """
    This function prepares the database that will be use in sampling.

    Arguments
    ---------
    database : dict
        an unpickled version of the FreeSolv database
    mol2_directory : String
        the path to the FreeSolv mol2 files containing geometry and charges
    verbose : Boolean, optional
        verbosity

    Returns
    -------
    database : dict
        An updated version of the database dict containing OEMols
    """

    start_time = time.time()
    if verbose:
        print("Reading all molecules in dataset. Will use charges and coordinates from dataset.")
    for cid in database.keys():
        entry = database[cid]

        # Store temperature
        # TODO: Get this from database?
        entry['temperature'] = 300.0 * units.kelvin

        # Extract relevant entry data from database.
        smiles = entry['smiles']
        iupac_name = entry['iupac']
        experimental_DeltaG = entry['expt'] * units.kilocalories_per_mole
        experimental_dDeltaG = entry['d_expt'] * units.kilocalories_per_mole

        # Read molecule.
        molecule = openeye.oechem.OEMol()

        # Load the mol2 file.
        tripos_mol2_filename = os.path.join(mol2_directory, cid + '.mol2')
        omolstream = oechem.oemolistream(tripos_mol2_filename)
        oechem.OEReadMolecule(omolstream, molecule)
        omolstream.close()
        molecule.SetTitle(iupac_name)
        molecule.SetData('cid', cid)

        # Add explicit hydrogens.
        oechem.OEAddExplicitHydrogens(molecule)

        # Store molecule.
        entry['molecule'] = oechem.OEMol(molecule)

        if verbose:
            print "%d molecules read" % len(database.keys())
            end_time = time.time()
            elapsed_time = end_time - start_time
            print "%.3f s elapsed" % elapsed_time
    return database

def create_openmm_systems(database, verbose=False, path_to_prmtops=None):
    """
    Create an OpenMM system for each molecule in the database

    Arguments
    ---------
    database : dict
        dict containing FreeSolv molecules (prepared using prepare_database)
    verbose : Boolean, optional
        verbosity
    path_to_prmtops : str, optional, default=None
        Path to directory containing inpcrd and prmtop files.
        If None, will be set to ${FREESOLV_PATH}/mol2files_gaff/

    Returns
    -------
    database : dict
        The FreeSolv database dict containing OpenMM systems for each molecule
    """

    if path_to_prmtops is None:
        FREESOLV_PATH = os.environ["FREESOLV_PATH"]
        path_to_prmtops = os.path.join(FREESOLV_PATH + "/mol2files_gaff/")

    for cid, entry in database.items():

        prmtop_filename = os.path.join(path_to_prmtops, "%s.prmtop" % cid)
        inpcrd_filename = os.path.join(path_to_prmtops, "%s.inpcrd" % cid)

        # Create OpenMM System object for molecule in vacuum.
        prmtop = app.AmberPrmtopFile(prmtop_filename)
        inpcrd = app.AmberInpcrdFile(inpcrd_filename)
        system = prmtop.createSystem(nonbondedMethod=app.NoCutoff, constraints=app.HBonds, implicitSolvent=None, removeCMMotion=False)
        positions = inpcrd.getPositions()

        # Store system and positions.
        entry['system'] = system
        entry['positions'] = positions

    return database

if __name__=="__main__":
    import os
    import pickle
    import sys
    from optparse import OptionParser
    usage = """\
    usage: %prog --types typefile --parameters parameterfile --output outputpickle
    """
    version_string = "%prog 1.0"
    parser = OptionParser(usage=usage, version=version_string)
    parser.add_option("-t", "--types", metavar='TYPES',
                      action="store", type="string", dest='atomtypes_filename', default='',
                      help="Filename defining atomtypes as SMARTS atom matches.")
    parser.add_option("-p", "--parameters", metavar='PARAMETERS',
                      action="store", type="string", dest='parameters_filename', default='',
                      help="File containing initial parameter set.")
    parser.add_option("-o", "--dbout", metavar='DBOUT',
                      action="store", type="string", dest='dbout', default='database.prepared.pickle',
                      help="Output name of the prepared database pickle")
    (options,args) = parser.parse_args()
    if options.atomtypes_filename=='' or options.parameters_filename=='':
        parser.print_help()
        parser.error("All input files must be specified.")


    database_filepath = os.path.join(os.environ['FREESOLV_PATH'], 'database.pickle')
    database_file = open(database_filepath,'r')
    database_raw = pickle.load(database_file)
    database_file.close()
    mol2_directory = os.path.join(os.environ['FREESOLV_PATH'], 'tripos_mol2')
    parameters = read_gbsa_parameters(options.parameters_filename)
    database_prepared = prepare_database(database_raw, options.atomtypes_filename, parameters, mol2_directory)
    outfile = open(options.dbout,'w')
    pickle.dump(database_prepared, outfile)
    outfile.close()