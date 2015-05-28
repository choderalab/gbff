#!/usr/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Utilities for Bayesian parameterization of GBSA models based on hydration free energies of small molecules.

AUTHORS

John D. Chodera <john.chodera@choderalab.org>

The AtomTyper class is based on 'patty' by Pat Walters, Vertex Pharmaceuticals.

"""
#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import os
import os.path
import time
import copy

import simtk.openmm as openmm
import simtk.unit as units
import simtk.openmm.app as app

import openeye.oechem

from openeye import oechem
import numpy as np
import numpy.linalg as linalg
from atomtyping import type_atoms

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
            if len(elements) == 3:
                [atomtype, radius, scalingFactor] = elements
                parameters['%s_%s' % (atomtype,'radius')] = float(radius)
                parameters['%s_%s' % (atomtype,'scalingFactor')] = float(scalingFactor)

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
# Computation of hydration free energies
#=============================================================================================

def compute_hydration_energies(database, parameters, platform_name='CPU'):
    """
    Compute solvation energies of a set of molecules given a GBSA parameter set.

    ARGUMENTS

    molecules (list of OEMol) - molecules with GBSA assigned atom types in type field
    parameters (dict) - dictionary of GBSA parameters keyed on GBSA atom types

    RETURNS

    energies (dict) - energies[molecule] is the computed solvation energy of given molecule

    """

    delta_gs = np.zeros(len(database.keys()))

    for (molecule_index, cid) in enumerate(database.keys()):
        entry = database[cid]
        print(entry.keys())
        delta_gs[molecule_index] = compute_hydration_energy(entry, parameters, platform_name)
    return delta_gs / units.kilocalories_per_mole



def compute_hydration_energy(entry, parameters, platform_name="CPU"):
    """
    Compute hydration energy of a single molecule given a GBSA parameter set.

    ARGUMENTS

    molecule (OEMol) - molecule with GBSA atom types
    parameters (dict) - parameters for GBSA atom types

    RETURNS

    energy (float) - hydration energy in kcal/mol

    """


    platform = openmm.Platform.getPlatformByName(platform_name)

    from pymbar import MBAR

    timestep = 2 * units.femtoseconds

    molecule = entry['molecule']
    iupac_name = entry['iupac']
    cid = molecule.GetData('cid')

    # Retrieve OpenMM System.
    vacuum_system = entry['system']
    solvent_system = copy.deepcopy(entry['solvated_system'])

    # Get nonbonded force.
    forces = { solvent_system.getForce(index).__class__.__name__ : solvent_system.getForce(index) for index in range(solvent_system.getNumForces()) }
    nonbonded_force = forces['NonbondedForce']
    gbsa_force = forces['CustomGBForce']

    # Build indexable list of atoms.
    atoms = [atom for atom in molecule.GetAtoms()]
    natoms = len(atoms)


    # Create context for solvent system.
    timestep = 2.0 * units.femtosecond
    solvent_integrator = openmm.VerletIntegrator(timestep)


    # Create context for vacuum system.
    vacuum_integrator = openmm.VerletIntegrator(timestep)

    # Assign GBSA parameters.
    for (atom_index, atom) in enumerate(atoms):
        [charge, sigma, epsilon] = nonbonded_force.getParticleParameters(atom_index)
        atomtype = atom.GetStringData("gbsa_type") # GBSA atomtype
        radius = parameters['%s_%s' % (atomtype, 'radius')] * units.angstroms
        scalingFactor = parameters['%s_%s' % (atomtype, 'scalingFactor')]
        gbsa_force.setParticleParameters(atom_index, [charge, radius, scalingFactor])

    solvent_context = openmm.Context(solvent_system, solvent_integrator)
    vacuum_context = openmm.Context(vacuum_system, vacuum_integrator)

    # Compute energy differences.
    temperature = entry['temperature']
    kT = kB * temperature
    beta = 1.0 / kT

    initial_time = time.time()
    x_n = entry['x_n']
    u_n = entry['u_n']
    nsamples = len(u_n)
    nstates = 3 # number of thermodynamic states
    u_kln = np.zeros([3,3,nsamples], np.float64)
    for sample in range(nsamples):
        positions = units.Quantity(x_n[sample,:,:], units.nanometers)

        u_kln[0,0,sample] = u_n[sample]

        vacuum_context.setPositions(positions)
        vacuum_state = vacuum_context.getState(getEnergy=True)
        u_kln[0,1,sample] = beta * vacuum_state.getPotentialEnergy()

        solvent_context.setPositions(positions)
        solvent_state = solvent_context.getState(getEnergy=True)
        u_kln[0,2,sample] = beta * solvent_state.getPotentialEnergy()

    N_k = np.zeros([nstates], np.int32)
    N_k[0] = nsamples


    mbar = MBAR(u_kln, N_k)
    try:
        df_ij, ddf_ij, _ = mbar.getFreeEnergyDifferences()
    except linalg.LinAlgError:
        return np.inf

    DeltaG_in_kT = df_ij[1,2]
    dDeltaG_in_kT = ddf_ij[1,2]

    final_time = time.time()
    elapsed_time = final_time - initial_time
    #print "%48s | %48s | reweighting took %.3f s" % (cid, iupac_name, elapsed_time)

    # Clean up.
    del solvent_context, solvent_integrator
    del vacuum_context, vacuum_integrator

    energy = kT * DeltaG_in_kT

    #print "%48s | %48s | DeltaG = %.3f +- %.3f kT " % (cid, iupac_name, DeltaG_in_kT, dDeltaG_in_kT)

    return energy / units.kilocalories_per_mole







def hydration_energy_factory(entry):
    def hydration_energy(**parameters):
        return compute_hydration_energy(entry, parameters, platform_name="CPU")
    return hydration_energy

def array_hydration_energies_factory(database):
    def hydration_energies(**parameters):
        return compute_hydration_energies(database, parameters, platform_name='CPU')
    return hydration_energies

#=============================================================================================
# Prepare the FreeSolv-format database for calculations.
#=============================================================================================

def prepare_database(database, atomtypes_filename,parameters,  mol2_directory, verbose=False):
    """
    Wrapper function to prepare the database for sampling

    """

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
    print("Creating")
    
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

