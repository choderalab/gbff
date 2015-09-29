__author__ = 'Patrick B. Grinaway'

import time
import numpy as np
import os
import numpy.linalg as linalg
import simtk.openmm as openmm
import copy
import simtk.unit as units
from . import app
from celery import Celery
import yaml
from celery import group

kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA # Boltzmann constant

config_file = open(os.environ['CELERY_CONFIG'], 'r')
config = yaml.load(config_file)
config_file.close()


app = Celery('hydration_energies',
             broker=config['broker'],
             backend=config['backend'],
             include=['hydration_energies.energytasks'])
@app.task
def compute_hydration_energy(entry, parameters, platform_name="CPU"):
    """
    Compute hydration energy of a single molecule given a GBSA parameter set.

    ARGUMENTS

    molecule (OEMol) - molecule with GBSA atom types
    parameters (dict) - parameters for GBSA atom types

    RETURNS

    energy (float) - hydration energy in kcal/mol

    """


    platform = openmm.Platform.getPlatformByName('CPU')

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

    solvent_context = openmm.Context(solvent_system, solvent_integrator,platform)
    vacuum_context = openmm.Context(vacuum_system, vacuum_integrator, platform)

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

    #print "%48s | %48s | DeltaG = %.3f +- %.3f kT " % (cid, iupac_name, energy, dDeltaG_in_kT)
    print(DeltaG_in_kT)
    print(type(DeltaG_in_kT))

    return DeltaG_in_kT

@app.task
def compute_hydration_energy_allmodel(entry, parameters, platform_name="CPU"):
    """
    Compute hydration energy of a single molecule given a GBSA parameter set.

    ARGUMENTS

    molecule (OEMol) - molecule with GBSA atom types
    parameters (dict) - parameters for GBSA atom types

    RETURNS

    energy (float) - hydration energy in kcal/mol

    """


    platform = openmm.Platform.getPlatformByName('CPU')

    from simtk.openmm.app.internal import customgbforces
    from pymbar import MBAR

    timestep = 2 * units.femtoseconds
    molecule = entry['molecule']
    iupac_name = entry['iupac']
    cid = molecule.GetData('cid')
    gbmodel = parameters['gbmodel']

    print(parameters)
    # Retrieve OpenMM System.
    vacuum_system = entry['system']
    solvent_system = copy.deepcopy(entry['system'])
    # Get nonbonded force.
    forces = { solvent_system.getForce(index).__class__.__name__ : solvent_system.getForce(index) for index in range(solvent_system.getNumForces()) }
    nonbonded_force = forces['NonbondedForce']
    forces = { solvent_system.getForce(index).__class__.__name__ : solvent_system.getForce(index) for index in range(solvent_system.getNumForces()) }
    nonbonded_force = forces['NonbondedForce']
    if  gbmodel == 0:
        gbsa_force = customgbforces.GBSAHCTForce(SA='ACE')
    elif gbmodel == 1:
        gbsa_force = customgbforces.GBSAOBC1Force(SA='ACE')
    elif gbmodel == 2:
        gbsa_force = customgbforces.GBSAOBC2Force(SA='ACE')
    elif gbmodel == 3:
        gbsa_force = customgbforces.GBSAGBnForce(SA='ACE')
    elif gbmodel == 4:
        gbsa_force = customgbforces.GBSAGBn2Force(SA='ACE')
    else:
        raise Exception("Unsupported gbmodel index")

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
        if gbmodel == 4:
            alpha = parameters['%s_alpha' % atomtype]
            beta = parameters['%s_beta' % atomtype]
            gamma = parameters['%s_gamma' % atomtype]
            gbsa_force.addParticle([charge, radius, scalingFactor, alpha, beta, gamma])
        else:
            gbsa_force.addParticle([charge, radius, scalingFactor])

    solvent_system.addForce(gbsa_force)
    solvent_context = openmm.Context(solvent_system, solvent_integrator,platform)
    vacuum_context = openmm.Context(vacuum_system, vacuum_integrator, platform)

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

    #print "%48s | %48s | DeltaG = %.3f +- %.3f kT " % (cid, iupac_name, energy, dDeltaG_in_kT)


    return DeltaG_in_kT.in_units_of(units.kilocalories_per_mole)



def compute_hydration_energies_sequentially(database, parameters, platform_name='CPU'):
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
        energy = compute_hydration_energy_allmodel(entry, parameters, platform_name)
        delta_gs[molecule_index] = energy
    return delta_gs

def sequential_computation_factory(database):
    def compute_hydration_energies(**parameters):
        return compute_hydration_energies_sequentially(database, parameters, platform_name='CPU')
    return compute_hydration_energies

def compute_hydration_energies_celery(database, parameters, platform_name='CPU'):
    """
    This function accepts an array of parameters and the database, and uses celery to distribute computation
    """
    energy_group = group(compute_hydration_energy_allmodel.s(database[cid], parameters, platform_name) for cid in database.keys())()
    energies = [float(energy) for energy in energy_group.get()]
    print("Energy is %s" % str(energies))
    return energies

def celery_hydration_energies_factory(database):
    def hydration_energies(**parameters):
        return compute_hydration_energies_celery(database, parameters, platform_name='CPU')
    return hydration_energies
