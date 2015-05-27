__author__ = 'Patrick B. Grinaway'

import pymc
import numpy as np
import math
import copy
import simtk.unit as units
import simtk.openmm as openmm
import simtk.openmm.app.internal.customgbforces as customgbforces


class GBFFModel(object):
    """
    A base class representing a Bayesian model fitting GB parameters to solvation free energy data
    """

    def __init__(self, database, initial_parameters, parameter_types, hydration_energy_factory):
        """
        Arguments
        ---------
        database : dict
            Database of FreeSolv solvation free energy data
        initial_parameters : dict
            Dict containing the starting set of parameters for the model
        parameter_types : list of strings
            A list of the different types of parameters (e.g., ['radius','scalingFactor'] for OBC2)
        hydration_energy_factory : function
            This will generate a function to compute the hydration free energy of molecules
        """

        solvated_system_database = self._create_solvated_systems(database, initial_parameters)
        self.hydration_energy_factory = hydration_energy_factory
        self.parameter_types = parameter_types
        self.parameter_model = self._create_parameter_model(solvated_system_database, initial_parameters)
        self.model = self._create_bayesian_gbmodel(solvated_system_database, initial_parameters)







    def _create_bayesian_gbmodel(self, database, initial_parameters):
        """
        Generates the PyMC model to sample within one GB model (no Reversible Jump)

        Arguments
        ---------
        database : dict
            Database of FreeSolv solvation free energy data
        initial_parameters : dict
            Dict containing the starting set of parameters for the model

        Returns
        -------
        gbffmodel : dict
            A dict containing the nodes of a PyMC model to sample

        """

        gbffmodel = dict()
        log_sigma_min = math.log(0.01) # kcal/mol
        log_sigma_max = math.log(10.0) # kcal/mol
        log_sigma_guess = math.log(0.2)
        cid_list = database.keys()



        def RMSE(**args):
            nmolecules = len(cid_list)
            error = np.zeros([nmolecules], np.float64)
            for (molecule_index, cid) in enumerate(cid_list):
                entry = database[cid]
                error[molecule_index] = args['dg_gbsa_%s' % cid] - float(entry['expt'])
            mse = np.mean((error - np.mean(error))**2)
            return np.sqrt(mse)



        gbffmodel['log_sigma'] = pymc.Uniform('log_sigma', lower=log_sigma_min, upper=log_sigma_max, value=log_sigma_guess)
        gbffmodel['sigma'] = pymc.Lambda('sigma', lambda log_sigma=gbffmodel['log_sigma'] : math.exp(log_sigma))
        gbffmodel['tau'] = pymc.Lambda('tau', lambda sigma=gbffmodel['sigma']: sigma**(-2))

        gbffmodel.update(self.parameter_model)
        gbffmodel_with_mols = self._add_mols_gbffmodel(database, gbffmodel)



        RMSE_parents = {'dg_gbsa_%s'%cid : gbffmodel_with_mols['dg_gbsa_%s' % cid] for cid in cid_list}
        gbffmodel_with_mols['RMSE'] = pymc.Deterministic(eval=RMSE, name='RMSE', parents=RMSE_parents, doc='RMSE', dtype=float, trace=True, verbose=1)
        return gbffmodel_with_mols


    def _create_solvated_systems(self, database, initial_parameters):
        """
        Generate a system with the appropriate GB force added. This does not prevent recompilation. Specific
        to the GB model, not implemented here.

        Arguments
        ---------
        database : dict
            FreeSolv database
        initial_parameters : dict
            Dictionary of parameters and their initial values

        Returns
        -------
        database : dict
            Dictionary containing solvated systems with appropriate forces.
        """
        raise NotImplementedError

    def _create_parameter_model(self, database, initial_parameters):
        """
        Abstract - Creates the PyMC nodes for each parameter. Will be specific to each model.

        Arguments
        ---------
        database : dict
            Dict containing FreeSolv database of solvation free energies
        initial_parameters : dict
            Dict containing the starting set of parameters for the model

        Returns
        -------
        parameters : dict
            PyMC model containing the parameters to sample
        """
        raise NotImplementedError

    def _add_mols_gbffmodel(self, database, gbffmodel):

        """
        Create the error model for the hydration free energies
        Arguments
        ---------
        database : dict
            FreeSolv solvation free energy database in dict form

        """
       # gbffmodel = dict()
        cid_list = database.keys()
        for (molecule_index, cid) in enumerate(cid_list):
            entry = database[cid]
            dg_exp_name = "dg_exp_%s" % cid
            dg_gbsa_name = "dg_gbsa_%s" % cid
            parents = self._get_parameters_of_molecule(entry['molecule'])
            dg_exp = float(entry['expt']) # observed hydration free energy in kcal/mol
            ddg_exp = float(entry['d_expt']) # observed hydration free energy uncertainty in kcal/mol
            gbffmodel['tau_%s' % cid] = pymc.Lambda('tau_%s' % cid, lambda sigma=gbffmodel['sigma'] : 1.0 / (sigma**2 + ddg_exp**2) ) # Include model error
            hydration_energy_function = self.hydration_energy_factory(entry)
            gbffmodel[dg_gbsa_name] = pymc.Deterministic(eval=hydration_energy_function, doc=cid, name=dg_gbsa_name, parents=parents, dtype=float, trace=True, verbose=1)
            gbffmodel[dg_exp_name] = pymc.Normal(dg_exp_name, mu=gbffmodel['dg_gbsa_%s' % cid], tau=gbffmodel['tau_%s' % cid], value=dg_exp, observed=True)
        return gbffmodel


    def _add_parallel_gbffmodel(self, database, gbffmodel):
        """
        Create a version of the GBFF model using arrays inside the PyMC objects

        """

        cid_list = database.keys()
        dg_exp = [float(database[cid]['expt']) for cid in enumerate(cid_list)]
        ddg_exp = [float(database[cid]['d_expt']) for cid in enumerate(cid_list)]
        gbffmodel['taus'] = [self._make_tau(cid, database, gbffmodel) for cid in enumerate(cid_list)]
        hydration_energy_function = self.hydration_energy_factory(database)
        gbffmodel['dg_gbsa'] = pymc.Deterministic(eval=hydration_energy_function, doc='ComputedDeltaG', name='dg_gbsa', parents=self.parameter_model, dtype=float, trace=True, verbose=1)
        gbffmodel['dg_exp'] = pymc.Normal('dg_exp', mu=gbffmodel['dg_gbsa'], tau=gbffmodel['taus'], value = dg_exp, observed=True)
        return gbffmodel




    def _make_tau(self, cid, database, gbffmodel):
        """
        An auxiliary function to make the molecule taus in a cleaner, more separated fashion
        than having the list comprehension do everything

        Arguments
        ---------
        cid : string
            The compound ID
        database : dcit
            The FreeSolv database
        gbffmodel : dict
            A dictionary of the nodes for the pymc model

        Returns
        -------
        tau : pymc.Lambda
            The tau pymc lambda
        """
        ddg_exp = float(database[cid]['d_expt'])
        lambda_sigma  = lambda sigma=gbffmodel['sigma'] : 1.0 / (sigma**2 + ddg_exp**2)
        return pymc.Lambda('tau_%s' % cid, lambda_sigma)




    def _get_parameters_of_molecule(self, mol):
        """
        This is a convenience function to identify the parameters necessary for a given molecule.

        Arguments
        ---------
        mol : OEMol
            an OEMol object that has been annotated by the AtomTyper

        Returns
        -------
        parents : dict
            Dictionary of the PyMC stochastics representing necessary parameters
        """
        parents = dict()
        for atom in mol.GetAtoms():
            atomtype = atom.GetStringData("gbsa_type")
            for parameter_name in self.parameter_types:
                stochastic_name = '%s_%s' % (atomtype,parameter_name)
                parents[stochastic_name] = self.parameter_model[stochastic_name]
        return parents

    @property
    def pymc_model(self):
        return self.model


class GBFFThreeParameterModel(GBFFModel):
    """
    A class that samples within the GBSA HCT, OBC1, OBC2 models
    """

    def __init__(self, database, initial_parameters, hydration_energy_factory, gbmodel=1):
        """
        Arguments
        ---------
        database : dict
            Database of FreeSolv solvation free energy data
        initial_parameters : dict
            Dict containing the starting set of parameters for the model
        hydration_energy_function : function
            The function to use in computing the energies of molecules
        gbmodel : int
            Select HCT, OBC1, or OBC2 using 1, 2, 3. Default 1 (HTC)
        """
        self.gbmodel = gbmodel
        parameter_types = ['radius', 'scalingFactor']
        super(GBFFThreeParameterModel, self).__init__(database, initial_parameters, parameter_types, hydration_energy_factory)


    def _create_solvated_systems(self, database, initial_parameters):
        """
        Create the solvated systems for the GB-HCT, OBC1, or OBC2 models

        Arguments
        ---------
        database : dict
            A dictionary of the FreeSolv database, prepared with vacuum openmm systems.
        initial_parameters : dict
            A dictionary of the initial parameters for the HCT force
        """

        cid_list = database.keys()

        for (molecule_index, cid) in enumerate(cid_list):
            entry = database[cid]
            molecule = entry['molecule']
            solvent_system = copy.deepcopy(entry['system'])
            forces = { solvent_system.getForce(index).__class__.__name__ : solvent_system.getForce(index) for index in range(solvent_system.getNumForces()) }
            nonbonded_force = forces['NonbondedForce']
            if self.gbmodel == 1:
                gbsa_force = customgbforces.GBSAHCTForce(SA='ACE')
            elif self.gbmodel == 2:
                gbsa_force = customgbforces.GBSAOBC1Force(SA='ACE')
            elif self.gbmodel == 3:
                gbsa_force = customgbforces.GBSAOBC2Force(SA='ACE')
            else:
                raise ValueError("Unsupported GBmodel %i selected" % self.gbmodel)
            atoms = [atom for atom in molecule.GetAtoms()]
            for (atom_index, atom) in enumerate(atoms):
                [charge, sigma, epsilon] = nonbonded_force.getParticleParameters(atom_index)
                atomtype = atom.GetStringData("gbsa_type") # GBSA atomtype
                radius = initial_parameters['%s_%s' % (atomtype, 'radius')] * units.angstroms
                scalingFactor = initial_parameters['%s_%s' % (atomtype, 'scalingFactor')]
                gbsa_force.addParticle([charge, radius, scalingFactor])
            solvent_system.addForce(gbsa_force)

            platform = openmm.Platform.getPlatformByName('CPU')

            entry['solvated_system'] = solvent_system
            timestep = 2.0 * units.femtosecond
            solvent_integrator = openmm.VerletIntegrator(timestep)
            solvent_context = openmm.Context(solvent_system, solvent_integrator, platform)
            print('adding the integrator and context to the dict!')
            entry['solvent_integrator'] = solvent_integrator
            entry['solvent_context'] = solvent_context

            vacuum_system = entry['system']
            vacuum_integrator = openmm.VerletIntegrator(timestep)
            vacuum_context = openmm.Context(vacuum_system, vacuum_integrator, platform)
            entry['vacuum_integrator'] = vacuum_integrator
            entry['vacuum_context'] = vacuum_context

            database[cid] = entry

        return database

    def _create_parameter_model(self, database, initial_parameters):
        """
        Creates set of stochastics representing the HCT parameters

        Arguments
        ---------
        database : dict
            FreeSolv database
        initial_parameters : dict
            The set of initial values of the parameters

        Returns
        -------
        parameters : dict
            PyMC dictionary containing the parameters to sample.\
        """

        parameters = dict() # just the parameters
        for (key, value) in initial_parameters.iteritems():
            (atomtype, parameter_name) = key.split('_')
            if parameter_name == 'scalingFactor':
                stochastic = pymc.Uniform(key, value=value, lower=+0.01, upper=+1.0)
            elif parameter_name == 'radius':
                stochastic = pymc.Uniform(key, value=value, lower=0.5, upper=3.5)
            else:
                raise Exception("Unrecognized parameter name: %s" % parameter_name)
            parameters[key] = stochastic
        return parameters




class GBFFGBnModel(GBFFModel):
    """
    A class to sample the parameters of the GBSAGBn model
    """

    def __init__(self, database, initial_parameters, hydration_energy_function):
        """
        Arguments
        ---------
        database : dict
            Database of FreeSolv solvation free energy data
        initial_parameters : dict
            Dict containing the starting set of parameters for the model
        hydration_energy_function : function
            The function to use in computing the energies of molecules

        """
        parameter_types = ['radius', 'scalingFactor']
        super(GBFFGBnModel, self).__init__(database, initial_parameters, parameter_types, hydration_energy_function)

    def _create_solvated_systems(self, database, initial_parameters):
        """
        Create the solvated systems for the GB-HCT, OBC1, or OBC2 models

        Arguments
        ---------
        database : dict
            A dictionary of the FreeSolv database, prepared with vacuum openmm systems.
        initial_parameters : dict
            A dictionary of the initial parameters for the HCT force
        """

        cid_list = database.keys()

        for (molecule_index, cid) in enumerate(cid_list):
            entry = database[cid]
            molecule = entry['molecule']
            solvent_system = copy.deepcopy(entry['system'])
            forces = { solvent_system.getForce(index).__class__.__name__ : solvent_system.getForce(index) for index in range(solvent_system.getNumForces()) }
            nonbonded_force = forces['NonbondedForce']
            atoms = [atom for atom in molecule.GetAtoms()]
            gbsa_force = customgbforces.GBSAGBnForce(SA='ACE')
            for (atom_index, atom) in enumerate(atoms):
                [charge, sigma, epsilon] = nonbonded_force.getParticleParameters(atom_index)
                atomtype = atom.GetStringData("gbsa_type") # GBSA atomtype
                radius = initial_parameters['%s_%s' % (atomtype, 'radius')] * units.angstroms
                scalingFactor = initial_parameters['%s_%s' % (atomtype, 'scalingFactor')]
                gbsa_force.addParticle([charge, radius, scalingFactor])
            solvent_system.addForce(gbsa_force)


            platform = openmm.Platform.getPlatformByName('CPU')
            timestep = 2.0 * units.femtosecond
            solvent_integrator = openmm.VerletIntegrator(timestep)
            solvent_context = openmm.Context(solvent_system, solvent_integrator, platform)
            entry['solvated_system'] = solvent_system
            print('adding the integrator and context to the dict!')
            entry['solvent_integrator'] = solvent_integrator
            entry['solvent_context'] = solvent_context

            vacuum_system = entry['system']
            vacuum_integrator = openmm.VerletIntegrator(timestep)
            vacuum_context = openmm.Context(vacuum_system, vacuum_integrator, platform)
            entry['vacuum_integrator'] = vacuum_integrator
            entry['vacuum_context'] = vacuum_context
            database[cid] = entry
        return database

    def _create_parameter_model(self, database, initial_parameters):
        """
        Creates set of stochastics representing the HCT parameters

        Arguments
        ---------
        database : dict
            FreeSolv database
        initial_parameters : dict
            The set of initial values of the parameters

        Returns
        -------
        parameters : dict
            PyMC dictionary containing the parameters to sample.\
        """

        parameters = dict() # just the parameters
        for (key, value) in initial_parameters.iteritems():
            (atomtype, parameter_name) = key.split('_')
            if parameter_name == 'scalingFactor':
                stochastic = pymc.Uniform(key, value=value, lower=+0.01, upper=+1.5)
            elif parameter_name == 'radius':
                stochastic = pymc.Uniform(key, value=value, lower=1, upper=1.91113)
            else:
                raise Exception("Unrecognized parameter name: %s" % parameter_name)
            parameters[key] = stochastic
        return parameters




class GBFFGBn2Model(GBFFModel):
    """
    A class that representes a Python object to sample over parameters for the GBn2 model
    """

    def _create_solvated_systems(self, database, initial_parameters):
        """
        Create the solvated systems for the GBn2 model

        Arguments
        ---------
        database : dict
            A dictionary of the FreeSolv database, prepared with vacuum openmm systems.
        initial_parameters : dict
            A dictionary of the initial parameters for the HCT force
        """
        cid_list = database.keys()

        for (molecule_index, cid) in enumerate(cid_list):
            entry = database[cid]
            molecule = entry['molecule']
            solvent_system = copy.deepcopy(entry['system'])
            forces = { solvent_system.getForce(index).__class__.__name__ : solvent_system.getForce(index) for index in range(solvent_system.getNumForces()) }
            nonbonded_force = forces['NonbondedForce']
            atoms = [atom for atom in molecule.GetAtoms()]
            gbsa_force = customgbforces.GBSAGBnForce(SA='ACE')
            for (atom_index, atom) in enumerate(atoms):
                [charge, sigma, epsilon] = nonbonded_force.getParticleParameters(atom_index)
                atomtype = atom.GetStringData("gbsa_type") # GBSA atomtype
                radius = initial_parameters['%s_%s' % (atomtype, 'radius')] * units.angstroms
                scalingFactor = initial_parameters['%s_%s' % (atomtype, 'scalingFactor')]
                alpha = initial_parameters['%s_%s' %(atomtype, 'alpha')]
                beta = initial_parameters['%s_%s' %(atomtype, 'beta')]
                gamma = initial_parameters['%s_%s' %(atomtype, 'gamma')]
                gbsa_force.addParticle([charge, radius, scalingFactor, alpha, beta, gamma])
            solvent_system.addForce(gbsa_force)
            entry['solvated_system'] = solvent_system
            database[cid] = entry
        return database

    def _create_parameter_model(self, database, initial_parameters):
        """
        Creates set of stochastics representing the HCT parameters

        Arguments
        ---------
        database : dict
            FreeSolv database
        initial_parameters : dict
            The set of initial values of the parameters

        Returns
        -------
        parameters : dict
            PyMC dictionary containing the parameters to sample.\
        """
        uninformative_tau = 0.0001
        parameters = dict() # just the parameters
        for (key, value) in initial_parameters.iteritems():
            (atomtype, parameter_name) = key.split('_')
            if parameter_name == 'scalingFactor':
                stochastic = pymc.Uniform(key, value=value, lower=+0.01, upper=+1.0)
            elif parameter_name == 'radius':
                stochastic = pymc.Uniform(key, value=value, lower=1, upper=2)
            elif parameter_name == 'alpha':
                stochastic = pymc.Normal(key, value=value, tau=uninformative_tau)
            elif parameter_name == 'beta':
                stochastic = pymc.Normal(key, value=value, tau=uninformative_tau)
            elif parameter_name == 'gamma':
                stochastic = pymc.Normal(key, value=value, tau=uninformative_tau)
            else:
                raise Exception("Unrecognized parameter name: %s" % parameter_name)
            parameters[key] = stochastic
        return parameters
