__author__ = 'Patrick B. Grinaway'

import pymc
import numpy as np
import math


class GBFFModel(object):
    """
    An abstract base class representing a Bayesian model fitting GB parameters to solvation free energy data
    """

    def __init__(self, database, initial_parameters, parameter_types, hydration_energy_function):
        """
        Arguments
        ---------
        database : dict
            Database of FreeSolv solvation free energy data
        initial_parameters : dict
            Dict containing the starting set of parameters for the model
        parameter_types : list of strings
            A list of the different types of parameters (e.g., ['radius','scalingFactor'] for OBC2)
        hydration_energy_function : function
            The function to use in computing the energies of molecules
        """

        solvated_system_database = self._create_solvated_systems(database, initial_parameters)
        self.hydration_energy_function = hydration_energy_function
        self.parameter_types = parameter_types
        self.parameter_model = self._create_parameter_model(solvated_system_database, initial_parameters)
        self.model = self._create_bayesian_gbmodel(solvated_system_database)






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
                molecule = entry['molecule']
                error[molecule_index] = args['dg_gbsa_%s' % cid] - float(entry['expt'])
            mse = np.mean((error - np.mean(error))**2)
            return np.sqrt(mse)



        gbffmodel['log_sigma'] = pymc.Uniform('log_sigma', lower=log_sigma_min, upper=log_sigma_max, value=log_sigma_guess)
        gbffmodel['sigma'] = pymc.Lambda('sigma', lambda log_sigma=gbffmodel['log_sigma'] : math.exp(log_sigma))
        gbffmodel['tau'] = pymc.Lambda('tau', lambda sigma=gbffmodel['sigma']: sigma**(-2))

        molecule_error_model = self._create_molecule_error_model(database)
        gbffmodel.update(molecule_error_model)

        RMSE_parents = {'dg_gbsa_%s'%cid : gbffmodel['dg_gbsa_%s' % cid] for cid in cid_list}
        gbffmodel['RMSE'] = pymc.Deterministic(eval=RMSE, name='RMSE', parents=RMSE_parents, doc='RMSE', dtype=float, trace=True, verbose=1)


    def _create_solvated_systems(self, database, initial_parameters):
        """
        Generate a system with the appropriate GB force added to prevent recompilation. Specific
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


    def _create_molecule_error_model(self, database):

        """
        Create the error model for the hydration free energies
        Arguments
        ---------
        database : dict
            FreeSolv solvation free energy database in dict form

        """
        molecule_error_model = dict()

        cid_list = database.keys()
        for (molecule_index, cid) in enumerate(cid_list):
            entry = database[cid]
            dg_exp_name = "dg_exp_%s" % cid
            dg_gbsa_name = "dg_gbsa_%s" % cid
            parents = self._get_parameters_of_molecule(entry['molecule'])
            molecule_error_model[dg_gbsa_name] = pymc.Deterministic(eval=self.hydration_energy_function,doc=cid, name=dg_gbsa_name, parents=parents, dtype=float, trace=True, verbose=1)
            dg_exp = float(entry['expt']) # observed hydration free energy in kcal/mol
            ddg_exp = float(entry['d_expt']) # observed hydration free energy uncertainty in kcal/mol
            molecule_error_model['tau_%s' % cid] = pymc.Lambda('tau_%s' % cid, lambda sigma=molecule_error_model['sigma'] : 1.0 / (sigma**2 + ddg_exp**2) ) # Include model error
            molecule_error_model[dg_exp_name] = pymc.Normal(dg_exp_name, mu=molecule_error_model['dg_gbsa_%s' % cid], tau=molecule_error_model['tau_%s' % cid], value=dg_exp, observed=True)
        return molecule_error_model

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
            atomtype = atom.GetStringData("gbsa_type") # GBSA atomtype
            for parameter_name in self.parameter_types:
                stochastic_name = '%s_%s' % (atomtype,parameter_name)
                parents[stochastic_name] = self.parameter_model[stochastic_name]
        return parents



class GBFFHCTModel(GBFFModel):
    """
    A class that samples within the GBSA HCT model
    """

    def __init__(self, database, initial_parameters, hydration_energy_function):



