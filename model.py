__author__ = 'Patrick B. Grinaway'

import pymc
import numpy as np
import math


class GBFFModel(object):
    """
    A base class representing a Bayesian model fitting GB parameters to solvation free energy data
    """

    def __init__(self, database, initial_parameters, gbmodel, energy_function):
        """
        Arguments
        ---------
        database : dict
            Database of FreeSolv solvation free energy data
        initial_parameters : dict
            Dict containing the starting set of parameters for the model
        ngbmodels : int
            The index of the GB model to sample
        energy_function : function
            The function to use in computing the energies of molecules
        """

        self.energy_function = energy_function





    def _create_single_gbmodel(self, database, gbmodel, initial_parameters):
        """
        Generates a PyMC model to sample within one GB model (no Reversible Jump)

        Arguments
        ---------
        database : dict
            Database of FreeSolv solvation free energy data
        gbmodel : int
            The index of the GB model to sample
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


        gbffmodel['log_sigma'] = pymc.Uniform('log_sigma', lower=log_sigma_min, upper=log_sigma_max, value=log_sigma_guess)
        gbffmodel['sigma'] = pymc.Lambda('sigma', lambda log_sigma=gbffmodel['log_sigma'] : math.exp(log_sigma))
        gbffmodel['tau'] = pymc.Lambda('tau', lambda sigma=gbffmodel['sigma']: sigma**(-2))


        def __generate_parameter_priors(self, )


