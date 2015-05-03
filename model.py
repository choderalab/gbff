__author__ = 'Patrick B. Grinaway'

import pymc
import numpy as np
import math

class GBFFModel(object):
    """
    A class representing a Bayesian model fitting GB parameters to solvation free energy data
    """

    def __init__(self, database, initial_parameters, gbmodel):
        """
        Arguments
        ---------
        database : dict
            Database of FreeSolv solvation free energy data
        initial_parameters : dict
            Dict containing the starting set of parameters for the model
        ngbmodels : int
            The index of the GB model to sample
        """

        for (key, value) in initial_parameters.iteritems():
            (atomtype, parameter_name) = key.split('_')
            if parameter_name == 'scalingFactor':
                stochastic = pymc.Uniform(key, value=value, lower=+0.01, upper=+1.0)
            elif parameter_name == 'radius':
                stochastic = pymc.Uniform(key, value=value, lower=0.5, upper=3.5)
            else:
                raise Exception("Unrecognized parameter name: %s" % parameter_name)
            setattr(self, key, stochastic)

        self.ngbmodels = ngbmodels
        gbmodel_dir = pymc.Dirichlet('gbmodel_dir', np.ones([ngbmodels]))
        self.gbmodel_prior = pymc.CompletedDirichlet('gbmodel_prior', gbmodel_dir)
        self.gbmodel = pymc.Categorical('gbmodel', p=self.gbmodel_prior)


        log_sigma_min = math.log(0.01) # kcal/mol
        log_sigma_max = math.log(10.0) # kcal/mol
        log_sigma_guess = math.log(0.2)
        self.log_sigma = pymc.Uniform('log_sigma', lower=log_sigma_min, upper=log_sigma_max, value=log_sigma_guess)
        self.sigma = pymc.Lambda('sigma', lambda log_sigma=self.log_sigma : math.exp(log_sigma))
        self.tau = pymc.Lambda('tau', lambda sigma=self.sigma: sigma**(-2))



    def create_single_gbmodel(self, database, gbmodel, initial_parameters):
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


        for (key, value) in initial_parameters.iteritems():
            (atomtype, parameter_name) = key.split('_')
            if parameter_name == 'scalingFactor':
                stochastic = pymc.Uniform(key, value=value, lower=+0.01, upper=+1.0)
            elif parameter_name == 'radius':
                stochastic = pymc.Uniform(key, value=value, lower=0.5, upper=3.5)
            else:
                raise Exception("Unrecognized parameter name: %s" % parameter_name)


        gbffmodel['log_sigma'] = pymc.Uniform('log_sigma', lower=log_sigma_min, upper=log_sigma_max, value=log_sigma_guess)
        gbffmodel['sigma'] = pymc.Lambda('sigma', lambda log_sigma=log_sigma: math.exp(log_sigma))
        gbffmodel['tau'] = pymc.Lambda('tau', lambda sigma=sigma: sigma**(-2))

