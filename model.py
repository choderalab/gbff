__author__ = 'Patrick B. Grinaway'

import pymc
import numpy as np
import math

class GBFFModel(object):
    """
    A class representing a Bayesian model fitting GB parameters to solvation free energy data
    """

    def __init__(self, database, initial_parameters, ngbmodels=3):
        """
        Arguments
        ---------
        database : dict
            Database of FreeSolv solvation free energy data
        initial_parameters : dict
            Dict containing the starting set of parameters for the model
        ngbmodels : int, optional
            The number of GB models to sample. Default 3
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
