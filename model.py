__author__ = 'Patrick B. Grinaway'

import pymc
import numpy as np


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

    self.ngbmodels = ngbmodels
    