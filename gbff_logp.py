__author__ = 'Patrick B. Grinaway'

import numpy as np
import scipy.stats as stats
import simtk.openmm as openmm
import simtk.unit as units
import simtk.openmm.app as app
import hydration_energies

class GBFFLogP(object):
    """
    This class will calculate the logp of the GBFF model without pymc
    """

    def __init__(self, database):
        self.database = database
        self.

    def logp(self, parameters):
        log_prior=0
        for (key, value) in parameters:
            (atomtype, parameter_name) = key.split('_')
            if parameter_name == 'scalingFactor':
                log_prior+=stats.uniform.logpdf(value, -0.8, 1.5)
            elif parameter_name == 'radius':
                log_prior+=stats.uniform.logpdb(value, 0.5, 2.5)
            else:
                raise Exception("Invalid parameter")


        delta_g_comp = hydration_energies.compute_hydration_energies_sequentially(self.database, parameters)


