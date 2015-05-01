"""
Use the openmoltools wrappers of OpenEye and Antechamber to rebuild input files
for the FreeSolv database.

Looks for freesolve database using environment variable FREESOLV_PATH
Outputs two LOCAL directories of files: ./tripos_mol2/ and ./mol2files_gaff/
"""

import pickle
import os
import glob
import openmoltools

FREESOLV_PATH = os.environ["FREESOLV_PATH"]
FREESOLV_FILENAME = os.path.join(FREESOLV_PATH, "primary-data/primary-data.pickle")

database = pickle.load(open(FREESOLV_FILENAME))

def make_path(filename):
    try:
        path = os.path.split(filename)[0]
        os.makedirs(path)
    except OSError:
        pass


for (key, entry) in database.items():
    print "Processing molecule %s ..." % (key)
    
    tripos_filename = os.path.join('tripos_mol2', key + '.mol2')
    frcmod_filename = os.path.join("./mol2files_gaff/", "%s.frcmod" % key)
    gaff_mol2_filename = os.path.join("./mol2files_gaff/", "%s.mol2" % key)
    prmtop_filename = os.path.join("./mol2files_gaff/", "%s.prmtop" % key)
    inpcrd_filename = os.path.join("./mol2files_gaff/", "%s.inpcrd" % key)

    #Add output directories if not already present
    make_path( 'tripos_mol2/' )
    make_path( 'mol2files_gaff/')

    molecule = openmoltools.openeye.smiles_to_oemol(entry['smiles'])
    charged = openmoltools.openeye.get_charges(molecule)
    openmoltools.openeye.molecule_to_mol2(charged, tripos_filename)

    _, _ = openmoltools.utils.run_antechamber(key, tripos_filename, charge_method=None, gaff_mol2_filename=gaff_mol2_filename, frcmod_filename=frcmod_filename)
    openmoltools.utils.run_tleap(key, gaff_mol2_filename, frcmod_filename, prmtop_filename=prmtop_filename, inpcrd_filename=inpcrd_filename)
