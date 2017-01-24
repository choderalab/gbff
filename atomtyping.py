import sys
import time
import string
from openeye import oechem

#=============================================================================================
# Atom Typer
#=============================================================================================

class AtomTyper(object):
    """
    Atom typer

    Based on 'Patty', by Pat Walters.

    """

    class TypingException(Exception):
        """
        Atom typing exception.

        """
        def __init__(self, molecule, atom):
            self.molecule = molecule
            self.atom = atom

        def __str__(self):
            return "Atom not assigned: %6d %8s" % (self.atom.GetIdx(), oechem.OEGetAtomicSymbol(self.atom.GetAtomicNum()))

    def __init__(self, infileName, tagname):
        self.pattyTag = oechem.OEGetTag(tagname)
        self.smartsList = []
        ifs = open(infileName)
        lines = ifs.readlines()
        for line in lines:
            # Strip trailing comments
            index = line.find('%')
            if index != -1:
                line = line[0:index]
            # Split into tokens.
            toks = string.split(line)
            if len(toks) == 2:
                smarts,type = toks
                pat = oechem.OESubSearch()
                pat.Init(smarts)
                pat.SetMaxMatches(0)
                self.smartsList.append([pat,type,smarts])

    def dump(self):
        for pat,type,smarts in self.smartsList:
            print(pat,type,smarts)

    def assignTypes(self,mol):
        # Assign null types.
        for atom in mol.GetAtoms():
            atom.SetStringData(self.pattyTag, "")

        # Assign atom types using rules.
        oechem.OEAssignAromaticFlags(mol)
        for pat,type,smarts in self.smartsList:
            for matchbase in pat.Match(mol):
                for matchpair in matchbase.GetAtoms():
                    matchpair.target.SetStringData(self.pattyTag,type)

        # Check if any atoms remain unassigned.
        for atom in mol.GetAtoms():
            if atom.GetStringData(self.pattyTag)=="":
                raise AtomTyper.TypingException(mol, atom)

    def debugTypes(self,mol):
        for atom in mol.GetAtoms():
            print("%6d %8s %8s" % (atom.GetIdx(),oechem.OEGetAtomicSymbol(atom.GetAtomicNum()),atom.GetStringData(self.pattyTag)))

    def getTypeList(self,mol):
        typeList = []
        for atom in mol.GetAtoms():
            typeList.append(atom.GetStringData(self.pattyTag))
        return typeList

#=============================================================================================
# Utility routines
#=============================================================================================



def type_atoms(database, atomtypes_filename, verbose=False):
    """
    Generate types for each atom in each molecule

    Arguments
    ---------
    database : dict
        dict containing Freesolv database (prepared with prep_database)
    atomtypes_filename : String
        location of the filename containing the set of atomtypes

    Returns
    -------
    database : dict
        Freesolv dict with molecule atom types set.
    """

    atom_typer = AtomTyper(atomtypes_filename, "gbsa_type")

    # Type all molecules with GBSA parameters.
    start_time = time.time()
    typed_molecules = list()
    untyped_molecules = list()
    for cid in database.keys():
        entry = database[cid]
        molecule = entry['molecule']

        if verbose:
            print("  " + molecule.GetTitle())

        # Assign GBSA types according to SMARTS rules.
        try:
            atom_typer.assignTypes(molecule)
            typed_molecules.append(oechem.OEGraphMol(molecule))
        except AtomTyper.TypingException as exception:
            name = molecule.GetTitle()
            print(name)
            print(exception)
            untyped_molecules.append(oechem.OEGraphMol(molecule))
            if len(untyped_molecules) > 10:
                sys.exit(-1)

        # DEBUG: Report types
        if verbose:
            print("Molecule %s : %s" % (cid, entry['iupac']))
            for atom in molecule.GetAtoms():
                print("%5s : %5s" % (atom.GetName(), atom.GetStringData("gbsa_type")))

    if verbose:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("%d molecules correctly typed" % (len(typed_molecules)))
        print("%d molecules missing some types" % (len(untyped_molecules)))
        print("%.3f s elapsed" % elapsed_time)

    return database
