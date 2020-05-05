import rdkit
from rdkit import Chem
from rdkit.Chem import FragmentCatalog

labels = []
all_molecules = []
with open('train.csv') as f:
    f.readline()
    for line in f:
        if line != '':
            l = line.strip().split(',')

            assert len(l) == 3

            smiles = l[1]
            labels.append(int(l[2]))
            mol = Chem.MolFromSmiles(smiles)
            all_molecules.append(mol)


>>> fName=os.path.join(RDConfig.RDDataDir,'FunctionalGroups.txt')
>>> from rdkit.Chem import FragmentCatalog
>>> fparams = FragmentCatalog.FragCatParams(1,6,fName)
>>> fparams.GetNumFuncGroups()
