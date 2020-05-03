from pysmiles import read_smiles
import networkx as nx


# %%

def get_data(filename):
    '''
    filename in the format of 'xxx.csv'
    '''
    # load ele2idx: important to have consistent indices!
    # create dictionary
    ele2idx = {}
    with open('ele2idx.txt', 'r') as f:
        for line in f:
            l = line.strip().split()
            ele2idx[l[0]] = int(l[1])

    # add unknown element into dictionary
    ele2idx['<unk>'] = len(ele2idx)

    # read data and get structure and label of all molecules
    all_molecules = []  # list containing all molecules
    labels = []

    with open(filename) as f:
        f.readline()
        for line in f:
            if line != '':
                l = line.strip().split(',')

                assert len(l) == 3

                smiles = l[1]
                labels.append(l[2])
                mol = read_smiles(smiles, explicit_hydrogen=True, reinterpret_aromatic=True)
                all_molecules.append(mol)
                # assign nodes in each molecule an idx according to ele2idx dict, get adjacency matrix for each molecule

    all_mol_idx = []
    mtx = []

    for mol in all_molecules:
        # set element idx if element is in the dictionary otherwise set to the <unk> idx
        all_mol_idx.append([ele2idx[i[1]] if i[1] in ele2idx else ele2idx['<unk>'] for i in mol.nodes(data='element')])
        mtx.append(nx.to_numpy_matrix(mol, weight='order'))  # edge weight is defined by order

    data = zip(all_mol_idx, mtx, labels)  # put all info related to each mol together
    return data


if __name__ == '__main__':
    train_data = get_data('train.csv')
    print(list(train_data))
