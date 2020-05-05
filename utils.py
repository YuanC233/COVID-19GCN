import dgl
import torch
from tqdm import tqdm
from pysmiles import read_smiles


def read_raw(filename, dataset, device):
    if dataset == 'covid-19':
        assertion_len = 3
        smile_idx = 1
        separator = ','
    else:
        assertion_len = 13
        smile_idx = 0
        separator = '\t'

    mols = []
    targets = []
    with open(filename) as f:
        if smile_idx:
            f.readline()
        f = tqdm(f)
        f.set_description('Reading raw data ... ')
        for line in f:
            if line != '':
                l = line.strip().split(separator)

                assert len(l) == assertion_len

                smiles = l[smile_idx]
                targets.append(torch.tensor(int(l[2]), device=device) if smile_idx
                               else torch.tensor([float(i) for i in l[2:]], device=device))
                mol = read_smiles(smiles, explicit_hydrogen=True, reinterpret_aromatic=True)
                mols.append(mol)

                # assign nodes in each molecule an idx according to ele2idx dict, get adjacency matrix for each molecule
    return mols, targets


def get_data(filename, dataset='covid-19', device='cpu'):
    assert dataset.lower() in ['covid-19', 'gdb-9']
    '''
    filename in the format of 'xxx.csv'
    '''
    # load ele2idx: important to have consistent indices!
    # create dictionary
    device = torch.device(device)
    ele2idx = {}
    with open('ele2idx.txt', 'r') as f:
        for line in f:
            l = line.strip().split()
            ele2idx[l[0]] = int(l[1])

    # add unknown element into dictionary
    ele2idx['<unk>'] = len(ele2idx)

    # read data and get structure and label of all molecules
    all_molecules, targets = read_raw(filename, dataset, device)

    all_mol_dgl = []
    # print(all_molecules[0].edges(data=True))

    all_molecules = tqdm(all_molecules)
    all_molecules.set_description('Converting to DGL graphs ...')

    for mol in all_molecules:
        for node in mol.nodes:
            # set element idx if element is in the dictionary otherwise set to the <unk> idx
            mol.nodes[node]['elem_idx'] = ele2idx[mol.nodes[node]['element']] \
                if mol.nodes[node]['element'] in ele2idx else ele2idx['<unk>']

        for edge in mol.edges:
            # set element idx if element is in the dictionary otherwise set to the <unk> idx
            mol.edges[edge]['weight'] = float(mol.edges[edge]['order'])
            mol.edges[edge]['order'] = mol.edges[edge]['order'] if isinstance(mol.edges[edge]['order'], int) else 0

        dgl_mol = dgl.DGLGraph()
        dgl_mol.from_networkx(mol, node_attrs=['elem_idx'], edge_attrs=['order', 'weight'])
        all_mol_dgl.append(dgl_mol.to(device))

        # all_mol_idx.append([ele2idx[i[1]] if i[1] in ele2idx else ele2idx['<unk>'] for i in mol.nodes(data='element')])
        # mtx.append(nx.to_numpy_matrix(mol, weight='order'))  # edge weight is defined by order
    data = zip(all_mol_dgl, targets)  # put all info related to each mol together
    return list(data)


if __name__ == '__main__':
    # train_data = get_data('train.csv')
    # print(list(train_data))
    train_data = get_data('gdb_9_clean.tsv', dataset='gdb-9')
    print(train_data)
