import dgl
import torch
import pandas as pd
import os
import numpy as np
from rdkit import Chem
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
from tqdm import tqdm
from pysmiles import read_smiles


def extract_atom_feature(all_smiles, device):
    fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

    unique_feat = {}
    with open('atom_unique_feature.txt', 'r') as f:
        for line in f:
            atm_feat = line.strip().split('\t')
            unique_feat[atm_feat[0]] = int(atm_feat[1])

    feature_vectors = []
    all_smiles = tqdm(all_smiles)
    for smiles in all_smiles:
        m = Chem.MolFromSmiles(smiles)
        m = Chem.AddHs(m)
        num_atom = m.GetNumAtoms()
        feats = factory.GetFeaturesForMol(m)
        m_vec = torch.zeros([num_atom, len(unique_feat)], device=device)

        for feat in feats:
            # print(unique_feat[feat.GetType()])
            # print(feat.GetAtomIds()[0])
            if feat.GetType() in unique_feat:
                for atomid in feat.GetAtomIds():
                    m_vec[atomid, unique_feat[feat.GetType()]] = 1
        feature_vectors.append(m_vec)
        all_smiles.set_description('Extracting Features ...')

    return feature_vectors


def read_raw(filename, dataset, device):
    if dataset == 'covid-19':
        assertion_len = 3
        smile_idx = 1
        separator = ','
    else:
        assertion_len = 13
        smile_idx = 0
        separator = '\t'

    all_smiles = []
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

                m = Chem.MolFromSmiles(l[smile_idx])
                m = Chem.AddHs(m)
                smiles = Chem.MolToSmiles(m)
                all_smiles.append(smiles)

                targets.append(torch.tensor(int(l[2]), device=device) if smile_idx
                               else torch.tensor([float(i) for i in l[2:]], device=device))
                mol = read_smiles(smiles.replace('[H]', '[G]'), explicit_hydrogen=False, reinterpret_aromatic=True)
                mols.append(mol)

    features = extract_atom_feature(all_smiles, device)

    return mols, targets, features


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
    all_molecules, targets, features = read_raw(filename, dataset, device)
    all_mol_dgl = []
    # print(all_molecules[0].edges(data=True))

    all_molecules = tqdm(all_molecules)
    all_molecules.set_description('Converting to DGL graphs ...')

    for idx, mol in enumerate(all_molecules):
        for node in mol.nodes:
            # set element idx if element is in the dictionary otherwise set to the <unk> idx
            mol.nodes[node]['elem_idx'] = ele2idx[mol.nodes[node]['element']] \
                if mol.nodes[node]['element'] in ele2idx else ele2idx['<unk>']
            try:
                mol.nodes[node]['atom_feat'] = features[idx][node]
                assert len(mol.nodes) == len(features[idx])
            except AssertionError:
                print(idx)
                print(len(features[idx]))
                print(mol.nodes)
                print(features[idx].size())
                exit()

        for edge in mol.edges:
            # set element idx if element is in the dictionary otherwise set to the <unk> idx
            mol.edges[edge]['weight'] = float(mol.edges[edge]['order'])
            mol.edges[edge]['order'] = mol.edges[edge]['order'] if isinstance(mol.edges[edge]['order'], int) else 0

        dgl_mol = dgl.DGLGraph()
        dgl_mol.from_networkx(mol, node_attrs=['elem_idx', 'atom_feat'], edge_attrs=['order', 'weight'])
        all_mol_dgl.append(dgl_mol.to(device))

        # all_mol_idx.append([ele2idx[i[1]] if i[1] in ele2idx else ele2idx['<unk>'] for i in mol.nodes(data='element')])
        # mtx.append(nx.to_numpy_matrix(mol, weight='order'))  # edge weight is defined by order
    data = zip(all_mol_dgl, targets)  # put all info related to each mol together
    return list(data)


if __name__ == '__main__':
    train_data = get_data('train.csv')
    # print(list(train_data))
    # train_data = get_data('gdb_9_clean.tsv', dataset='gdb-9')
    print(train_data)
    # print(extract_atom_feature_no_h('train.csv', 1, ','))
    # m = Chem.MolFromSmiles('CON=C1CN(c2nc3c(cc2F)c(=O)c(C(=O)O)cn3C2CC2)CC1CN')
    # m = Chem.AddHs(m)
    # print(m.GetNumAtoms())
    # print(Chem.MolToSmiles(m))
    # print(list(read_smiles(Chem.MolToSmiles(m), explicit_hydrogen=True, reinterpret_aromatic=True).nodes(data=True)))
    # print(list(read_smiles(Chem.MolToSmiles(m), explicit_hydrogen=True, reinterpret_aromatic=True).neighbors(48)))
    # print(list(read_smiles(Chem.MolToSmiles(m), explicit_hydrogen=True, reinterpret_aromatic=True).neighbors(7)))
    # print(list(read_smiles(Chem.MolToSmiles(m), explicit_hydrogen=True, reinterpret_aromatic=True).neighbors(17)))