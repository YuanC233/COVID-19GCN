import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.nn as nn
import os
import itertools
from model import GCN, MLP
from torch.utils.data import DataLoader
from gcn_utils import collate
from utils import get_data, get_data_for_mlp
from tqdm import tqdm
from evaluate_roc_auc import evaluate_auc

PATH = './train_cv/'
RUNS = 3

NO_H = False
NUM_ELEM = 27 + 1
EMBEDDING_DIM = 16
HIDDEN_DIM = 24
HIDDEN_DIM_ADD_ON = 8
EDGE_HIDDEN_DIM = 6
NUM_CLS = 2
NUM_LYS = 2
ADD_ON_FEATS = 0 # 9

ALL_BATCH_SIZE = [16, 32, 48]
ALL_DROP_RT = [0.1,  0.5]
ALL_LR = [0.01, 0.001]
ALL_EPOCHS = [40]
HYPER_PARA = list(itertools.product(ALL_BATCH_SIZE, ALL_DROP_RT, ALL_LR, ALL_EPOCHS))
HYPER_PARAM_SIZE = len(HYPER_PARA)
PATIENCE = 3

if __name__ == '__main__':
    folds = os.listdir(PATH)
    all_tst_roc = []
    all_tst_prc = []
    for fold in range(len(folds)):
        train_data = get_data_for_mlp(f'train_cv/fold_{fold}/train.csv', device='cuda:0', no_h=NO_H)
        dev_data = get_data_for_mlp(f'train_cv/fold_{fold}/dev.csv', device='cuda:0', no_h=NO_H)
        test_data = get_data_for_mlp(f'train_cv/fold_{fold}/dev.csv', device='cuda:0', no_h=NO_H)

        best_performance = (-1, -1)
        best_params = None
        best_model = None

        _dev_data_loader = DataLoader(dev_data, batch_size=32, shuffle=False)

        for idx, params in enumerate(HYPER_PARA):
            BATCH_SIZE, DROP_RT, LR, EPOCHS = params
            best_roc = []
            best_prc = []

            data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

            for run in range(1):
                model = MLP(NUM_ELEM, EMBEDDING_DIM,  HIDDEN_DIM_ADD_ON, HIDDEN_DIM, NUM_CLS, NUM_LYS, ADD_ON_FEATS, 100,
                            DROP_RT)
                loss_func = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=LR)

                device = torch.device('cuda:0')
                model.to(device)

                last_roc = -1
                last_prc = -1
                epochs_no_imprv = 0
                for epoch in range(EPOCHS):
                    model.train()
                    epoch_loss = 0
                    batch = tqdm(data_loader)
                    for elem, label, lengths, feats in batch:
                        optimizer.zero_grad()
                        prediction = model(elem, lengths, feats)
                        # loss = torch.mean(F.cross_entropy(prediction, label, reduction='none')
                        #                   * (torch.ones_like(label, dtype=torch.float, device=device) +
                        #                      torch.tensor(label.clone().detach(), dtype=torch.float,
                        #                                   device=device) * 8))
                        loss = loss_func(prediction, label)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.detach().item()
                        batch.set_description(f'fold {fold} Trail {idx} / {HYPER_PARAM_SIZE} epoch {epoch} loss {epoch_loss}')

                    if epoch > 19:
                        model.eval()
                        epoch_loss = 0
                        dev_batch = tqdm(_dev_data_loader)
                        all_pred = []
                        all_label = []
                        with torch.no_grad():
                            for elem, label, lengths, feats in dev_batch:
                                prediction = model(elem, lengths, feats)
                                all_pred += prediction.clone().detach()[:, 1].tolist()
                                all_label += label.clone().detach().tolist()
                                dev_batch.set_description(f'-DEV- epoch {epoch}')
                        roc, prc = evaluate_auc(all_label, all_pred)
                        print(f'-DEV- epoch {epoch} - ROC: {roc} PRC: {prc}')

                        if roc + prc < last_roc + last_prc:
                            epochs_no_imprv += 1
                        else:
                            epochs_no_imprv = 0
                        last_roc = roc
                        last_prc = prc

                    if last_roc > 0 and epochs_no_imprv >= PATIENCE:
                        break
                best_roc.append(last_roc)
                best_prc.append(last_prc)

            avg_roc = sum(best_roc) / 1
            avg_prc = sum(best_prc) / 1

            print(f'fold {fold} Params : [batch_size : {BATCH_SIZE}, drop_rate : {DROP_RT}, LR: {LR}, EPOCHS: {epoch}]')
            print(f'final roc : {avg_roc} --- final prc : {avg_prc}')

            if sum(best_performance) < avg_roc + avg_prc:
                print('<***> best model sets to the current model <***>')
                best_performance = avg_roc, avg_prc
                # best_model = model
                best_params = BATCH_SIZE, DROP_RT, LR, EPOCHS

        print(f'--- fold {fold} --- BEST')
        print(f'--- Params : {best_params}')
        print(f'--- fold {fold} --- TEST BEGINS')

        _test_data_loader = DataLoader(dev_data, batch_size=32, shuffle=False)

        test_roc = []
        test_prc = []

        BATCH_SIZE, DROP_RT, LR, EPOCHS = best_params
        data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

        for run in range(RUNS):
            model = MLP(NUM_ELEM, EMBEDDING_DIM, HIDDEN_DIM_ADD_ON, HIDDEN_DIM, NUM_CLS, NUM_LYS, ADD_ON_FEATS, 100,
                        DROP_RT)

            loss_func = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=LR)

            device = torch.device('cuda:0')
            model.to(device)

            last_roc = -1
            last_prc = -1
            epochs_no_imprv = 0
            for epoch in range(EPOCHS):
                model.train()
                epoch_loss = 0
                batch = tqdm(data_loader)
                for elem, label, lengths, feats in batch:
                    optimizer.zero_grad()
                    prediction = model(elem, lengths, feats)
                    # loss = torch.mean(F.cross_entropy(prediction, label, reduction='none')
                    #                   * (torch.ones_like(label, dtype=torch.float, device=device) +
                    #                      torch.tensor(label.clone().detach(), dtype=torch.float,
                    #                                   device=device) * 8))
                    loss = loss_func(prediction, label)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.detach().item()
                    batch.set_description(
                        f'fold {fold} Verf {run} / {RUNS} epoch {epoch} loss {epoch_loss}')

                if epoch > 19:
                    model.eval()
                    epoch_loss = 0
                    dev_batch = tqdm(_dev_data_loader)
                    all_pred = []
                    all_label = []
                    with torch.no_grad():
                        for elem, label, lengths, feats in dev_batch:
                            prediction = model(elem, lengths, feats)
                            all_pred += prediction.clone().detach()[:, 1].tolist()
                            all_label += label.clone().detach().tolist()
                            dev_batch.set_description(f'-DEV- epoch {epoch}')
                    roc, prc = evaluate_auc(all_label, all_pred)
                    print(f'-DEV- epoch {epoch} - ROC: {roc} PRC: {prc}')

                    if roc + prc < last_roc + last_prc:
                        epochs_no_imprv += 1
                    else:
                        epochs_no_imprv = 0
                    last_roc = roc
                    last_prc = prc

                if last_roc > 0 and epochs_no_imprv >= PATIENCE:
                    break

            model.eval()
            test_batch = tqdm(_test_data_loader)
            all_pred = []
            all_label = []
            with torch.no_grad():
                for elem, label, lengths, feats in test_batch:
                    prediction = model(elem, lengths, feats)
                    all_pred += prediction.clone().detach()[:, 1].tolist()
                    all_label += label.clone().detach().tolist()
                    test_batch.set_description(f'-TEST-')
            roc, prc = evaluate_auc(all_label, all_pred)
            test_roc.append(roc)
            test_prc.append(prc)

        avg_tst_roc = sum(test_roc) / RUNS
        avg_tst_prc = sum(test_prc) / RUNS

        print(f'--- fold {fold} --- BEST --- TEST ROC : {avg_tst_roc} TEST PRC : {avg_tst_prc}')

        all_tst_roc.append(avg_tst_roc)
        all_tst_prc.append(avg_tst_prc)

    print(sum(all_tst_roc)/len(all_tst_roc))
    print(sum(all_tst_prc)/len(all_tst_prc))
