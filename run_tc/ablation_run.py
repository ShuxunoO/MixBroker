# coding=utf-8
"""Unified ablation runner. Verbatim copy of main_cpu.train_10_fold / mytest
logic (seed 1029, KFold-10 rs=42, hidden 32->16, Adam lr0.01, BCEWithLogits,
100 epochs, min_epochs 10, threshold 0.75, best-by-F1) with TWO things made
configurable, nothing else:
  --backbone {gcn,gat,sage}     encoder operator (the repo's model.py ablation)
  --feat <csv> --drop <cols>    feature file + columns to drop (42d vs 82d)
"""
import argparse, json, os, random, time
import numpy as np
import pandas as pd
import torch
import sklearn.metrics as sm
from torch_geometric.data import Data
from model_ablation import GNN_NET

P = argparse.ArgumentParser()
P.add_argument('--backbone', required=True, choices=['gcn', 'gat', 'sage'])
P.add_argument('--feat', required=True)
P.add_argument('--drop', required=True, help='json list of columns to drop')
P.add_argument('--pos', default='./Dataset/Graph/train_pos_edge_10fold.csv')
P.add_argument('--neg', default='./Dataset/Graph/train_neg_edge_10fold.csv')
P.add_argument('--out', required=True, help='output prefix')
args = P.parse_args()
DROP = json.loads(args.drop)


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed = seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print('Success!')


def load_10_fold_data(t_edge):
    t_data = pd.read_csv(args.feat)
    t_data = t_data.drop(columns=DROP)
    names = t_data.columns.tolist()
    names = np.delete(names, 0)
    texId2index = {}
    for index, row in t_data.iterrows():
        texId2index[int(row.iloc[0])] = index
    x = t_data.iloc[:, 1:]
    x = x.reset_index(drop=True)
    x = x.to_numpy().astype(np.float32)
    x[x == np.inf] = 1.
    x[np.isnan(x)] = 0.
    edges = []
    labels = []
    for _, row in t_edge.iterrows():
        id_1, id_2 = int(row.iloc[0]), int(row.iloc[1])
        label = int(row.iloc[2])
        if id_1 not in texId2index or id_2 not in texId2index:
            continue
        edges.append((texId2index[id_1], texId2index[id_2]))
        labels.append(label)
    x = torch.tensor(x, dtype=torch.float32)
    d_edges = np.array(edges)
    edges = torch.tensor(d_edges.T, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.float32)
    data = Data(x=x, edge_index=edges, edge_label=labels, edge_label_index=edges)
    return data, names


def mytest(model, data):
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
        model.train()
    label_test = data.edge_label.cpu().numpy()
    out_np = out.cpu().numpy()
    label_pred = np.where(out_np > 0.75, 1, 0)
    TP, FP, FN, TN = 0, 0, 0, 0
    for i in range(len(label_test)):
        if label_test[i] == 1 and label_pred[i] == 1:
            TP += 1
        elif label_test[i] == 1 and label_pred[i] == 0:
            FN += 1
        elif label_test[i] == 0 and label_pred[i] == 1:
            FP += 1
        else:
            TN += 1
    FPR = FP / (FP + TN)
    FNR = FN / (TP + FN)
    accuracy = sm.accuracy_score(label_test, label_pred)
    precision = sm.precision_score(label_test, label_pred)
    recall = sm.recall_score(label_test, label_pred)
    f_1 = sm.f1_score(label_test, label_pred)
    roc_auc = sm.roc_auc_score(label_test, out_np)
    return accuracy, precision, recall, f_1, FPR, FNR, roc_auc


def train_10_fold():
    start = time.perf_counter()
    df_pos_data = pd.read_csv(args.pos)
    df_neg_data = pd.read_csv(args.neg)
    arr = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device, 'backbone=', args.backbone, 'feat=', args.feat)
    from sklearn.model_selection import KFold
    skf = KFold(n_splits=10, shuffle=True, random_state=42)
    neg_idx_train, neg_idx_test = [], []
    for fold, (tr, te) in enumerate(skf.split(df_neg_data)):
        neg_idx_train.append(tr)
        neg_idx_test.append(te)
    for fold, (train_pos_idx, test_pos_idx) in enumerate(skf.split(df_pos_data)):
        print('**' * 10, 'The', fold + 1, 'Fold', '**' * 10)
        df_train = pd.concat([df_pos_data.iloc[train_pos_idx], df_neg_data.iloc[neg_idx_train[fold]]]).reset_index(drop=True).drop(columns=['Unnamed: 0'])
        train_data, _ = load_10_fold_data(df_train)
        train_data = train_data.to(device)
        df_test = pd.concat([df_pos_data.iloc[test_pos_idx], df_neg_data.iloc[neg_idx_test[fold]]]).reset_index(drop=True).drop(columns=['Unnamed: 0'])
        test_data, _ = load_10_fold_data(df_test)
        test_data = test_data.to(device)
        model = GNN_NET(train_data.num_features, 32, 16, backbone=args.backbone).to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
        criterion = torch.nn.BCEWithLogitsLoss()
        min_epochs = 10
        best = [0, 0, 0, 0, 0, 0, 0]
        best_epoch = 0
        model.train()
        for epoch in range(1, 101):
            optimizer.zero_grad()
            out = model(train_data.x, train_data.edge_index, train_data.edge_label_index).view(-1)
            loss = criterion(out, train_data.edge_label)
            loss.backward()
            optimizer.step()
            acc, prec, rec, f1, fpr, fnr, auc = mytest(model, test_data)
            if epoch > min_epochs and f1 > best[3]:
                best_epoch = epoch
                best = [acc, prec, rec, f1, fpr, fnr, auc]
        print('best_epoch {:03d} best_test_f_1 {:.4f}'.format(best_epoch, best[3]))
        arr.append(best)
    cols = ['test_accuracy', 'test_precision', 'test_recall', 'test_f_1', 'test_FPR', 'test_FNR', 'test_auc']
    df = pd.DataFrame(arr, columns=cols).astype(float)
    df.insert(0, 'fold', range(1, len(df) + 1))
    df.to_csv(args.out + '_per_fold.csv', index=False)
    summary = {c: {'mean': float(df[c].mean()), 'std': float(df[c].std())} for c in cols}
    summary['_meta'] = {'backbone': args.backbone, 'feat': os.path.basename(args.feat),
                        'n_features': int(train_data.num_features), 'time_sec': time.perf_counter() - start}
    json.dump(summary, open(args.out + '_summary.json', 'w'), indent=2)
    print('MEANS', {c: round(summary[c]['mean'], 4) for c in cols})
    print('time', round(time.perf_counter() - start, 1), 's')
    return df


if __name__ == '__main__':
    seed_torch(1029)
    train_10_fold()
