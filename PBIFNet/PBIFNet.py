# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : liushiyun
# @File     : PBIFNet.py
# @Time     : 2025/7/1 11:34
# @Desc     :


############################utils.py#################################
import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             precision_score, recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mask_batch_edges(edge_index, batch_src, batch_dst, max_dst_nodes):
    if len(batch_src) == 0:
        return edge_index
    MAX = max_dst_nodes + 1
    edge_hash = edge_index[0] * MAX + edge_index[1]
    batch_hash = batch_src * MAX + batch_dst
    mask = ~torch.isin(edge_hash, batch_hash)
    return edge_index[:, mask]


def calculate_metrics(y_true, y_scores):
    y_pred_bin = (y_scores >= 0.5).astype(int)

    auc = roc_auc_score(y_true, y_scores)
    aupr = average_precision_score(y_true, y_scores)
    acc = accuracy_score(y_true, y_pred_bin)
    prec = precision_score(y_true, y_pred_bin, zero_division=0)
    rec = recall_score(y_true, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true, y_pred_bin, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred_bin)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'AUC': auc, 'AUPR': aupr, 'Accuracy': acc,
        'Precision': prec, 'Recall': rec, 'F1': f1,
        'MCC': mcc, 'Specificity': spec
    }


#############################model.py####################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv


class DrugConditionedPTMAdaptation(nn.Module):
    """DPA: drug-conditioned PTM adaptation """
    def __init__(self, hidden_dim=128, num_subspaces=2):
        super(DrugConditionedPTMAdaptation, self).__init__()
        self.num_subspaces = num_subspaces
        self.env_proj = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden_dim, hidden_dim * num_subspaces))
        self.drug_query = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, p_feat, d_feat):
        batch_size = p_feat.size(0)
        envs = self.env_proj(p_feat).view(batch_size, self.num_subspaces, -1)
        d_q = self.drug_query(d_feat).unsqueeze(1)
        attn_scores = torch.softmax(torch.sum(envs * d_q, dim=-1), dim=-1)
        selected_p_feat = torch.sum(envs * attn_scores.unsqueeze(-1), dim=1)
        return selected_p_feat, attn_scores


class PTMConditionedDrugAdaptation(nn.Module):
    """PDA: PTM-conditioned drug adaptation """
    def __init__(self, hidden_dim=128, num_subspaces=2):
        super(PTMConditionedDrugAdaptation, self).__init__()
        self.num_subspaces = num_subspaces
        self.pharma_proj = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden_dim, hidden_dim * num_subspaces))
        self.ptm_query = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, d_feat, p_feat):
        batch_size = d_feat.size(0)
        pharmas = self.pharma_proj(d_feat).view(batch_size, self.num_subspaces, -1)
        p_q = self.ptm_query(p_feat).unsqueeze(1)
        attn_scores = torch.softmax(torch.sum(pharmas * p_q, dim=-1), dim=-1)
        selected_d_feat = torch.sum(pharmas * attn_scores.unsqueeze(-1), dim=1)
        return selected_d_feat, attn_scores


class CA_HGT_Model(nn.Module):
    def __init__(self, metadata, K=2, d_in=768, p_in=1280, hidden_dim=128):
        super(CA_HGT_Model, self).__init__()

        self.d_sem_proj = nn.Sequential(nn.Dropout(0.4), nn.Linear(d_in, hidden_dim), nn.LayerNorm(hidden_dim))
        self.p_sem_proj = nn.Sequential(nn.Dropout(0.4), nn.Linear(p_in, hidden_dim), nn.LayerNorm(hidden_dim))

        self.drug_prompt = nn.Parameter(torch.empty(1, hidden_dim))
        self.ptm_prompt = nn.Parameter(torch.empty(1, hidden_dim))
        nn.init.xavier_uniform_(self.drug_prompt)
        nn.init.xavier_uniform_(self.ptm_prompt)

        self.hgt = HGTConv(hidden_dim, hidden_dim, metadata, heads=16)

        self.dpa = DrugConditionedPTMAdaptation(hidden_dim=hidden_dim, num_subspaces=K)
        self.pda = PTMConditionedDrugAdaptation(hidden_dim=hidden_dim, num_subspaces=K)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        self.alpha = nn.Parameter(torch.tensor([0.0]))
        self.tau = nn.Parameter(torch.tensor([10.0]))

    def forward(self, x_dict, edge_index_dict, batch_d_idx, batch_p_idx):
        d_anchor = self.d_sem_proj(x_dict['drug'])
        p_anchor = self.p_sem_proj(x_dict['ptm'])

        d_prompted = d_anchor + self.drug_prompt
        p_prompted = p_anchor + self.ptm_prompt

        h_graph = self.hgt({'drug': d_prompted, 'ptm': p_prompted}, edge_index_dict)

        D_raw = h_graph['drug'][batch_d_idx]
        P_raw = h_graph['ptm'][batch_p_idx]

        d_pda_out, pda_attn = self.pda(D_raw, P_raw)
        p_dpa_out, dpa_attn = self.dpa(P_raw, D_raw)

        D_final = d_pda_out + D_raw
        P_final = p_dpa_out + P_raw

        interaction = D_final * P_final
        mlp_logits = self.classifier(interaction).squeeze(-1)

        d_norm = F.normalize(D_final, p=2, dim=1)
        p_norm = F.normalize(P_final, p=2, dim=1)
        cos_logits = (d_norm * p_norm).sum(dim=1) * self.tau

        weight = torch.sigmoid(self.alpha)
        logits = (weight * mlp_logits) + ((1.0 - weight) * cos_logits)

        if not self.training:
            return logits, dpa_attn, pda_attn
        return logits


############################datautils.py#################################
import pandas as pd
import torch
from torch.utils.data import Dataset


class DPPA_Dataset(Dataset):
    def __init__(self, csv_file, d2idx, ptm2idx):
        self.data = pd.read_csv(csv_file)
        self.d2idx, self.p2idx = d2idx, ptm2idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return torch.tensor(self.d2idx[str(row['Drug'])], dtype=torch.long), \
            torch.tensor(self.p2idx[str(row['PTM'])], dtype=torch.long), \
            torch.tensor(float(row['Label']), dtype=torch.float32)


##########################trainer.py#######################################
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch_geometric.utils import dropout_edge
from sklearn.metrics import roc_auc_score


class PBIFNetRunner:
    def __init__(self, clean_dir, split_dir, K=2, batch_size=128, lr=1e-4,
                 weight_decay=5e-4, hidden_dim=128, max_epochs=80):
        self.clean_dir = clean_dir
        self.split_dir = split_dir
        self.K = K
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.hidden_dim = hidden_dim
        self.max_epochs = max_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        drug_list = pd.read_csv(os.path.join(self.clean_dir, 'Drug_SMILES.csv')).iloc[:, 0].astype(str).tolist()
        self.drug2idx = {d: i for i, d in enumerate(drug_list)}
        ptm_list = pd.read_csv(os.path.join(self.clean_dir, 'PTM_peptide.csv')).iloc[:, 0].astype(str).tolist()
        self.ptm2idx = {p: i for i, p in enumerate(ptm_list)}

    def run_train(self, fold):
        graph = torch.load(os.path.join(self.split_dir, f'Fold{fold}_HeteroGraph.pt'), weights_only=False).to(self.device)
        train_loader = DataLoader(DPPA_Dataset(os.path.join(self.split_dir, f'Train_dataset_Fold{fold}.csv'), self.drug2idx, self.ptm2idx),
                                  batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(DPPA_Dataset(os.path.join(self.split_dir, f'Val_dataset_Fold{fold}.csv'), self.drug2idx, self.ptm2idx),
                                batch_size=self.batch_size, num_workers=0, pin_memory=True)

        base_edge_index_dict = graph.edge_index_dict
        num_drugs, num_ptms = graph['drug'].x.shape[0], graph['ptm'].x.shape[0]

        model = CA_HGT_Model(graph.metadata(), K=self.K, hidden_dim=self.hidden_dim).to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=1e-6)

        best_val_auc, patience_counter = 0.0, 0

        for epoch in range(1, self.max_epochs + 1):
            model.train()
            for d, p, l in train_loader:
                d, p, l = d.to(self.device), p.to(self.device), l.to(self.device)
                optimizer.zero_grad()
                pos_mask = (l == 1.0)
                batch_pos_d, batch_pos_p = d[pos_mask], p[pos_mask]

                batch_edge_index_dict = {}
                for edge_type, edge_index in base_edge_index_dict.items():
                    rel = edge_type[1]
                    safe_edge = mask_batch_edges(edge_index, batch_pos_d, batch_pos_p, num_ptms) if rel == 'interacts_with' else (
                        mask_batch_edges(edge_index, batch_pos_p, batch_pos_d, num_drugs) if rel == 'interacted_by' else edge_index)
                    drop_prob = 0.2 if rel in ['interacts_with', 'interacted_by', 'co_perturbed_with'] else 0.1
                    batch_edge_index_dict[edge_type] = dropout_edge(safe_edge, p=drop_prob, training=True)[0]

                logits = model(graph.x_dict, batch_edge_index_dict, d, p)
                loss = F.binary_cross_entropy_with_logits(logits, l)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            model.eval()
            v_preds, v_labels = [], []
            with torch.no_grad():
                for d, p, l in val_loader:
                    d, p, l = d.to(self.device), p.to(self.device), l.to(self.device)
                    out = model(graph.x_dict, base_edge_index_dict, d, p)
                    logits = out[0] if isinstance(out, tuple) else out
                    v_preds.extend(torch.sigmoid(logits).cpu().numpy())
                    v_labels.extend(l.cpu().numpy())

            v_auc = roc_auc_score(v_labels, v_preds)
            scheduler.step()

            if v_auc > best_val_auc:
                best_val_auc = v_auc
                patience_counter = 0
                torch.save(model.state_dict(), f"best_sota_fold{fold}.pth")
            else:
                patience_counter += 1
                if patience_counter >= 15:
                    break

    def evaluate_on_test_set(self, test_csv_path, fold):
        model_path = f"best_sota_fold{fold}.pth"
        if not os.path.exists(model_path):
            return None

        test_loader = DataLoader(DPPA_Dataset(test_csv_path, self.drug2idx, self.ptm2idx), batch_size=self.batch_size, shuffle=False,
                                 num_workers=0, pin_memory=True)
        graph = torch.load(os.path.join(self.split_dir, f'Fold{fold}_HeteroGraph.pt'), weights_only=False).to(self.device)

        model = CA_HGT_Model(graph.metadata(), K=self.K, hidden_dim=self.hidden_dim).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()

        test_preds, test_labels = [], []
        with torch.no_grad():
            for d, p, l in test_loader:
                d, p, l = d.to(self.device), p.to(self.device), l.to(self.device)
                out = model(graph.x_dict, graph.edge_index_dict, d, p)
                logits = out[0] if isinstance(out, tuple) else out

                test_preds.extend(torch.sigmoid(logits).cpu().numpy())
                test_labels.extend(l.cpu().numpy())

        y_true = np.array(test_labels)
        y_scores = np.array(test_preds)

        df_plot = pd.DataFrame({
            'Fold': [f'Fold {fold}'] * len(y_true),
            'Label': y_true,
            'Pred_Score': y_scores
        })
        plot_csv_path = 'PBIFNet_ROC_PR_Data.csv'
        write_header = not os.path.exists(plot_csv_path)
        df_plot.to_csv(plot_csv_path, mode='a', header=write_header, index=False)

        res = calculate_metrics(y_true, y_scores)
        res['Fold'] = fold
        return res


##############################################################################
import time
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="PBIFNet: Prompt-Guided Bi-directional Induced Fit Network for Drug-PTM Association Prediction"
    )

    parser.add_argument('--clean_dir', type=str, default='/public/home/wangb/liushy/project/project/DPPA/data/Clean_data/',
                        help="Path to clean feature files directory.")
    parser.add_argument('--split_dir', type=str, default='/public/home/wangb/liushy/project/project/DPPA/data/split_data_no_cold',
                        help="Path to split dataset directory.")
    parser.add_argument('-k', '--K', type=int, default=2,
                        help="Number of latent subspaces (shared by DPA and PDA). Default is 2.")
    parser.add_argument('-b', '--batch_size', type=int, default=128,
                        help="Batch size for training and evaluation. Default is 128.")
    parser.add_argument('-e', '--epochs', type=int, default=80,
                        help="Maximum training epochs per fold. Default is 80.")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help="Learning rate. Default is 1e-4.")

    args = parser.parse_args()

    seed_everything(42)

    if os.path.exists('PBIFNet_ROC_PR_Data.csv'):
        os.remove('PBIFNet_ROC_PR_Data.csv')

    runner = PBIFNetRunner(
        clean_dir=args.clean_dir,
        split_dir=args.split_dir,
        K=args.K,
        batch_size=args.batch_size,
        lr=args.lr,
        max_epochs=args.epochs
    )

    for f in range(1, 6):
        runner.run_train(fold=f)

    test_csv = os.path.join(args.split_dir, 'Test_dataset.csv')
    fold_results = []

    for f in range(1, 6):
        res = runner.evaluate_on_test_set(test_csv, fold=f)
        if res is not None:
            fold_results.append(res)

    if len(fold_results) > 0:
        metrics_keys = ['AUC', 'AUPR', 'Accuracy', 'Precision', 'Recall', 'F1', 'MCC', 'Specificity']
        avg_res = {k: np.mean([r[k] for r in fold_results]) for k in metrics_keys}
        std_res = {k: np.std([r[k] for r in fold_results]) for k in metrics_keys}

        print(f"{'Fold':<6} {'AUC':<10} {'AUPR':<10} {'accuracy':<10} {'precision':<10} {'recall':<10} {'F1-score':<10} {'MCC':<10} {'specificity':<12}")
        print("-" * 100)
        for r in fold_results:
            print(f"{r['Fold']:<6} {r['AUC']:.4f}     {r['AUPR']:.4f}     {r['Accuracy']:.4f}     {r['Precision']:.4f}      {r['Recall']:.4f}     {r['F1']:.4f}       {r['MCC']:.4f}     {r['Specificity']:.4f}")
        print("-" * 100)
        print(f"{'avg':<6} {avg_res['AUC']:.4f}     {avg_res['AUPR']:.4f}     {avg_res['Accuracy']:.4f}     {avg_res['Precision']:.4f}      {avg_res['Recall']:.4f}     {avg_res['F1']:.4f}       {avg_res['MCC']:.4f}     {avg_res['Specificity']:.4f}")
        print(f"{'std':<6} {std_res['AUC']:.4f}     {std_res['AUPR']:.4f}     {std_res['Accuracy']:.4f}     {std_res['Precision']:.4f}      {std_res['Recall']:.4f}     {std_res['F1']:.4f}       {std_res['MCC']:.4f}     {std_res['Specificity']:.4f}")

        res_file = 'sota_k_full_metrics_results.csv'
        write_header = not os.path.exists(res_file)
        with open(res_file, 'a') as f_out:
            if write_header:
                f_out.write("K,Fold,AUC,AUPR,Accuracy,Precision,Recall,F1_score,MCC,Specificity\n")

            for r in fold_results:
                f_out.write(f"{args.K},{r['Fold']},{r['AUC']:.4f},{r['AUPR']:.4f},{r['Accuracy']:.4f},{r['Precision']:.4f},{r['Recall']:.4f},{r['F1']:.4f},{r['MCC']:.4f},{r['Specificity']:.4f}\n")

            f_out.write(f"{args.K},avg,{avg_res['AUC']:.4f},{avg_res['AUPR']:.4f},{avg_res['Accuracy']:.4f},{avg_res['Precision']:.4f},{avg_res['Recall']:.4f},{avg_res['F1']:.4f},{avg_res['MCC']:.4f},{avg_res['Specificity']:.4f}\n")
            f_out.write(f"{args.K},std,{std_res['AUC']:.4f},{std_res['AUPR']:.4f},{std_res['Accuracy']:.4f},{std_res['Precision']:.4f},{std_res['Recall']:.4f},{std_res['F1']:.4f},{std_res['MCC']:.4f},{std_res['Specificity']:.4f}\n")


if __name__ == '__main__':
    t0 = time.time()
    main()
    print('using time: {} m {:.2f}s'.format(int((time.time() - t0) // 60), (time.time() - t0) % 60))
