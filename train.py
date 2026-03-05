import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Model1'))
from model import EGNNGlobalModel

SEQ_INPUT_DIM = 15 * 2560
PROB_THRESHOLD = 0.5


#FocalLoss
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = targets.long()
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        alpha_t = self.alpha.to(logits.device).gather(0, targets) if self.alpha is not None else 1.0
        loss = -alpha_t * (1 - pt) ** self.gamma * log_pt
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss

def analysis(binary_pred, binary_true, y_pred):
    return {
        'binary_acc': metrics.accuracy_score(binary_true, binary_pred),
        'precision':  metrics.precision_score(binary_true, binary_pred),
        'recall':     metrics.recall_score(binary_true, binary_pred),
        'f1':         metrics.f1_score(binary_true, binary_pred),
        'AUC':        metrics.roc_auc_score(binary_true, y_pred),
        'mcc':        metrics.matthews_corrcoef(binary_true, binary_pred),
    }


def train_fold(train_dataset, valid_dataset, fold, device, output_dir, num_epochs):
    focal_alpha = torch.tensor([0.75, 0.25])
    criterion = FocalLoss(alpha=focal_alpha, gamma=2.0)

    model = EGNNGlobalModel(hidden_channels=64, num_layer=4, num_vn=2, seq_input_dim=SEQ_INPUT_DIM)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader   = DataLoader(valid_dataset, batch_size=16, shuffle=False)

    best_auc   = 0.0
    best_epoch = 0
    best_val   = {}

    def move_to_device(data):
        data.x             = data.x.to(device)
        data.coords        = data.coords.to(device)
        data.batch         = data.batch.to(device)
        data.edge_index[0] = data.edge_index[0].to(device)
        data.edge_index[1] = data.edge_index[1].to(device)
        data.edge_attr     = data.edge_attr.to(device)
        data.y             = data.y.to(device)
        return data

    def trainone(loader):
        model.train()
        losses = []
        for data in loader:
            optimizer.zero_grad()
            data = move_to_device(data)
            seq_feat = data.s.to(device)
            out = model(data.x, data.coords, data.batch, data.edge_index, data.edge_attr, seq_feat)
            loss = criterion(out, data.y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())
        return float(np.mean(losses))

    def evaluate(loader):
        model.eval()
        losses, preds, trues, probs = [], [], [], []
        with torch.no_grad():
            for data in loader:
                data = move_to_device(data)
                seq_feat = data.s.to(device)
                out = model(data.x, data.coords, data.batch, data.edge_index, data.edge_attr, seq_feat)
                loss = criterion(out, data.y)
                losses.append(loss.item())
                prob_pos = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                pred = (prob_pos >= PROB_THRESHOLD).astype(np.int64)
                probs += list(prob_pos)
                preds += list(pred)
                trues += list(data.y.cpu().numpy())
        acc = sum(p == t for p, t in zip(preds, trues)) / len(trues)
        return acc, float(np.mean(losses)), preds, trues, probs

    for epoch in range(1, num_epochs + 1):
        train_loss = trainone(train_loader)
        train_acc, _, _, _, _ = evaluate(train_loader)
        val_acc, val_loss, val_pred, val_true, val_prob = evaluate(val_loader)
        result = analysis(val_pred, val_true, val_prob)

        print(f'Epoch: {epoch:03d}  Train Acc: {train_acc:.4f}  Val Acc: {val_acc:.4f}'
              f'  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}')
        print(f'  Acc={result["binary_acc"]:.4f}  Prec={result["precision"]:.4f}'
              f'  Rec={result["recall"]:.4f}  F1={result["f1"]:.4f}'
              f'  AUC={result["AUC"]:.4f}  MCC={result["mcc"]:.4f}')

        if result['AUC'] > best_auc:
            best_auc   = result['AUC']
            best_epoch = epoch
            best_val   = result
            save_path  = os.path.join(output_dir, f'VNEGNN_Fold{fold}_best_model.pkl')
            torch.save(model.state_dict(), save_path)
            print(f'  → 模型已保存至 {save_path}')

    return best_epoch, best_auc, best_val


#main
def main():
    parser = argparse.ArgumentParser(description='Train EGNNGlobalModel with K-Fold CV')
    parser.add_argument('-i', '--input',      required=True,  help='训练数据集路径 (.pt)')
    parser.add_argument('-o', '--output-dir', default='./Model', help='模型保存目录 (默认: ./Model)')
    parser.add_argument('-e', '--epochs',     type=int, default=30,   help='每折训练轮数 (默认: 30)')
    parser.add_argument('-k', '--folds',      type=int, default=5,    help='K折数 (默认: 5)')
    parser.add_argument('--seed',             type=int, default=42,   help='随机种子 (默认: 42)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    print(f'加载数据集: {args.input}')
    all_pyg_list = torch.load(args.input)
    print(f'样本数: {len(all_pyg_list)}')

    for data in all_pyg_list:
        data.edge_attr  = data.edge_attr.to(torch.float32)
        data.coords     = data.coords.to(torch.float32)
        data.s          = data.s.to(torch.float32)
        data.edge_index = [data.edge_index[0], data.edge_index[1]]

    random.seed(args.seed)
    random.shuffle(all_pyg_list)

    kfold = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    best_epochs, valid_aucs, valid_results = [], [], []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(all_pyg_list)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1} / {args.folds}")
        print(f"{'='*50}")
        train_ds = [all_pyg_list[i] for i in train_idx]
        val_ds   = [all_pyg_list[i] for i in val_idx]

        best_epoch, best_auc, best_val = train_fold(
            train_ds, val_ds, fold, device, args.output_dir, args.epochs
        )
        best_epochs.append(str(best_epoch))
        valid_aucs.append(best_auc)
        valid_results.append(best_val)

    print('\n\n' + '='*50)
    print('训练完成')
    print('Best epoch per fold: ' + ' '.join(best_epochs))
    print('\nValid AUCs:')
    for i, auc in enumerate(valid_aucs):
        print(f'  Fold {i}: {auc:.4f}')
    print(f'\n平均 AUC: {np.mean(valid_aucs):.4f} ± {np.std(valid_aucs):.4f}')
    print('\nBest valid results per fold:')
    for i, r in enumerate(valid_results):
        print(f'  Fold {i}: {r}')


if __name__ == '__main__':
    main()
