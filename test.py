import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
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


#评估指标
def analysis(binary_pred, binary_true, y_pred):
    return {
        'binary_acc': metrics.accuracy_score(binary_true, binary_pred),
        'precision':  metrics.precision_score(binary_true, binary_pred),
        'recall':     metrics.recall_score(binary_true, binary_pred),
        'f1':         metrics.f1_score(binary_true, binary_pred),
        'AUC':        metrics.roc_auc_score(binary_true, y_pred),
        'mcc':        metrics.matthews_corrcoef(binary_true, binary_pred),
    }


def run_test(model, loader, device):
    focal_alpha = torch.tensor([0.75, 0.25])
    criterion = FocalLoss(alpha=focal_alpha, gamma=2.0)

    model.eval()
    losses, preds, trues, probs = [], [], [], []

    with torch.no_grad():
        for data in loader:
            data.x             = data.x.to(device)
            data.coords        = data.coords.to(device)
            data.batch         = data.batch.to(device)
            data.edge_index[0] = data.edge_index[0].to(device)
            data.edge_index[1] = data.edge_index[1].to(device)
            data.edge_attr     = data.edge_attr.to(device)
            data.y             = data.y.to(device)
            seq_feat           = data.s.to(device)

            out = model(data.x, data.coords, data.batch, data.edge_index, data.edge_attr, seq_feat)
            loss = criterion(out, data.y)
            losses.append(loss.item())

            prob_pos = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            pred = (prob_pos >= PROB_THRESHOLD).astype(np.int64)
            probs += list(prob_pos)
            preds += list(pred)
            trues += list(data.y.cpu().numpy())

    return float(np.mean(losses)), preds, trues, probs


# main
def main():
    parser = argparse.ArgumentParser(description='Test EGNNGlobalModel on independent test set')
    parser.add_argument('-i', '--input', required=True,
                        help='测试数据集路径 (.pt)')
    parser.add_argument('-m', '--model', default='VNEGNN_Fold4_best_model.pkl',
                        help='模型权重路径 (默认: VNEGNN_Fold4_best_model.pkl)')
    parser.add_argument('-b', '--batch-size', type=int, default=16,
                        help='批大小 (默认: 16)')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    print(f'加载数据集: {args.input}')
    test_dataset = torch.load(args.input)
    print(f'测试样本数: {len(test_dataset)}')

    for data in test_dataset:
        data.edge_attr  = data.edge_attr.to(torch.float32)
        data.coords     = data.coords.to(torch.float32)
        data.s          = data.s.to(torch.float32)
        data.edge_index = [data.edge_index[0], data.edge_index[1]]

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f'加载模型: {args.model}')
    model = EGNNGlobalModel(hidden_channels=64, num_layer=4, num_vn=2, seq_input_dim=SEQ_INPUT_DIM)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.to(device)

    loss, preds, trues, probs = run_test(model, test_loader, device)
    result = analysis(preds, trues, probs)

    print('\n========== Evaluate Test Set ==========')
    print(f'Test Loss:     {loss:.4f}')
    print(f'Accuracy:      {result["binary_acc"]:.4f}')
    print(f'Precision:     {result["precision"]:.4f}')
    print(f'Recall:        {result["recall"]:.4f}')
    print(f'F1:            {result["f1"]:.4f}')
    print(f'AUC:           {result["AUC"]:.4f}')
    print(f'MCC:           {result["mcc"]:.4f}')


if __name__ == '__main__':
    main()
