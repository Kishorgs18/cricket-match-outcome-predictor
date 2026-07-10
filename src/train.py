"""
train.py
Training loop with early stopping and k-fold cross-validation.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score

from src.model import CricketOutcomeNet


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int = 64, shuffle: bool = True):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, acc, auc, np.array(all_labels), np.array(all_probs)


def train_model(
    X_train, y_train, X_val, y_val,
    input_dim, hidden_dims=None, dropout=0.3,
    epochs=80, lr=1e-3, batch_size=64, patience=12,
    device=None
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CricketOutcomeNet(input_dim, hidden_dims or [128, 64, 32], dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    train_loader = make_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader   = make_loader(X_val,   y_val,   batch_size, shuffle=False)

    best_val_loss = float('inf')
    best_state = None
    no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_auc': []}

    for epoch in range(1, epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc, vl_auc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(vl_loss)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['val_acc'].append(vl_acc)
        history['val_auc'].append(vl_auc)

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | train_loss={tr_loss:.4f} | val_loss={vl_loss:.4f} | val_acc={vl_acc:.4f} | val_auc={vl_auc:.4f}")

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model, history


def cross_validate(X, y, input_dim, n_splits=5, **train_kwargs):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    print(f"\n{'='*50}")
    print(f"  {n_splits}-Fold Stratified Cross Validation")
    print(f"{'='*50}")

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(X, y), 1):
        print(f"\nFold {fold}/{n_splits}")
        model, history = train_model(
            X[tr_idx], y[tr_idx], X[vl_idx], y[vl_idx],
            input_dim=input_dim, **train_kwargs
        )
        device = next(model.parameters()).device
        val_loader = make_loader(X[vl_idx], y[vl_idx], shuffle=False)
        criterion = nn.CrossEntropyLoss()
        _, acc, auc, labels, probs = evaluate(model, val_loader, criterion, device)
        fold_results.append({'fold': fold, 'accuracy': acc, 'auc': auc})
        print(f"  Fold {fold} — Accuracy: {acc:.4f} | AUC: {auc:.4f}")

    accs = [r['accuracy'] for r in fold_results]
    aucs = [r['auc'] for r in fold_results]
    print(f"\n{'='*50}")
    print(f"  CV Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  CV AUC      : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"{'='*50}\n")
    return fold_results
