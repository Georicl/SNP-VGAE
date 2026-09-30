"""Small GRM ridge/BLUP helpers with disjoint validation tuning."""
import numpy as np

def blup_predict(grm, labels, train_mask, test_mask, lam):
    train_idx = np.flatnonzero(np.asarray(train_mask, dtype=bool))
    test_idx = np.flatnonzero(np.asarray(test_mask, dtype=bool))
    if not len(train_idx) or not len(test_idx):
        raise ValueError("BLUP requires nonempty fitting and prediction sets")
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("BLUP lambda must be finite and positive")
    y_train = labels[train_idx]
    mu = y_train.mean()
    K_train = grm[np.ix_(train_idx, train_idx)]
    K_test = grm[np.ix_(test_idx, train_idx)]
    alpha = np.linalg.solve(K_train + lam * np.eye(len(train_idx)), y_train - mu)
    return K_test @ alpha + mu

def select_blup_lambda(grm, labels, train_mask, val_mask, lambda_grid):
    train_mask = np.asarray(train_mask, dtype=bool)
    val_mask = np.asarray(val_mask, dtype=bool)
    if train_mask.shape != (len(labels),) or val_mask.shape != (len(labels),):
        raise ValueError("BLUP masks must have one entry per label")
    if np.any(train_mask & val_mask):
        raise ValueError("BLUP fitting and validation samples must be disjoint")
    if not len(lambda_grid):
        raise ValueError("BLUP lambda grid must not be empty")
    losses = []
    for lam in lambda_grid:
        pred = blup_predict(grm, labels, train_mask, val_mask, lam)
        loss = float(np.mean((pred - labels[val_mask]) ** 2))
        if not np.isfinite(loss):
            raise ValueError("BLUP validation loss must be finite")
        losses.append(loss)
    return lambda_grid[int(np.argmin(losses))]
