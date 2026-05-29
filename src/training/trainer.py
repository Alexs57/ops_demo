"""Supervised training + evaluation + test-time mask injection utilities.

Kept deliberately small. Day 1 only needs: train a classifier on clean data,
evaluate on (possibly mask-degraded) test data, return accuracy.
"""

from __future__ import annotations

import copy
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset


def _to_loader(
    x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool
) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


class MaskAugDataset(Dataset):
    """Apply a freshly sampled observation mask to each sample on access.

    Used at training time to inject observation-pattern augmentation. Every
    __getitem__ call generates a new mask via ``mask_fn(sample_shape, rng)``,
    so across epochs each sample sees many mask realizations.

    With ``concat_mask=True`` returns ``[X⊙M, M]`` along the channel axis
    (shape doubles from C to 2C) for mask-aware models.

    Use with ``num_workers=0`` in DataLoader so the internal RNG is shared
    across iterations (subprocess workers would each fork the RNG and produce
    correlated masks).
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        mask_fn: Callable[[Tuple[int, ...], np.random.Generator], np.ndarray],
        concat_mask: bool = False,
    ):
        self.x = x
        self.y = y
        self.mask_fn = mask_fn
        self.concat_mask = concat_mask
        self.sample_shape = x.shape[1:]
        self._rng = np.random.default_rng()

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int):
        mask = self.mask_fn(self.sample_shape, self._rng)
        x_masked = (self.x[i] * mask).astype(np.float32)
        if self.concat_mask:
            inp = np.concatenate([x_masked, mask.astype(np.float32)], axis=0)
            return torch.from_numpy(inp), int(self.y[i])
        return torch.from_numpy(x_masked), int(self.y[i])


class MaskAugTwoViewDataset(Dataset):
    """Yields two masked views of each sample on access.

    Used at training time for two-view consistency objectives. Returns
    ``(v1, v2, y)``. With ``concat_mask=True`` each view is ``[X⊙M, M]``
    so shape is (2C, T); otherwise just ``X⊙M`` with shape (C, T).

    Symmetric mode (``weak_mask_fn=None``, the default): both views are
    independent draws from ``mask_fn``. This matches R-Drop-style symmetric
    consistency training.

    Asymmetric mode (``weak_mask_fn`` provided): v1 is sampled from
    ``mask_fn`` (the "strong"/heavy view, e.g. mixed at rate 0.7) and v2
    from ``weak_mask_fn`` (a lighter mask, e.g. mixed at rate 0.1). The
    paired trainer then treats v2 as a teacher distribution for v1 (weak
    -> strong distillation), which avoids the failure mode where two
    equally heavy views collapse to a trivial agreeing prediction.

    Use with ``num_workers=0`` (see MaskAugDataset note).
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        mask_fn: Callable[[Tuple[int, ...], np.random.Generator], np.ndarray],
        concat_mask: bool = False,
        weak_mask_fn: Optional[
            Callable[[Tuple[int, ...], np.random.Generator], np.ndarray]
        ] = None,
    ):
        self.x = x
        self.y = y
        self.mask_fn = mask_fn
        self.weak_mask_fn = weak_mask_fn
        self.concat_mask = concat_mask
        self.sample_shape = x.shape[1:]
        self._rng = np.random.default_rng()

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int):
        m1 = self.mask_fn(self.sample_shape, self._rng)
        m2_fn = self.weak_mask_fn if self.weak_mask_fn is not None else self.mask_fn
        m2 = m2_fn(self.sample_shape, self._rng)
        x1 = (self.x[i] * m1).astype(np.float32)
        x2 = (self.x[i] * m2).astype(np.float32)
        if self.concat_mask:
            v1 = np.concatenate([x1, m1.astype(np.float32)], axis=0)
            v2 = np.concatenate([x2, m2.astype(np.float32)], axis=0)
        else:
            v1, v2 = x1, x2
        return torch.from_numpy(v1), torch.from_numpy(v2), int(self.y[i])


def make_mask_aware_input(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Build ``[X⊙M, M]`` along the channel dim for mask-aware models.

    x, mask: (N, C, T) float32 arrays. Returns (N, 2C, T) float32.
    """
    x_masked = (x * mask).astype(np.float32)
    return np.concatenate([x_masked, mask.astype(np.float32)], axis=1)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_classifier(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cuda",
    seed: int = 0,
    verbose: bool = True,
    train_loader: Optional[DataLoader] = None,
) -> Tuple[nn.Module, dict]:
    """Train a classifier with Adam + CE. Return (best_model_by_val_acc, history).

    If ``train_loader`` is provided it overrides the default loader built from
    ``x_train``/``y_train``. Useful for injecting MaskAugDataset-style
    training-time observation augmentation.
    """
    set_seed(seed)
    device = device if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    if train_loader is None:
        train_loader = _to_loader(x_train, y_train, batch_size, shuffle=True)
    val_loader = _to_loader(x_val, y_val, batch_size, shuffle=False)

    if verbose:
        n_train = len(train_loader.dataset)
        print(f"  training on {n_train} samples for {epochs} epochs on {device}...", flush=True)

    history = {"train_loss": [], "val_acc": []}
    best_state = None
    best_acc = -1.0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optim.step()
            epoch_loss += loss.item() * len(xb)
            n += len(xb)
        train_loss = epoch_loss / max(n, 1)
        val_acc = evaluate(model, x_val, y_val, batch_size=batch_size, device=device)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
        if verbose:
            print(f"  epoch {epoch+1:>2}/{epochs}  loss={train_loss:.4f}  val_acc={val_acc:.4f}",
                  flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_val_acc"] = best_acc
    return model, history


@torch.no_grad()
def evaluate(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int = 256,
    device: str = "cuda",
) -> float:
    """Return classification accuracy on (x, y)."""
    device = device if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    loader = _to_loader(x, y, batch_size, shuffle=False)
    correct = 0
    total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb).argmax(dim=-1)
        correct += (preds == yb).sum().item()
        total += len(yb)
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_with_tta(
    model: nn.Module,
    x_signal_masked: np.ndarray,
    mask: np.ndarray,
    y: np.ndarray,
    *,
    mask_aware: bool,
    tta_mask_fn: Optional[Callable[[Tuple[int, ...], np.random.Generator], np.ndarray]] = None,
    tta_k: int = 1,
    batch_size: int = 256,
    device: str = "cuda",
    seed: int = 0,
) -> float:
    """Evaluate with optional test-time multi-mask augmentation (TTA).

    Args:
        x_signal_masked: (N, C, T) degraded test signal X⊙M.
        mask: (N, C, T) test mask M.
        y: (N,) labels.
        mask_aware: True if model expects 2C-channel input [X⊙M, M] (OPACT mask-aware).
        tta_mask_fn(shape, rng) -> (C, T) mask: sampler for additional TTA masks.
            Use a mild distribution (e.g. mixed at low rate) so views remain useful.
        tta_k: number of TTA views (1 = no TTA, just one forward pass).
        seed: TTA RNG seed for reproducibility.

    Returns:
        Accuracy after averaging softmax probabilities over K TTA views.

    For tta_k=1 (default) the behaviour matches plain evaluate(): we just build
    the model input from (x_signal_masked, mask) according to ``mask_aware``
    and run one forward pass per batch.

    For tta_k>1, each test sample is forwarded K times. The k-th view applies an
    additional mask Mk on top of M, so the model sees inputs
    ``[X⊙M⊙Mk, M⊙Mk]`` (mask-aware) or ``X⊙M⊙Mk`` (plain). Softmax outputs are
    averaged across the K views before argmax. This converts OPACT-style train-
    time mask-view consistency into an inference-time ensemble.
    """
    device_ = device if torch.cuda.is_available() else "cpu"
    model = model.to(device_)
    model.eval()

    N = len(x_signal_masked)
    correct = 0
    total = 0

    # tta_k == 1 fast path: identical to evaluate(), no extra mask sampling.
    if tta_k <= 1 or tta_mask_fn is None:
        if mask_aware:
            inp = np.concatenate([x_signal_masked, mask], axis=1).astype(np.float32)
        else:
            inp = x_signal_masked.astype(np.float32)
        loader = _to_loader(inp, y, batch_size, shuffle=False)
        for xb, yb in loader:
            xb, yb = xb.to(device_), yb.to(device_)
            preds = model(xb).argmax(dim=-1)
            correct += (preds == yb).sum().item()
            total += len(yb)
        return correct / max(total, 1)

    # TTA path
    rng = np.random.default_rng(seed)
    sample_shape = x_signal_masked.shape[1:]
    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        xb_sig = x_signal_masked[batch_start:batch_end]
        xb_m = mask[batch_start:batch_end]
        yb = torch.from_numpy(y[batch_start:batch_end].astype(np.int64)).to(device_)

        accum: Optional[torch.Tensor] = None
        for _ in range(tta_k):
            tta_masks = np.empty_like(xb_sig)
            for i in range(len(xb_sig)):
                tta_masks[i] = tta_mask_fn(sample_shape, rng)
            aug_sig = (xb_sig * tta_masks).astype(np.float32)
            if mask_aware:
                aug_mask = (xb_m * tta_masks).astype(np.float32)
                aug_inp = np.concatenate([aug_sig, aug_mask], axis=1)
            else:
                aug_inp = aug_sig
            aug_t = torch.from_numpy(aug_inp).to(device_)
            logits = model(aug_t)
            probs = F.softmax(logits, dim=-1)
            accum = probs if accum is None else accum + probs

        avg_probs = accum / tta_k
        preds = avg_probs.argmax(dim=-1)
        correct += (preds == yb).sum().item()
        total += len(yb)

    return correct / max(total, 1)


def apply_mask_per_sample(
    x: np.ndarray,
    mask_fn: Callable[[Tuple[int, int], Optional[np.random.Generator]], np.ndarray],
    rng: np.random.Generator,
    return_masks: bool = False,
):
    """Apply an independently sampled mask to each sample in x.

    x: (N, C, T) float32
    mask_fn(shape, rng) -> (C, T) {0,1} float mask
    Returns the masked copy (does not mutate x).
    If return_masks=True, returns (x_masked, masks) where masks has shape (N, C, T).
    """
    out = np.empty_like(x)
    masks = np.empty_like(x) if return_masks else None
    sample_shape = x.shape[1:]
    for i in range(x.shape[0]):
        m = mask_fn(sample_shape, rng)
        out[i] = x[i] * m
        if masks is not None:
            masks[i] = m
    if return_masks:
        return out, masks
    return out


def train_classifier_two_view(
    model: nn.Module,
    train_loader: DataLoader,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    consistency_weight: float = 1.0,
    consistency_mode: str = "symmetric",
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cuda",
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[nn.Module, dict]:
    """Two-view consistency training.

    ``train_loader`` yields ``(v1, v2, y)`` triples (built externally from a
    MaskAugTwoViewDataset).

    consistency_mode="symmetric" (R-Drop-style; matches the original OPACT
    description). Both views are expected to come from the same mask
    distribution. Loss:

        L = CE(p1, y) + CE(p2, y) + λ * (KL(p1||p2) + KL(p2||p1))

    consistency_mode="asymmetric" (weak -> strong distillation). The dataset
    is expected to provide v1 from a heavy (strong) mask and v2 from a
    lighter (weak) mask. v2 acts as a soft teacher with stop-gradient; v1
    is the student that learns to match the teacher's prediction. This
    breaks the symmetric-failure mode where two equally heavy views collapse
    to a degenerate agreeing prediction:

        L = CE(p1, y) + CE(p2, y) + λ * KL( stop_grad(p2) || p1 )

    ``x_val`` must already be in the format the model expects.
    """
    if consistency_mode not in ("symmetric", "asymmetric"):
        raise ValueError(f"consistency_mode must be 'symmetric' or 'asymmetric', "
                         f"got {consistency_mode!r}")

    set_seed(seed)
    device = device if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    ce_loss = nn.CrossEntropyLoss()

    val_loader = _to_loader(x_val, y_val, batch_size, shuffle=False)

    if verbose:
        n_train = len(train_loader.dataset)
        print(
            f"  two-view training on {n_train} samples for {epochs} epochs on "
            f"{device} (λ={consistency_weight}, mode={consistency_mode})...",
            flush=True,
        )

    history = {
        "train_loss": [], "train_ce": [], "train_con": [], "val_acc": [],
    }
    best_state = None
    best_acc = -1.0

    for epoch in range(epochs):
        model.train()
        epoch_total = epoch_ce = epoch_con = 0.0
        n = 0
        for v1, v2, y in train_loader:
            v1, v2, y = v1.to(device), v2.to(device), y.to(device)
            logits1 = model(v1)
            logits2 = model(v2)
            loss_ce = ce_loss(logits1, y) + ce_loss(logits2, y)
            log_p1 = F.log_softmax(logits1, dim=-1)
            log_p2 = F.log_softmax(logits2, dim=-1)

            # F.kl_div(input=log_q, target=p) computes sum p * (log p - log_q) = KL(p || q)
            if consistency_mode == "symmetric":
                p1 = log_p1.exp()
                p2 = log_p2.exp()
                kl_12 = F.kl_div(log_p2, p1, reduction="batchmean")  # KL(p1||p2)
                kl_21 = F.kl_div(log_p1, p2, reduction="batchmean")  # KL(p2||p1)
                loss_con = kl_12 + kl_21
            else:  # asymmetric: v1 student, v2 weak teacher with stop-grad
                p2_detached = log_p2.detach().exp()
                # KL(teacher.detach() || student) — forward KL distillation
                loss_con = F.kl_div(log_p1, p2_detached, reduction="batchmean")

            loss = loss_ce + consistency_weight * loss_con

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optim.step()

            bsz = y.size(0)
            epoch_total += loss.item() * bsz
            epoch_ce += loss_ce.item() * bsz
            epoch_con += loss_con.item() * bsz
            n += bsz

        train_loss = epoch_total / max(n, 1)
        train_ce = epoch_ce / max(n, 1)
        train_con = epoch_con / max(n, 1)
        val_acc = evaluate(model, x_val, y_val, batch_size=batch_size, device=device)

        history["train_loss"].append(train_loss)
        history["train_ce"].append(train_ce)
        history["train_con"].append(train_con)
        history["val_acc"].append(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
        if verbose:
            print(
                f"  epoch {epoch+1:>2}/{epochs}  loss={train_loss:.4f} "
                f"(ce={train_ce:.4f} con={train_con:.4f})  val_acc={val_acc:.4f}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_val_acc"] = best_acc
    return model, history
