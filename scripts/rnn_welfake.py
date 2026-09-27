"""
RECURRENT BASELINE — attention-pooled BiLSTM on WELFake.

Published WELFake leaderboards are dominated by recurrent models (attention BiLSTM
97.66%, CNN-BiLSTM 97.74%, BERT+BiLSTM 98.10%, hybrid CNN+dual-BiLSTM 98.85%),
while our linear TF-IDF system reaches 97.24%. This script asks two questions:

  Q1  Does a recurrent model beat the linear system on OUR protocol?
  Q2  How much of the gap to published numbers is simply DUPLICATE LEAKAGE?
      We de-duplicate before splitting (72,134 -> 63,676, 11.7% removed); published
      work generally uses the raw file, where the same article can sit in both train
      and test. So each model is run twice: once de-duplicated, once raw.

Everything else is held fixed against the linear experiments: same 70/15/15
stratified split, same seed, same threshold tuning on validation, same metric suite.

    python scripts/rnn_welfake.py
"""
from __future__ import annotations

import re, sys, time, warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

SEED = 42
FAKE, REAL = 0, 1
WELFAKE_CSV = Path("/mnt/user-data/uploads/Downloads/WELFake_Dataset.csv")
OUT = REPO / "results"; OUT.mkdir(exist_ok=True)

MAX_VOCAB, MAX_LEN = 40_000, 250
EMB, HID, BATCH, EPOCHS, PATIENCE = 128, 96, 128, 6, 2

torch.manual_seed(SEED); np.random.seed(SEED)
torch.set_num_threads(2)
_TOK = re.compile(r"[a-z0-9']+")


# ------------------------------------------------------------------ metrics
def tune_threshold(y, p):
    best_t, best_f = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        f = f1_score(y, np.where(p >= t, FAKE, REAL), average="macro")
        if f > best_f:
            best_f, best_t = f, t
    return best_t, best_f


def full_metrics(y, p, t):
    pred = np.where(p >= t, FAKE, REAL)
    cm = confusion_matrix(y, pred, labels=[FAKE, REAL])
    tp, fn, fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    return {"macro_f1": f1_score(y, pred, average="macro"),
            "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, pos_label=FAKE, zero_division=0),
            "recall": recall_score(y, pred, pos_label=FAKE, zero_division=0),
            "roc_auc": roc_auc_score((y == FAKE).astype(int), p),
            "mcc": matthews_corrcoef(y, pred),
            "fake_FNR": fn / max(fn + tp, 1), "fake_FPR": fp / max(fp + tn, 1)}


# ------------------------------------------------------------------ data
def load_welfake(deduplicate: bool) -> pd.DataFrame:
    df = pd.read_csv(WELFAKE_CSV)
    df["title"] = df["title"].fillna(""); df["text"] = df["text"].fillna("")
    df["doc"] = (df["title"] + ". " + df["text"]).str.strip()
    df = df[df["doc"].str.len() > 0].dropna(subset=["label"])
    if deduplicate:
        df = df.drop_duplicates(subset=["doc"])
    # WELFake ships 0=real, 1=fake -> flip onto repo convention fake=0, real=1
    df["label"] = 1 - df["label"].astype(int)
    return df.reset_index(drop=True)[["doc", "label"]]


def split(df):
    tr, tmp = train_test_split(df, test_size=.30, stratify=df["label"], random_state=SEED)
    va, te = train_test_split(tmp, test_size=.50, stratify=tmp["label"], random_state=SEED)
    return (x.reset_index(drop=True) for x in (tr, va, te))


def build_vocab(texts):
    c = Counter()
    for t in texts:
        c.update(_TOK.findall(t.lower())[:MAX_LEN])
    itos = ["<pad>", "<unk>"] + [w for w, _ in c.most_common(MAX_VOCAB - 2)]
    return {w: i for i, w in enumerate(itos)}


def encode(texts, stoi):
    X = np.zeros((len(texts), MAX_LEN), dtype=np.int64)
    for i, t in enumerate(texts):
        ids = [stoi.get(w, 1) for w in _TOK.findall(t.lower())[:MAX_LEN]]
        X[i, :len(ids)] = ids
    return torch.from_numpy(X)


# ------------------------------------------------------------------ model
class AttnBiLSTM(nn.Module):
    """Embedding -> BiLSTM -> additive attention pooling -> classifier."""

    def __init__(self, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab, EMB, padding_idx=0)
        self.lstm = nn.LSTM(EMB, HID, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(HID * 2, 1)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(HID * 2, 1)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1)                 # (B, L, 1)
        h, _ = self.lstm(self.emb(x))                 # (B, L, 2H)
        a = self.attn(h).masked_fill(~mask, -1e9)     # ignore padding
        w = torch.softmax(a, dim=1)
        pooled = (w * h).sum(1)                       # (B, 2H)
        return self.fc(self.drop(pooled)).squeeze(-1)


def predict_proba(model, X, bs=256):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(torch.sigmoid(model(X[i:i + bs])).numpy())
    return np.concatenate(out)


def train_rnn(Xtr, ytr, Xva, yva, vocab):
    """Train to best validation macro-F1, with early stopping."""
    model = AttnBiLSTM(vocab)
    # target = P(FAKE); pos_weight balances the two classes
    t_tr = torch.from_numpy((ytr == FAKE).astype(np.float32))
    pos_w = torch.tensor([(t_tr == 0).sum() / max((t_tr == 1).sum(), 1)])
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_f, best_state, bad = -1.0, None, 0
    n = len(Xtr)
    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n)
        t0 = time.time()
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = lossf(model(Xtr[idx]), t_tr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        pv = predict_proba(model, Xva)
        _, f = tune_threshold(yva, pv)
        print(f"      epoch {ep}  val macro-F1 {f:.4f}   ({time.time()-t0:.0f}s)", flush=True)
        if f > best_f:
            best_f, bad = f, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                print("      early stop", flush=True); break
    model.load_state_dict(best_state)
    return model


# ------------------------------------------------------------------ main
def main():
    rows = []
    for dedup in (True, False):
        tag = "de-duplicated (our protocol)" if dedup else "RAW, duplicates kept (published protocol)"
        df = load_welfake(dedup)
        tr, va, te = split(df)
        y_tr, y_va, y_te = (x["label"].to_numpy() for x in (tr, va, te))
        print(f"\n===== WELFake — {tag} =====")
        print(f"  rows {len(df):,}   train {len(tr):,}  val {len(va):,}  test {len(te):,}", flush=True)

        # ---- linear reference on the SAME data variant ----
        vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3,
                              max_features=100_000, sublinear_tf=True,
                              strip_accents="unicode")
        Xtr_s = vec.fit_transform(tr["doc"]); Xva_s = vec.transform(va["doc"])
        Xte_s = vec.transform(te["doc"])
        lin = CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=SEED),
                                     cv=3, method="sigmoid").fit(Xtr_s, y_tr)
        col = list(lin.classes_).index(FAKE)
        pv = lin.predict_proba(Xva_s)[:, col]; pt = lin.predict_proba(Xte_s)[:, col]
        t, _ = tune_threshold(y_va, pv)
        m = full_metrics(y_te, pt, t)
        rows.append({"variant": tag, "model": "Linear TF-IDF + calibrated LinearSVC", **m})
        print(f"  linear  acc {m['accuracy']:.4f}  macro-F1 {m['macro_f1']:.4f}", flush=True)

        # ---- BiLSTM ----
        stoi = build_vocab(tr["doc"])
        Xtr = encode(tr["doc"], stoi); Xva = encode(va["doc"], stoi); Xte = encode(te["doc"], stoi)
        print(f"  vocab {len(stoi):,}  seq {MAX_LEN}  training BiLSTM ...", flush=True)
        model = train_rnn(Xtr, y_tr, Xva, y_va, len(stoi))
        pv = predict_proba(model, Xva); pt = predict_proba(model, Xte)
        t, _ = tune_threshold(y_va, pv)
        m = full_metrics(y_te, pt, t)
        rows.append({"variant": tag, "model": "Attention BiLSTM (this work)", **m})
        print(f"  BiLSTM  acc {m['accuracy']:.4f}  macro-F1 {m['macro_f1']:.4f}", flush=True)

    res = pd.DataFrame(rows).round(4)
    print("\n================ RESULTS ================")
    print(res.to_string(index=False))
    res.to_csv(OUT / "rnn_welfake.csv", index=False)
    print(f"\nwrote {OUT/'rnn_welfake.csv'}")


if __name__ == "__main__":
    main()
