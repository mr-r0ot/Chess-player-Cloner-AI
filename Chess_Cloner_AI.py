#!/usr/bin/env python3
# Chess_Cloner_AI.py
# CPU-first, robust, practical "clone" trainer for chess.com user(s).
# Usage:
# python Chess_Cloner_AI.py --dataset path/to/csv_or_dir_or_prefix --outmodel mymodel.pt --max-rows 100000

from __future__ import annotations
import argparse, os, sys, glob, math, random, json, re, time
from collections import Counter
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from tqdm import tqdm

# ---------------- CONFIG ----------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cpu")  # CPU only by design
NUM_PLANES = 12 + 1 + 4 + 1  # piece planes + stm + castling KQkq + halfmove normalized
PROMO_PIECES = ['q','r','b','n']
# ----------------------------------------

# --------- Utilities: expand dataset path (files/dir/prefix) ----------
def expand_dataset_paths(input_paths: List[str]) -> List[str]:
    out = []
    for p in input_paths:
        p = os.path.expanduser(p)
        p = os.path.abspath(p)
        if os.path.isfile(p):
            out.append(p); continue
        if os.path.isdir(p):
            csvs = sorted(glob.glob(os.path.join(p, "*.csv")))
            out.extend(csvs); continue
        # try p.csv
        p_csv = p + ".csv"
        if os.path.isfile(p_csv):
            out.append(p_csv); continue
        # try prefix glob
        found = sorted(glob.glob(p + "*.csv"))
        if found:
            out.extend(found); continue
        parent = os.path.dirname(p)
        if parent and os.path.isdir(parent):
            cand = os.path.join(parent, os.path.basename(p) + "*.csv")
            found = sorted(glob.glob(cand))
            if found:
                out.extend(found); continue
    # unique preserve order
    seen = set(); res=[]
    for f in out:
        if f not in seen:
            seen.add(f); res.append(f)
    return res

# ---------- Robust FEN sanitizer ----------
def sanitize_fen(fen: str) -> str:
    """
    Clean malformed FEN fragments. Keep first 4 tokens (board, stm, castle, ep),
    then try to extract integer tokens for halfmove/fullmove from remaining tokens.
    Fallback to startpos if totally broken.
    """
    try:
        parts = fen.strip().split()
        if len(parts) >= 6:
            # typical case includes extra garbage like '1+2 0 32' -> we'll extract integer substrings
            placement, stm, castling, ep = parts[0], parts[1], parts[2], parts[3]
            # scan parts[4:] for integers
            half = None; full = None
            for tok in parts[4:]:
                m = re.search(r'\d+', tok)
                if m:
                    v = int(m.group(0))
                    if half is None: half = v
                    full = v
            if half is None: half = 0
            if full is None: full = 1
            return f"{placement} {stm} {castling} {ep} {half} {full}"
        elif len(parts) == 5:
            # maybe halfmove is '1+2' or so -> try to extract digits
            placement, stm, castling, ep = parts[0], parts[1], parts[2], parts[3]
            m = re.search(r'\d+', parts[4])
            half = int(m.group(0)) if m else 0
            full = 1
            return f"{placement} {stm} {castling} {ep} {half} {full}"
        elif len(parts) == 4:
            placement, stm, castling, ep = parts
            return f"{placement} {stm} {castling} {ep} 0 1"
        else:
            return chess.Board().fen()
    except Exception:
        return chess.Board().fen()

# --------- Encoder: FEN -> planes tensor (NUM_PLANES,8,8) ----------
def encode_fen_to_numpy(fen: str) -> np.ndarray:
    try:
        board = chess.Board(fen)
    except Exception:
        # try sanitize
        try:
            board = chess.Board(sanitize_fen(fen))
        except Exception:
            board = chess.Board()
    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)
    piece_plane = {chess.PAWN:0, chess.KNIGHT:1, chess.BISHOP:2, chess.ROOK:3, chess.QUEEN:4, chess.KING:5}
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            plane = piece_plane[piece.piece_type]
            r = 7 - chess.square_rank(sq)
            c = chess.square_file(sq)
            if piece.color == chess.WHITE:
                planes[plane, r, c] = 1.0
            else:
                planes[plane+6, r, c] = 1.0
    planes[12,:,:] = 1.0 if board.turn == chess.WHITE else 0.0
    planes[13,:,:] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    planes[14,:,:] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    planes[15,:,:] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    planes[16,:,:] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    try:
        halfmove = int(board.fen().split()[4])
    except Exception:
        halfmove = 0
    planes[17,:,:] = float(halfmove)/100.0
    return planes

# ---------- Augmentation: flip colors & mirror move ----------
def mirror_uci_move(m: str) -> str:
    if len(m) < 4:
        return m
    frm = chess.parse_square(m[0:2]); to = chess.parse_square(m[2:4])
    frm2 = 63 - frm; to2 = 63 - to
    base = chess.square_name(frm2) + chess.square_name(to2)
    if len(m) == 5:
        return base + m[4]
    return base

def augment_flip(planes: np.ndarray, move_uci: str) -> Tuple[np.ndarray,str]:
    p = np.flip(np.flip(planes, axis=1), axis=2)
    p_swapped = np.zeros_like(p)
    p_swapped[0:6] = p[6:12]; p_swapped[6:12] = p[0:6]
    p_swapped[12] = 1.0 - p_swapped[12]
    p_swapped[13] = p[15]; p_swapped[14] = p[16]; p_swapped[15] = p[13]; p_swapped[16] = p[14]
    p_swapped[17] = p[17]
    return p_swapped, mirror_uci_move(move_uci)

# ---------- Read & sample CSVs robustly (chunking) ----------
def sample_rows_from_csvs(csv_paths: List[str], target_rows: int) -> pd.DataFrame:
    """
    Build a dataframe sampled approximately uniformly from CSV files.
    We'll read each file in chunks and accumulate until target reached.
    """
    if target_rows is None:
        # read everything (caution - may be huge)
        dfs = [pd.read_csv(p, usecols=['fen_before','move_uci','side_to_move','white','black']) for p in csv_paths]
        df = pd.concat(dfs, ignore_index=True)
        return df
    collected = []
    remaining = target_rows
    for p in csv_paths:
        if remaining <= 0:
            break
        try:
            for chunk in pd.read_csv(p, usecols=['fen_before','move_uci','side_to_move','white','black'], chunksize=100000):
                if remaining <= 0:
                    break
                take = min(len(chunk), remaining)
                if take < len(chunk):
                    chunk = chunk.sample(n=take, random_state=SEED)
                collected.append(chunk)
                remaining -= take
                if remaining <= 0:
                    break
        except Exception as e:
            print(f"[WARN] reading {p} failed: {e}")
            continue
    if not collected:
        raise RuntimeError("No data collected from CSVs")
    df = pd.concat(collected, ignore_index=True)
    if len(df) > target_rows:
        df = df.sample(n=target_rows, random_state=SEED).reset_index(drop=True)
    return df.reset_index(drop=True)

# ---------- Build move vocab from sampled dataframe ----------
def build_move_vocab_from_df(df: pd.DataFrame, min_count: int = 1) -> Dict[str,int]:
    c = Counter(df['move_uci'].astype(str).values)
    moves = [m for m,ct in c.items() if ct >= min_count]
    moves = sorted(moves)
    vocab = {m:i for i,m in enumerate(moves)}
    return vocab

# ---------- Dataset object that returns tensors ----------
class MemDataset(Dataset):
    def __init__(self, X: np.ndarray, move_idx: List[int], from_idx: List[int], to_idx: List[int], promo_idx: List[int], move_vocab: Dict[str,int], augment_flip_prob: float = 0.0):
        """
        X: numpy array (N, NUM_PLANES, 8, 8)
        move_idx: list of vocab indices (or -1)
        from_idx, to_idx: 0..63 ints
        promo_idx: -1 or 0..3 (index into PROMO_PIECES)
        """
        self.X = torch.from_numpy(X).float()
        self.move_idx = torch.tensor(move_idx, dtype=torch.long)
        self.from_idx = torch.tensor(from_idx, dtype=torch.long)
        self.to_idx = torch.tensor(to_idx, dtype=torch.long)
        self.promo_idx = torch.tensor(promo_idx, dtype=torch.long)
        self.augment_flip_prob = augment_flip_prob
        self.move_vocab = move_vocab

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        mv = int(self.move_idx[idx].item())
        frm = int(self.from_idx[idx].item())
        to = int(self.to_idx[idx].item())
        promo = int(self.promo_idx[idx].item())
        if self.augment_flip_prob > 0 and random.random() < self.augment_flip_prob:
            # flip data on-the-fly
            p = x.numpy()
            p2, _ = augment_flip(p, "a2a3")  # we only need planes; move will be rewritten below
            x = torch.from_numpy(p2).float()
            # remap from/to by mirroring indices:
            frm = 63 - frm; to = 63 - to
            # promo: remains same mapping (q/r/b/n)
        return x, mv, frm, to, promo

# ---------- Model: conv trunk + mixed heads (policy_vocab + from/to/promo + value) ----------
class CloneNet(nn.Module):
    def __init__(self, in_planes=NUM_PLANES, n_filters=80, n_blocks=3, vocab_size=1000):
        super().__init__()
        self.conv_in = nn.Conv2d(in_planes, n_filters, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(n_filters, n_filters, 3, padding=1),
                nn.BatchNorm2d(n_filters),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_filters, n_filters, 3, padding=1),
                nn.BatchNorm2d(n_filters),
                nn.ReLU(inplace=True),
            ))
        # vocabulary policy head
        self.policy_vocab = nn.Sequential(
            nn.Conv2d(n_filters, 32, kernel_size=1), nn.ReLU(inplace=True),
            nn.Flatten(), nn.Linear(32*8*8, 512), nn.ReLU(inplace=True),
            nn.Linear(512, vocab_size)
        )
        # factorized heads
        self.from_head = nn.Sequential(nn.Conv2d(n_filters, 16, 1), nn.ReLU(inplace=True), nn.Flatten(), nn.Linear(16*8*8, 256), nn.ReLU(inplace=True), nn.Linear(256, 64))
        self.to_head   = nn.Sequential(nn.Conv2d(n_filters, 16, 1), nn.ReLU(inplace=True), nn.Flatten(), nn.Linear(16*8*8, 256), nn.ReLU(inplace=True), nn.Linear(256, 64))
        self.promo_head = nn.Sequential(nn.Conv2d(n_filters, 8, 1), nn.ReLU(inplace=True), nn.Flatten(), nn.Linear(8*8*8, 64), nn.ReLU(inplace=True), nn.Linear(64, len(PROMO_PIECES)+1))
        self.value_head = nn.Sequential(nn.Conv2d(n_filters, 8, 1), nn.ReLU(inplace=True), nn.Flatten(), nn.Linear(8*8*8, 64), nn.ReLU(inplace=True), nn.Linear(64,1), nn.Tanh())

    def forward(self, x):
        z = self.conv_in(x)
        for b in self.blocks:
            z = z + b(z)
        pv = self.policy_vocab(z)
        f = self.from_head(z)
        t = self.to_head(z)
        pr = self.promo_head(z)
        val = self.value_head(z).squeeze(-1)
        return pv, f, t, pr, val

# ---------- Helper: convert move uci -> indices ----------
def move_to_from_to_promo(move_uci: str) -> Tuple[int,int,int]:
    """
    return from_idx (0..63), to_idx (0..63), promo_idx (-1 for none else 0..3)
    """
    try:
        frm = chess.parse_square(move_uci[0:2]); to = chess.parse_square(move_uci[2:4])
    except Exception:
        # fallback to 0,0
        return 0,0,-1
    promo = -1
    if len(move_uci) == 5:
        p = move_uci[4].lower()
        if p in PROMO_PIECES:
            promo = PROMO_PIECES.index(p)
    return frm, to, promo

# ---------- Training loop ----------
def train(model: nn.Module, dataset: MemDataset, outpath: str, epochs: int=6, batch_size: int=128, lr: float=1e-3, alpha=1.0, beta=0.6, gamma=0.2):
    model.to(DEVICE)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    model.train()
    for ep in range(epochs):
        total_loss=0.0; n=0
        pbar = tqdm(loader, desc=f"Epoch {ep+1}/{epochs}", leave=False)
        for batch in pbar:
            xb, y_vocab, y_from, y_to, y_promo = batch
            xb = xb.to(DEVICE); y_vocab = y_vocab.to(DEVICE); y_from=y_from.to(DEVICE); y_to=y_to.to(DEVICE); y_promo=y_promo.to(DEVICE)
            pv_logits, f_logits, t_logits, pr_logits, val = model(xb)
            loss_policy = ce(pv_logits, y_vocab)
            loss_from = ce(f_logits, y_from)
            loss_to = ce(t_logits, y_to)
            loss_promo = ce(pr_logits, y_promo)
            # optional tiny value loss to regularize (target unknown -> skip)
            loss = alpha * loss_policy + beta * (loss_from + loss_to) + gamma * loss_promo
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * xb.size(0); n += xb.size(0)
            pbar.set_postfix(loss=total_loss/n)
        print(f"Epoch {ep+1} avg loss: {total_loss/(n or 1):.4f}")
    # save
    torch.save({'state': model.state_dict(), 'in_planes': NUM_PLANES, 'vocab_size': dataset.move_vocab_len if hasattr(dataset,'move_vocab_len') else None}, outpath)

# ---------- Inference selection (prefers vocab policy; fallback to factorized) ----------
def select_move(fen: str, model: nn.Module, move_vocab: Dict[str,int], inv_vocab: Dict[int,str], temperature: float = 1.0):
    model.to(DEVICE); model.eval()
    x = torch.from_numpy(encode_fen_to_numpy(fen)).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        pv_logits, f_logits, t_logits, pr_logits, val = model(x)
        pv_logits = pv_logits.squeeze(0).cpu().numpy()
        f_logits = f_logits.squeeze(0).cpu().numpy()
        t_logits = t_logits.squeeze(0).cpu().numpy()
    board = chess.Board(fen)
    legal = list(board.legal_moves)
    # try vocab-first
    cand = []
    for m in legal:
        u = m.uci()
        if u in move_vocab:
            cand.append((move_vocab[u], u))
    if cand:
        # pick argmax of pv_logits among known
        scores = [(pv_logits[idx], u) for idx,u in cand]
        scores.sort(reverse=True, key=lambda x: x[0])
        return scores[0][1]
    # fallback: compose from+to logits and mask legal moves
    scores = []
    for m in legal:
        u = m.uci()
        frm = chess.parse_square(u[0:2]); to = chess.parse_square(u[2:4])
        score = f_logits[frm] + t_logits[to]
        scores.append((score, u))
    if not scores:
        return None
    scores.sort(reverse=True, key=lambda x: x[0])
    return scores[0][1]

# ---------- Main CLI ----------
def main():
    parser = argparse.ArgumentParser(prog='Chess_Cloner_AI.py')
    parser.add_argument('--dataset', nargs='+', required=True, help='CSV file(s) or directory/prefix containing CSV(s)')
    parser.add_argument('--outmodel', default='chess_clone.pt')
    parser.add_argument('--vocab', default='move_vocab.json')
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--augment-flip-prob', type=float, default=0.25)
    parser.add_argument('--max-rows', type=int, default=100000, help='Sample size to load for training (use smaller for speed).')
    parser.add_argument('--nfilters', type=int, default=80)
    parser.add_argument('--nblocks', type=int, default=3)
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    print("Expanding dataset paths...")
    csvs = expand_dataset_paths(args.dataset)
    if not csvs:
        print("No CSV files found. Provide a CSV file or a directory containing CSVs.")
        sys.exit(1)
    print(f"Found {len(csvs)} CSV(s):")
    for c in csvs: print(" ", c)

    print(f"Sampling up to {args.max_rows} rows from CSVs...")
    df = sample_rows_from_csvs(csvs, args.max_rows)
    print("Loaded rows:", len(df))

    if 'fen_before' not in df.columns or 'move_uci' not in df.columns:
        print("CSV missing required columns 'fen_before' and 'move_uci'. Aborting.")
        sys.exit(1)

    print("Building move vocab...")
    mvocab = build_move_vocab_from_df(df)
    inv_vocab = {v:k for k,v in mvocab.items()}
    print("vocab size:", len(mvocab))
    # save vocab
    with open(args.vocab, 'w', encoding='utf8') as f:
        json.dump(mvocab, f, ensure_ascii=False, indent=2)

    # Precompute tensors (will fit in memory if max_rows reasonable)
    N = len(df)
    print("Encoding positions to tensors (this may take a few moments)...")
    X = np.zeros((N, NUM_PLANES, 8, 8), dtype=np.float32)
    move_idx = [-1]*N; from_idx=[0]*N; to_idx=[0]*N; promo_idx=[len(PROMO_PIECES)]*N
    for i, row in tqdm(df.iterrows(), total=N):
        fen = row['fen_before']
        mv = str(row['move_uci'])
        try:
            X[i] = encode_fen_to_numpy(fen)
        except Exception:
            X[i] = encode_fen_to_numpy(chess.Board().fen())
        # indices
        move_idx[i] = mvocab.get(mv, 0)  # if not in vocab map to 0 (common fallback) - but vocab built from df so should exist
        frm, to, p = move_to_from_to_promo(mv)
        from_idx[i] = frm; to_idx[i] = to; promo_idx[i] = (p if p is not None and p>=0 else len(PROMO_PIECES))

    print("Dataset prepared: samples:", N, "mem approx:", (X.nbytes/1024**2).__round__(2), "MB")
    ds = MemDataset(X, move_idx, from_idx, to_idx, promo_idx, mvocab, augment_flip_prob=args.augment_flip_prob)
    # attach length info for save maybe
    ds.move_vocab_len = len(mvocab)

    print("Constructing model...")
    model = CloneNet(in_planes=NUM_PLANES, n_filters=args.nfilters, n_blocks=args.nblocks, vocab_size=len(mvocab))
    print("Training (CPU-only). epochs:", args.epochs, "batch:", args.batch)
    train(model, ds, args.outmodel, epochs=args.epochs, batch_size=args.batch, lr=args.lr)
    print("Saved model:", args.outmodel, "vocab:", args.vocab)
    print("Done.")

if __name__ == "__main__":
    main()
