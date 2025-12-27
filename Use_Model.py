#!/usr/bin/env python3
"""
Dependencies: python-chess, torch, pandas, numpy
pip install python-chess torch pandas numpy

Coded by: Mohammad Taha Gorji
GitHub: https://github.com/mr-r0ot/Chess-player-Cloner-AI
"""

import os, sys, json, math, random, time, glob, threading
from collections import Counter
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import chess
import chess.pgn
import chess.engine

import numpy as np
import torch
import torch.nn as nn

# ---------------- Configuration & defaults ----------------
NUM_PLANES = 18
BOOK_DEFAULT_DEPTH_FULLMOVES = 10
STOCKFISH_REL_PATH = os.path.join(os.path.dirname(__file__), 'stockfish', 'stockfish-windows-x86-64-avx2.exe')
DEFAULT_SF_DEPTH = 8
DEFAULT_BLUNDER_CP = 200
DEFAULT_PRESERVE_PROB = 0.75
DEFAULT_MODEL_TOPK = 8
DEFAULT_TEMP = 0.9

# Weight strategy presets (user-facing names -> parameters)
WEIGHT_STRATEGIES = [
    ("No change",            {'preserve_prob':0.75,'blunder_cp':200,'sf_depth':6,'force_sf':False}),
    ("Small changes",        {'preserve_prob':0.7, 'blunder_cp':180,'sf_depth':8,'force_sf':False}),
    ("Preserve style (max)", {'preserve_prob':0.9, 'blunder_cp':300,'sf_depth':6,'force_sf':False}),
    ("Reduce blunders",      {'preserve_prob':0.6, 'blunder_cp':120,'sf_depth':8,'force_sf':False}),
    ("Ideal (no blunders)", {'preserve_prob':0.2, 'blunder_cp':50, 'sf_depth':12,'force_sf':True}),
    ("Ultra optimal",        {'preserve_prob':0.35,'blunder_cp':20, 'sf_depth':14,'force_sf':True}),
]

# ---------------- Utilities ----------------
import re

def sanitize_fen(fen: str) -> str:
    """Sanitize possibly malformed FEN and return a safe FEN string."""
    try:
        parts = str(fen).strip().split()
        if len(parts) >= 6:
            placement, stm, castling, ep = parts[0], parts[1], parts[2], parts[3]
            half=None; full=None
            for tok in parts[4:]:
                m = re.search(r"\\d+", str(tok))
                if m:
                    v = int(m.group(0))
                    if half is None: half=v
                    full=v
            if half is None: half=0
            if full is None: full=1
            return f"{placement} {stm} {castling} {ep} {half} {full}"
        else:
            b = chess.Board(fen)
            return b.fen()
    except Exception:
        return chess.Board().fen()


def fen_key_for_book(fen: str) -> str:
    try:
        b = chess.Board(fen)
        parts = b.fen().split()
        return ' '.join(parts[:4])
    except Exception:
        return ' '.join(sanitize_fen(fen).split()[:4])


def ply_from_start(fen: str) -> int:
    b = chess.Board(fen)
    return (b.fullmove_number - 1) * 2 + (0 if b.turn==chess.WHITE else 1)

# ---------------- Opening book ----------------
import pandas as pd

def build_opening_book_from_folder(folder: str, book_moves: int = BOOK_DEFAULT_DEPTH_FULLMOVES, max_rows: int = None, progress_callback=None):
    """Scan CSV/PGN/JSON files in folder to build an opening book dict (fen_key -> Counter(move->count)).
       progress_callback(optional): function(progress_message) called during long ops.
    """
    folder = os.path.abspath(folder)
    files = []
    for ext in ('*.csv','*.pgn','*.json','*.ndjson'):
        files += sorted(glob.glob(os.path.join(folder, ext)))
    book = {}
    total = 0
    for f in files:
        if max_rows is not None and total >= max_rows:
            break
        try:
            if progress_callback:
                progress_callback(f'Reading {os.path.basename(f)}...')
            if f.lower().endswith('.pgn'):
                with open(f, encoding='utf8', errors='ignore') as fh:
                    while True:
                        game = chess.pgn.read_game(fh)
                        if game is None: break
                        board = game.board()
                        for mv in game.mainline_moves():
                            fen = board.fen()
                            mvuci = mv.uci()
                            ply = ply_from_start(fen)
                            if ply <= book_moves*2:
                                key = fen_key_for_book(fen)
                                book.setdefault(key, Counter())[mvuci] += 1
                                total += 1
                            board.push(mv)
                            if max_rows is not None and total>=max_rows: break
            elif f.lower().endswith('.csv'):
                for chunk in pd.read_csv(f, usecols=['fen_before','move_uci'], chunksize=200000):
                    for _, row in chunk.iterrows():
                        fen = str(row['fen_before']); mvuci = str(row['move_uci'])
                        ply = ply_from_start(fen)
                        if ply <= book_moves*2:
                            key = fen_key_for_book(fen)
                            book.setdefault(key, Counter())[mvuci] += 1
                            total += 1
                        if max_rows is not None and total>=max_rows: break
                    if max_rows is not None and total>=max_rows: break
            elif f.lower().endswith('.json') or f.lower().endswith('.ndjson'):
                with open(f, encoding='utf8', errors='ignore') as fh:
                    for line in fh:
                        if not line.strip(): continue
                        j = json.loads(line)
                        fen = j.get('fen_before') or j.get('fen') or j.get('position')
                        mvuci = j.get('move_uci') or j.get('move')
                        if fen and mvuci:
                            ply = ply_from_start(fen)
                            if ply <= book_moves*2:
                                key = fen_key_for_book(fen)
                                book.setdefault(key, Counter())[mvuci] += 1
                                total += 1
                        if max_rows is not None and total>=max_rows: break
        except Exception as e:
            # non-fatal, skip file
            print('Warning: failed to read', f, e)
    return {k: dict(v) for k,v in book.items()}


def save_opening_book(book: dict, path: str):
    with open(path, 'w', encoding='utf8') as f:
        json.dump(book, f, ensure_ascii=False, indent=2)


def load_opening_book(path: str) -> dict:
    with open(path, 'r', encoding='utf8') as f:
        book = json.load(f)
    for k in list(book.keys()):
        book[k] = {mv:int(c) for mv,c in book[k].items()}
    return book

# ---------------- Model (must match training) ----------------
class CloneNet(nn.Module):
    def __init__(self, in_planes=NUM_PLANES, n_filters=80, n_blocks=3, vocab_size=1000):
        super().__init__()
        self.conv_in = nn.Conv2d(in_planes, n_filters, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(n_filters, n_filters, 3, padding=1),
                nn.BatchNorm2d(n_filters), nn.ReLU(inplace=True),
                nn.Conv2d(n_filters, n_filters, 3, padding=1),
                nn.BatchNorm2d(n_filters), nn.ReLU(inplace=True)
            ))
        self.policy_vocab = nn.Sequential(nn.Conv2d(n_filters, 32, 1), nn.ReLU(), nn.Flatten(), nn.Linear(32*8*8, 512), nn.ReLU(), nn.Linear(512, vocab_size))
        self.from_head = nn.Sequential(nn.Conv2d(n_filters, 16, 1), nn.ReLU(), nn.Flatten(), nn.Linear(16*8*8, 256), nn.ReLU(), nn.Linear(256, 64))
        self.to_head   = nn.Sequential(nn.Conv2d(n_filters, 16, 1), nn.ReLU(), nn.Flatten(), nn.Linear(16*8*8, 256), nn.ReLU(), nn.Linear(256, 64))
        self.promo_head = nn.Sequential(nn.Conv2d(n_filters, 8, 1), nn.ReLU(), nn.Flatten(), nn.Linear(8*8*8, 64), nn.ReLU(), nn.Linear(64, 5))
        self.value_head = nn.Sequential(nn.Conv2d(n_filters, 8, 1), nn.ReLU(), nn.Flatten(), nn.Linear(8*8*8, 64), nn.ReLU(), nn.Linear(64,1), nn.Tanh())

    def forward(self, x):
        z = self.conv_in(x)
        for b in self.blocks:
            z = z + b(z)
        pv = self.policy_vocab(z)
        f = self.from_head(z); t = self.to_head(z); pr = self.promo_head(z); val = self.value_head(z).squeeze(-1)
        return pv, f, t, pr, val

# ---------- Encoder ----------

def encode_fen_to_numpy(fen: str) -> np.ndarray:
    fen = sanitize_fen(fen)
    try:
        board = chess.Board(fen)
    except Exception:
        board = chess.Board()
    planes = np.zeros((NUM_PLANES,8,8), dtype=np.float32)
    piece_plane = {chess.PAWN:0, chess.KNIGHT:1, chess.BISHOP:2, chess.ROOK:3, chess.QUEEN:4, chess.KING:5}
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            idx = piece_plane[piece.piece_type]
            r = 7 - chess.square_rank(sq); c = chess.square_file(sq)
            if piece.color == chess.WHITE:
                planes[idx, r, c] = 1.0
            else:
                planes[idx+6, r, c] = 1.0
    planes[12,:,:] = 1.0 if board.turn==chess.WHITE else 0.0
    planes[13,:,:] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    planes[14,:,:] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    planes[15,:,:] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    planes[16,:,:] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    try:
        half = int(board.fen().split()[4])
    except Exception:
        half = 0
    planes[17,:,:] = float(half)/100.0
    return planes

# ---------------- Loading model & vocab (single button flow) ----------------

def load_model_and_vocab_single(pt_path=None, json_path=None):
    """If pt_path is None, opens file dialog for .pt then finds or asks for matching .json.
       Returns (model, mvocab, inv, model_name)
    """
    if pt_path is None:
        pt_path = filedialog.askopenfilename(title='Select model (.pt/.pth)', filetypes=[('PyTorch','*.pt *.pth'),('All','*.*')])
        if not pt_path: return None
    if json_path is None:
        # try same-name json
        guess = pt_path.rsplit('.',1)[0] + '.json'
        if os.path.exists(guess): json_path = guess
        else:
            json_path = filedialog.askopenfilename(title='Select vocab (.json)', filetypes=[('JSON','*.json'),('All','*.*')])
            if not json_path:
                messagebox.showerror('Vocab missing', 'You must provide the move-vocab JSON (move->idx).')
                return None
    raw = json.load(open(json_path,'r',encoding='utf8'))
    mvocab = {}; inv = {}
    def looks_like_move(s): return isinstance(s,str) and re.match(r'^[a-h][1-8][a-h][1-8][qrbn]?$', s)
    keys = list(raw.keys())[:10]; vals = list(raw.values())[:10]
    if all(looks_like_move(k) for k in keys):
        for k,v in raw.items(): mvocab[k]=int(v); inv[int(v)]=k
    elif all(str(k).isdigit() for k in keys):
        for k,v in raw.items(): inv[int(k)]=v; mvocab[v]=int(k)
    else:
        if all(looks_like_move(str(v)) for v in vals):
            for k,v in raw.items(): inv[int(k)]=v; mvocab[v]=int(k)
        else:
            i=0
            for k,v in raw.items(): mvocab[str(k)]=i; inv[i]=str(k); i+=1
    ckpt = torch.load(pt_path, map_location='cpu')
    state = None
    if isinstance(ckpt, dict):
        for key in ('state','state_dict','model_state','model'):
            if key in ckpt: state = ckpt[key]; break
        if state is None: state = ckpt
    else:
        state = ckpt
    vocab_size = max(inv.keys())+1 if inv else max(mvocab.values())+1
    model = CloneNet(in_planes=NUM_PLANES, n_filters=80, n_blocks=3, vocab_size=vocab_size)
    try:
        model.load_state_dict(state)
    except Exception:
        try: model.load_state_dict(state, strict=False)
        except Exception as e: print('Warning: model load non-strict failed', e)
    model.eval()
    return model, mvocab, inv, os.path.basename(pt_path)

# ---------------- Stockfish eval helper ----------------

def eval_moves_with_stockfish(stockfish_path, fen, moves_list, depth=DEFAULT_SF_DEPTH):
    if not stockfish_path or not os.path.exists(stockfish_path):
        raise FileNotFoundError('Stockfish not found: '+str(stockfish_path))
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    board = chess.Board(fen)
    scores = {}
    for mv in moves_list:
        try:
            b2 = board.copy(); b2.push(chess.Move.from_uci(mv))
            info = engine.analyse(b2, chess.engine.Limit(depth=depth))
            sc = info.get('score')
            try:
                cp = sc.white().score(mate_score=100000)
            except Exception:
                try: mate = sc.white().mate(); cp = 100000 if mate>0 else -100000
                except Exception: cp = 0
            scores[mv] = int(cp) if board.turn==chess.WHITE else -int(cp)
        except Exception:
            scores[mv] = -999999
    try: engine.quit()
    except: pass
    return scores

# ---------------- Helpers: softmax & classify ----------------

def softmax(x):
    x = np.array(x, dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    return e / (e.sum()+1e-12)


def classify_move_quality_english(fen, chosen_move, pool_moves, stockfish_path=None, sf_depth=DEFAULT_SF_DEPTH):
    """Return quality label among: ['error','blunder','mistake','inaccuracy','poor','good','excellent','best','brilliant','unknown']"""
    try:
        if stockfish_path and os.path.exists(stockfish_path):
            evals = eval_moves_with_stockfish(stockfish_path, fen, pool_moves, depth=sf_depth)
        else:
            evals = None
    except Exception:
        evals = None
    if not evals:
        return 'unknown'
    best_mv, best_sc = max(evals.items(), key=lambda x:x[1])
    chosen_sc = evals.get(chosen_move, -999999)
    delta = best_sc - chosen_sc
    if abs(best_sc) > 90000 or abs(chosen_sc) > 90000:
        return 'brilliant' if chosen_move==best_mv else 'blunder'
    if delta > 1000: return 'blunder'
    if delta > 300: return 'mistake'
    if delta > 100: return 'inaccuracy'
    if chosen_move == best_mv: return 'best'
    if chosen_sc >= best_sc - 50: return 'excellent'
    if chosen_sc >= best_sc - 150: return 'good'
    if delta > 0: return 'poor'
    return 'unknown'

# ---------------- Hybrid selection ----------------

def select_move_hybrid(fen, book, model, mvocab, inv, stockfish_path=None, book_moves=BOOK_DEFAULT_DEPTH_FULLMOVES, topk=DEFAULT_MODEL_TOPK, temp=DEFAULT_TEMP, sf_depth=DEFAULT_SF_DEPTH, blunder_cp=DEFAULT_BLUNDER_CP, preserve_prob=DEFAULT_PRESERVE_PROB, force_sf=False):
    # 1) book
    key = fen_key_for_book(fen)
    if book and key in book:
        freq = book[key]
        moves = list(freq.keys()); counts = np.array([freq[m] for m in moves], dtype=float)
        probs = counts / counts.sum()
        return np.random.choice(moves, p=probs), 'book'
    # 2) model
    x = torch.from_numpy(encode_fen_to_numpy(fen)).unsqueeze(0).float()
    with torch.no_grad():
        pv_logits, f_logits, t_logits, pr_logits, val = model(x)
    pv = pv_logits.squeeze(0).cpu().numpy()
    board = chess.Board(fen)
    legal = [m.uci() for m in board.legal_moves]
    cand = []
    for idx, score in enumerate(pv):
        if idx in inv:
            u = inv[idx]
            if u in legal:
                cand.append((idx, float(score), u))
    if not cand:
        # factorized fallback
        f_arr = softmax(f_logits.squeeze(0).cpu().numpy())
        t_arr = softmax(t_logits.squeeze(0).cpu().numpy())
        best=None; best_sc=-1e9
        for u in legal:
            fr = chess.parse_square(u[:2]); to = chess.parse_square(u[2:4])
            sc = f_arr[fr] + t_arr[to]
            if sc > best_sc: best_sc=sc; best=u
        return best, 'factorized'
    cand.sort(key=lambda x:-x[1])
    top = cand[:max(1,topk)]; moves = [m for _,_,m in top]; scores = np.array([s for _,s,_ in top], dtype=float)
    probs = softmax(scores / max(1e-6, temp))
    model_choice = np.random.choice(moves, p=probs)
    # 3) stockfish fixer
    if stockfish_path and os.path.exists(stockfish_path):
        try:
            evals = eval_moves_with_stockfish(stockfish_path, fen, moves, depth=sf_depth)
            sf_best = max(evals.items(), key=lambda x:x[1])[0]
            sf_best_score = evals[sf_best]
            model_score = evals.get(model_choice, -999999)
            delta = sf_best_score - model_score
            if delta > blunder_cp or force_sf:
                if force_sf:
                    return sf_best, 'sf_best_forced'
                if random.random() < preserve_prob:
                    return model_choice, 'model_preserved_blunder'
                safe = [m for m,s in evals.items() if sf_best_score - s <= blunder_cp]
                if safe:
                    svals = np.array([evals[m] for m in safe], dtype=float)
                    ps = np.exp(svals - svals.max()); ps = ps / ps.sum()
                    return np.random.choice(safe, p=ps), 'model_fixed_by_sf'
                else:
                    return sf_best, 'sf_best'
            else:
                return model_choice, 'model_ok'
        except Exception:
            return model_choice, 'model_engine_err'
    else:
        return model_choice, 'model_no_sf'

# ---------------- UI: polished Tkinter app ----------------
class ChessApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Chess Player Cloner — Play like the clone')
        self.geometry('1100x780')
        self.configure(bg='#0d1117')
        # state
        self.model = None; self.vocab = None; self.inv = None; self.model_name = '(none)'
        self.book = None; self.book_path = None
        self.stockfish_path = STOCKFISH_REL_PATH if os.path.exists(STOCKFISH_REL_PATH) else None
        self.side = 'white'
        self.board = chess.Board()
        self.selected_sq = None
        # hybrid params (user-configurable via listbox)
        self.topk = DEFAULT_MODEL_TOPK; self.temp = DEFAULT_TEMP
        # default weights
        self.preserve_prob = DEFAULT_PRESERVE_PROB; self.blunder_cp = DEFAULT_BLUNDER_CP; self.sf_depth = DEFAULT_SF_DEPTH; self.force_sf=False
        self._build_ui()
        self.after(150, self._maybe_engine_first)

    def _build_ui(self):
        # Top toolbar
        toolbar = tk.Frame(self, bg='#0b1220', height=60)
        toolbar.pack(side='top', fill='x')
        style_btn = {'padx':8,'pady':6}
        load_btn = tk.Button(toolbar, text='Load Model', command=self._on_load_model, bg='#2b8cff', fg='white', **style_btn)
        load_btn.pack(side='left', padx=8, pady=8)
        sf_btn = tk.Button(toolbar, text='Load Stockfish', command=self._on_choose_stockfish, bg='#444', fg='white', **style_btn)
        sf_btn.pack(side='left', padx=8, pady=8)
        book_btn = tk.Button(toolbar, text='Load Opening Book (folder)', command=self._on_load_book_thread, bg='#3aa849', fg='white', **style_btn)
        book_btn.pack(side='left', padx=8, pady=8)
        new_btn = tk.Button(toolbar, text='New Game', command=self._on_new_game, bg='#8b5cf6', fg='white', **style_btn)
        new_btn.pack(side='left', padx=8, pady=8)
        undo_btn = tk.Button(toolbar, text='Undo', command=self._on_undo, bg='#ef4444', fg='white', **style_btn)
        undo_btn.pack(side='left', padx=8, pady=8)

        # credits
        cred = tk.Label(toolbar, text='Coded By Mohammad Taha Gorji | GitHub: github.com/mr-r0ot/Chess-player-Cloner-AI', bg='#0b1220', fg='#9aa1a6')
        cred.pack(side='right', padx=10)
        self.model_label = tk.Label(toolbar, text='Model: (none)', bg='#0b1220', fg='white')
        self.model_label.pack(side='right')

        # center area
        center = tk.Frame(self, bg='#081018')
        center.pack(fill='both', expand=True, padx=12, pady=12)

        # board canvas
        self.canvas = tk.Canvas(center, width=600, height=600, bg='#101418', highlightthickness=0)
        self.canvas.pack(side='left', padx=12, pady=8)
        self.canvas.bind('<Button-1>', self._on_canvas_click)

        # right panel: controls and logs
        right = tk.Frame(center, bg='#081018', width=420)
        right.pack(side='right', fill='y', padx=12)

        # side selection
        side_frame = tk.Frame(right, bg='#081018')
        side_frame.pack(fill='x', pady=(6,12))
        tk.Label(side_frame, text='Play as:', bg='#081018', fg='white').pack(side='left')
        self.white_btn = tk.Button(side_frame, text='White', command=lambda:self._set_side('white'), bg='#ffffff', fg='#000')
        self.black_btn = tk.Button(side_frame, text='Black', command=lambda:self._set_side('black'), bg='#111', fg='#fff')
        self.white_btn.pack(side='left', padx=6); self.black_btn.pack(side='left', padx=6)
        self._refresh_side_buttons()

        # weight selection
        tk.Label(right, text='Model Strength / Style Weight', bg='#081018', fg='white', font=('Segoe UI',10,'bold')).pack(anchor='w', pady=(6,2))
        self.weights_lb = tk.Listbox(right, height=len(WEIGHT_STRATEGIES), activestyle='none')
        for name,_params in WEIGHT_STRATEGIES:
            self.weights_lb.insert('end', name)
        self.weights_lb.selection_set(2)  # default: Preserve style (max)
        self.weights_lb.pack(fill='x', pady=(0,6))
        apply_btn = tk.Button(right, text='Apply Strategy', command=self._apply_weight_strategy, bg='#2b8cff', fg='white')
        apply_btn.pack(fill='x', pady=(0,12))

        # status info
        tk.Label(right, text='Status', bg='#081018', fg='white', font=('Segoe UI',10,'bold')).pack(anchor='w')
        self.status_var = tk.StringVar(value='Ready')
        tk.Label(right, textvariable=self.status_var, bg='#081018', fg='#cfd8dc', justify='left', wraplength=380).pack(anchor='w')

        # advantage bar
        tk.Label(right, text='Model Value (approx cp)', bg='#081018', fg='white').pack(anchor='w', pady=(10,0))
        self.adv_canvas = tk.Canvas(right, width=60, height=220, bg='#0b1220', highlightthickness=0)
        self.adv_canvas.pack(pady=6)

        # move log
        tk.Label(right, text='Move Log', bg='#081018', fg='white', font=('Segoe UI',10,'bold')).pack(anchor='w', pady=(10,0))
        self.log = tk.Text(right, height=14, width=50, bg='#071018', fg='white', wrap='none')
        self.log.pack()

        # bottom statusbar
        self.statusbar = tk.Label(self, text='Welcome! Load model and book to begin', bd=1, relief='sunken', anchor='w')
        self.statusbar.pack(side='bottom', fill='x')

        self._draw_board()

    # ---------- UI helpers ----------
    def _set_side(self, s):
        self.side = s; self._refresh_side_buttons(); self._on_new_game()
    def _refresh_side_buttons(self):
        if self.side=='white':
            self.white_btn.config(relief='sunken', bg='#fff', fg='#000'); self.black_btn.config(relief='raised', bg='#111', fg='#fff')
        else:
            self.black_btn.config(relief='sunken', bg='#111', fg='#fff'); self.white_btn.config(relief='raised', bg='#fff', fg='#000')

    def _apply_weight_strategy(self):
        sel = self.weights_lb.curselection()
        if not sel: return
        choice = sel[0]
        _, params = WEIGHT_STRATEGIES[choice]
        self.preserve_prob = params['preserve_prob']
        self.blunder_cp = params['blunder_cp']
        self.sf_depth = params['sf_depth']
        self.force_sf = params.get('force_sf', False)
        self.status_var.set(f"Applied strategy: {WEIGHT_STRATEGIES[choice][0]}")
        self.statusbar.config(text=f"Strategy set: {WEIGHT_STRATEGIES[choice][0]}")

    def _on_load_model(self):
        try:
            res = load_model_and_vocab_single()
            if not res: return
            model, mvocab, inv, name = res
            self.model, self.vocab, self.inv, self.model_name = model, mvocab, inv, name
            self.model_label.config(text=f'Model: {self.model_name}')
            self.statusbar.config(text='Model & vocab loaded')
            self.status_var.set('Model loaded')
            # if playing black and board empty, engine plays first
            self._maybe_engine_first()
        except Exception as e:
            messagebox.showerror('Load Error', str(e))

    def _on_choose_stockfish(self):
        path = filedialog.askopenfilename(title='Select Stockfish executable', filetypes=[('exe','*.exe'),('All','*.*')])
        if not path: return
        self.stockfish_path = path
        self.statusbar.config(text='Stockfish set: '+path)

    def _on_load_book_thread(self):
        folder = filedialog.askdirectory(title='Select folder with dataset (CSVs/PGNs)')
        if not folder: return
        # run in thread to avoid UI freeze
        t = threading.Thread(target=self._build_book, args=(folder,), daemon=True)
        t.start()

    def _build_book(self, folder):
        try:
            self.statusbar.config(text='Building opening book (this can take time)...')
            book = build_opening_book_from_folder(folder, book_moves=BOOK_DEFAULT_DEPTH_FULLMOVES, max_rows=300000, progress_callback=lambda msg: self._set_status(msg))
            savepath = os.path.join(folder, f'opening_book_{BOOK_DEFAULT_DEPTH_FULLMOVES}moves.json')
            save_opening_book(book, savepath)
            self.book = book; self.book_path = savepath
            self._set_status(f'Opening book built: {len(book)} positions (saved to {savepath})')
            messagebox.showinfo('Book built', f'Opening book built with {len(book)} positions. Saved to:\n{savepath}')
        except Exception as e:
            messagebox.showerror('Book error', str(e))
            self._set_status('Failed to build opening book')

    def _set_status(self, msg):
        # thread-safe status update
        self.after(0, lambda: self.status_var.set(msg))

    def _on_new_game(self):
        self.board = chess.Board(); self.selected_sq=None; self.log.delete('1.0','end'); self._draw_board(); self.statusbar.config(text='New game')
        self._maybe_engine_first()

    def _on_undo(self):
        if len(self.board.move_stack)==0:
            self.statusbar.config(text='Cannot undo')
            return
        self.board.pop(); self._draw_board(); self.statusbar.config(text='Undid last move')

    def _maybe_engine_first(self):
        if getattr(self,'model',None) and self.side=='black' and len(self.board.move_stack)==0:
            # engine plays white
            self.after(300, self._engine_play)

    # ---------- Board interactions ----------
    def _on_canvas_click(self, event):
        size = 600//8
        col = event.x // size; row = event.y // size
        if self.side=='white': file = col; rank = 7-row
        else: file = 7-col; rank = row
        sq = chess.square(file, rank)
        piece = self.board.piece_at(sq)
        if self.selected_sq is None:
            if piece is None: return
            if (self.side=='white' and piece.color!=chess.WHITE) or (self.side=='black' and piece.color!=chess.BLACK): return
            self.selected_sq = sq; self._draw_board(highlight=sq); return
        else:
            # save previous fen BEFORE move for classification
            prevfen = self.board.fen()
            # create candidate: handle promotions and ambiguous moves
            candidate = chess.Move(self.selected_sq, sq)
            if candidate not in self.board.legal_moves:
                found = None
                for lm in self.board.legal_moves:
                    if lm.from_square==self.selected_sq and lm.to_square==sq:
                        found = lm; break
                if found is None:
                    self.selected_sq=None; self._draw_board(); return
                candidate = found
            # make move
            self.board.push(candidate)
            # build pool for classification
            pool = []
            try:
                if self.book and fen_key_for_book(prevfen) in self.book:
                    pool = list(self.book[fen_key_for_book(prevfen)].keys())
                else:
                    x = torch.from_numpy(encode_fen_to_numpy(prevfen)).unsqueeze(0).float()
                    with torch.no_grad(): pv_logits, f_logits, t_logits, pr_logits, v = self.model(x)
                    pv = pv_logits.squeeze(0).cpu().numpy(); legal = [m.uci() for m in chess.Board(prevfen).legal_moves]
                    cand = []
                    for idx, sc in enumerate(pv):
                        if idx in self.inv:
                            u = self.inv[idx]
                            if u in legal: cand.append((idx, float(sc), u))
                    cand.sort(key=lambda x:-x[1]); pool = [u for _,_,u in cand[:self.topk]] if cand else [m.uci() for m in chess.Board(prevfen).legal_moves]
            except Exception:
                pool = [m.uci() for m in chess.Board(prevfen).legal_moves]
            qlabel = classify_move_quality_english(prevfen, candidate.uci(), pool, self.stockfish_path, sf_depth=self.sf_depth)
            self._log_move(candidate.uci(), f'Player | {qlabel}')
            self.selected_sq=None; self._draw_board()
            if self.board.is_game_over(): self._on_game_over('Player'); return
            self.after(150, self._engine_play)

    def _engine_play(self):
        fen = self.board.fen()
        if not getattr(self,'model',None) or not getattr(self,'inv',None):
            self.statusbar.config(text='Model or vocab missing. Load model first.')
            return
        mv, src = select_move_hybrid(fen, self.book, self.model, self.vocab, self.inv, stockfish_path=self.stockfish_path, book_moves=BOOK_DEFAULT_DEPTH_FULLMOVES, topk=self.topk, temp=self.temp, sf_depth=self.sf_depth, blunder_cp=self.blunder_cp, preserve_prob=self.preserve_prob, force_sf=getattr(self,'force_sf',False))
        if not mv:
            self.statusbar.config(text='No move produced')
            return
        # build pool for classification
        pool = []
        try:
            x = torch.from_numpy(encode_fen_to_numpy(fen)).unsqueeze(0).float()
            with torch.no_grad(): pv_logits, f_logits, t_logits, pr_logits, v = self.model(x)
            pv = pv_logits.squeeze(0).cpu().numpy(); legal=[m.uci() for m in self.board.legal_moves]
            cand=[]
            for idx, sc in enumerate(pv):
                if idx in self.inv:
                    u=self.inv[idx]
                    if u in legal: cand.append((idx, float(sc), u))
            cand.sort(key=lambda x:-x[1]); pool = [u for _,_,u in cand[:self.topk]] if cand else [m.uci() for m in self.board.legal_moves]
        except Exception:
            pool = [m.uci() for m in self.board.legal_moves]
        # push
        try:
            self.board.push(chess.Move.from_uci(mv))
        except Exception:
            # invalid move? skip
            self.statusbar.config(text='Engine gave illegal move. Skipping')
            return
        qlabel = classify_move_quality_english(fen, mv, pool, self.stockfish_path, sf_depth=self.sf_depth)
        self._log_move(mv, f'{src} | {qlabel}')
        self._draw_board()
        if self.board.is_game_over(): self._on_game_over(src)

    def _on_game_over(self, who):
        res = f'Game over: {self.board.result()}'
        if self.board.is_checkmate(): res += ' — checkmate'
        elif self.board.is_stalemate(): res += ' — stalemate'
        self.statusbar.config(text=res)
        messagebox.showinfo('Game over', res)

    def _log_move(self, uci, info=''):
        move_no = len(self.board.move_stack)
        txt = f"{move_no}. {uci}  ({info})\n"
        self.log.insert('end', txt); self.log.see('end')
        self.statusbar.config(text=f'Last: {uci} — {info}')
        # update advantage bar using model value if available
        try:
            if getattr(self,'model',None):
                x = torch.from_numpy(encode_fen_to_numpy(self.board.fen())).unsqueeze(0).float()
                with torch.no_grad(): pv,f,t,pr,v = self.model(x); val = float(v.cpu().numpy().squeeze()) if hasattr(v,'cpu') else float(v)
                cp = int(val*1000)
            else:
                cp = 0
        except Exception:
            cp = 0
        self._draw_advantage(cp)

    def _draw_advantage(self, cp:int):
        self.adv_canvas.delete('all')
        h = 220; w = 60
        cp = max(-1000, min(1000, cp))
        mid = h//2
        if cp >= 0:
            top = mid - int((cp/1000.0)*mid)
            self.adv_canvas.create_rectangle(0,0,w,top, fill='#ffffff', outline='')
            self.adv_canvas.create_rectangle(0,top,w,h, fill='#2d7a2d', outline='')
        else:
            bot = mid + int(((-cp)/1000.0)*mid)
            self.adv_canvas.create_rectangle(0,0,w,mid, fill='#2d7a2d', outline='')
            self.adv_canvas.create_rectangle(0,mid,w,bot, fill='#771212', outline='')
        self.adv_canvas.create_text(w//2, h-8, text=f"{cp} cp", fill='white')

    def _draw_board(self, highlight=None):
        self.canvas.delete('all')
        size = 600//8
        light = '#f0d9b5'; dark = '#b58863'
        for r in range(8):
            for c in range(8):
                x0=c*size; y0=r*size; x1=x0+size; y1=y0+size
                color = light if (r+c)%2==0 else dark
                self.canvas.create_rectangle(x0,y0,x1,y1, fill=color, outline='')
        for sq in chess.SQUARES:
            piece = self.board.piece_at(sq)
            if not piece: continue
            rank = chess.square_rank(sq); file = chess.square_file(sq)
            if self.side=='white': r = 7-rank; c = file
            else: r = rank; c = 7-file
            x = c*size + size//2; y = r*size + size//2
            rad = size*0.42
            if piece.color==chess.WHITE:
                self.canvas.create_oval(x-rad, y-rad, x+rad, y+rad, fill='#ffffff', outline='#cccccc')
                fill = '#111111'
            else:
                self.canvas.create_oval(x-rad, y-rad, x+rad, y+rad, fill='#111111', outline='#000000')
                fill = '#ffffff'
            glyph = {'P':'♙','N':'♘','B':'♗','R':'♖','Q':'♕','K':'♔','p':'♟','n':'♞','b':'♝','r':'♜','q':'♛','k':'♚'}[piece.symbol()]
            self.canvas.create_text(x, y, text=glyph, font=('Segoe UI Symbol', int(size*0.45)), fill=fill)
        # highlight last move squares
        if self.board.move_stack:
            last = self.board.move_stack[-1]
            for s in (last.from_square, last.to_square):
                rank = chess.square_rank(s); file = chess.square_file(s)
                if self.side=='white': r = 7-rank; c = file
                else: r = rank; c = 7-file
                x0=c*size; y0=r*size; x1=x0+size; y1=y0+size
                self.canvas.create_rectangle(x0,y0,x1,y1, outline='#ffef7a', width=3)

# ---------- run ----------
if __name__ == '__main__':
    app = ChessApp()
    app.mainloop()
