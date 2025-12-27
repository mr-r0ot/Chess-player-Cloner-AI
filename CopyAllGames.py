#!/usr/bin/env python3
"""
CopyAllGames.py (improved)
Robust CLI to download chess.com games for one or more usernames.
Improves networking stability, prints debug info when user not found,
and uses a sensible User-Agent + retries.

Dependencies:
  pip install requests python-chess tqdm

Usage:
  python CopyAllGames.py Witty_Alien --wins-only --outdir dataset --format all --verbose
"""
from __future__ import annotations
import argparse, os, sys, time, json, csv, re, io
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import chess.pgn
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

API_BASE = "https://api.chess.com/pub/player/"

def make_session() -> requests.Session:
    s = requests.Session()
    # polite user-agent (some APIs block empty UA)
    s.headers.update({"User-Agent": "CopyAllGames/1.0 (+https://github.com/mr-r0ot/Chess-player-Cloner-AI)"})
    # retries for transient network errors
    retries = Retry(total=5, backoff_factor=0.6,
                    status_forcelist=(429, 500, 502, 503, 504),
                    allowed_methods=frozenset(["HEAD", "GET", "OPTIONS"]))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    return s

SESSION = make_session()

def norm_username(u: str) -> str:
    return u.strip().lower()

def get_profile(username: str, timeout: int = 10, verbose: bool = False) -> Optional[Dict[str, Any]]:
    username_norm = norm_username(username)
    url = urljoin(API_BASE, username_norm)
    try:
        if verbose: print(f"[DEBUG] GET {url}")
        r = SESSION.get(url, timeout=timeout)
        if verbose:
            print(f"[DEBUG] status_code={r.status_code}")
            # only print a bit of body to avoid huge logs
            print(f"[DEBUG] resp_snippet={r.text[:300]!r}")
        if r.status_code == 200:
            return r.json()
        else:
            # return detailed error info in verbose mode
            return None
    except requests.RequestException as e:
        if verbose:
            print(f"[DEBUG] requests exception while fetching profile: {e}")
        return None

def get_stats(username: str, timeout: int = 10):
    username_norm = norm_username(username)
    url = urljoin(API_BASE, f"{username_norm}/stats")
    try:
        if True:
            # use same session (retries) and print less verbose by default
            r = SESSION.get(url, timeout=timeout)
            if r.status_code == 200:
                return r.json()
        return None
    except requests.RequestException:
        return None

def get_archives(username: str, timeout: int = 10) -> List[str]:
    username_norm = norm_username(username)
    url = urljoin(API_BASE, f"{username_norm}/games/archives")
    try:
        r = SESSION.get(url, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return data.get("archives", [])
    except requests.RequestException:
        return []

def fetch_archive(archive_url: str, timeout: int = 20) -> Optional[Dict[str, Any]]:
    try:
        r = SESSION.get(archive_url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None

def username_is_winner(game_json: Dict[str, Any], username: str) -> bool:
    username = norm_username(username)
    white = game_json.get("white", {}).get("username", "").lower()
    black = game_json.get("black", {}).get("username", "").lower()
    wres = game_json.get("white", {}).get("result")
    bres = game_json.get("black", {}).get("result")
    if wres == "win" and white == username:
        return True
    if bres == "win" and black == username:
        return True
    return False

def sanitize_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def pgn_to_rows(game_pgn: str, game_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    try:
        game = chess.pgn.read_game(io.StringIO(game_pgn))
    except Exception:
        return rows
    if game is None:
        return rows
    board = game.board()
    node = game
    ply = 0
    while node.variations:
        next_node = node.variation(0)
        move = next_node.move
        fen_before = board.fen()
        move_uci = move.uci()
        try:
            move_san = board.san(move)
        except Exception:
            move_san = ""
        side = 'white' if board.turn == chess.WHITE else 'black'
        row = {
            'game_id': game_meta.get('url', '') or sanitize_filename(game_meta.get('end_time', str(time.time()))),
            'ply': ply,
            'fen_before': fen_before,
            'move_uci': move_uci,
            'move_san': move_san,
            'side_to_move': side,
            'result': game.headers.get('Result', game_meta.get('white', {}).get('result', '')),
            'white': game.headers.get('White', game_meta.get('white', {}).get('username', '')),
            'black': game.headers.get('Black', game_meta.get('black', {}).get('username', '')),
        }
        rows.append(row)
        board.push(move)
        node = next_node
        ply += 1
    return rows

def process_user(username: str, out_base: str, wins_only: bool, formats: List[str], since: Optional[str], until: Optional[str], threads: int = 4, verbose: bool = False) -> Dict[str, Any]:
    username_norm = norm_username(username)
    user_dir = os.path.join(out_base, username_norm)
    ensure_dir(user_dir)
    archives = get_archives(username)
    if not archives:
        return {'username': username, 'archives': 0, 'games_saved': 0, 'error': 'no_archives_or_user_not_found'}
    def in_range(archive_url: str) -> bool:
        if not (since or until):
            return True
        m = archive_url.split('/')[-1]
        ym = m.replace('/', '-')
        if since and ym < since:
            return False
        if until and ym > until:
            return False
        return True
    archives = [a for a in archives if in_range(a)]
    saved = 0
    total_games = 0
    dataset_rows = []
    with ThreadPoolExecutor(max_workers=threads) as ex:
        futures = {ex.submit(fetch_archive, a): a for a in archives}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"archives({username})"):
            archive_url = futures[fut]
            data = fut.result()
            if not data:
                if verbose:
                    print(f"[WARN] failed archive {archive_url}")
                continue
            games = data.get('games', [])
            total_games += len(games)
            for g in games:
                try:
                    if wins_only and not username_is_winner(g, username):
                        continue
                    gid = g.get('url', '') or g.get('pgn', '')[:30]
                    safe_gid = sanitize_filename(gid.split('/')[-1])
                    if 'json' in formats or 'all' in formats:
                        json_path = os.path.join(user_dir, f"{safe_gid}.json")
                        with open(json_path, 'w', encoding='utf8') as jf:
                            json.dump(g, jf, ensure_ascii=False, indent=2)
                    pgn_text = g.get('pgn')
                    if pgn_text and ('pgn' in formats or 'all' in formats):
                        pgn_path = os.path.join(user_dir, f"{safe_gid}.pgn")
                        with open(pgn_path, 'w', encoding='utf8') as pf:
                            pf.write(pgn_text)
                    if ('moves' in formats or 'all' in formats) and pgn_text:
                        rows = pgn_to_rows(pgn_text, g)
                        dataset_rows.extend(rows)
                    saved += 1
                except Exception as e:
                    if verbose:
                        print(f"[ERROR] saving game {g.get('url')}: {e}")
    if ('moves' in formats or 'all' in formats) and dataset_rows:
        csv_path = os.path.join(user_dir, f"{username_norm}_positions_moves.csv")
        with open(csv_path, 'w', newline='', encoding='utf8') as cf:
            fieldnames = ['game_id','ply','fen_before','move_uci','move_san','side_to_move','result','white','black']
            writer = csv.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            for r in dataset_rows:
                writer.writerow(r)
    return {
        'username': username,
        'archives': len(archives),
        'total_games_found': total_games,
        'games_saved': saved,
        'dataset_rows': len(dataset_rows)
    }

def pretty_print_profile(profile: Dict[str, Any], stats: Optional[Dict[str, Any]] = None):
    if not profile:
        print("<no profile>")
        return
    print(f"username: {profile.get('username')}")
    if profile.get('name'):
        print(f"name: {profile.get('name')}")
    print(f"url: {profile.get('url')}")
    if 'joined' in profile:
        try:
            jt = time.gmtime(profile['joined'])
            print(f"joined: {time.strftime('%Y-%m-%d', jt)}")
        except Exception:
            pass
    if stats:
        ratings = {}
        for key, val in stats.items():
            if isinstance(val, dict) and 'last' in val and isinstance(val['last'], dict):
                ratings[key] = val['last'].get('rating')
        if ratings:
            print("ratings:")
            for k, v in ratings.items():
                print(f"  {k}: {v}")

def parse_args():
    p = argparse.ArgumentParser(description='Download chess.com games for one or more users (PGN/JSON/positions)')
    p.add_argument('usernames', nargs='+', help='One or more chess.com usernames')
    p.add_argument('--wins-only', '-w', action='store_true', help='Only save games that the username won')
    p.add_argument('--outdir', '-o', default='chess_datasets', help='Base output directory')
    p.add_argument('--format', choices=['pgn', 'json', 'moves', 'all'], default='all', help='What to save')
    p.add_argument('--group-name', default=None, help='If multiple usernames, provide group name to store under')
    p.add_argument('--no-interactive', action='store_true', help='Do not prompt; construct folder name automatically')
    p.add_argument('--since', default=None, help='Only include archives on/after this YYYY-MM (e.g. 2021-06)')
    p.add_argument('--until', default=None, help='Only include archives on/at/before this YYYY-MM')
    p.add_argument('--threads', type=int, default=6, help='Concurrent threads for downloading archives')
    p.add_argument('--verbose', '-v', action='store_true')
    return p.parse_args()

def main():
    args = parse_args()
    usernames = args.usernames
    wins_only = args.wins_only
    outdir = args.outdir
    fmt = args.format
    group_name = args.group_name
    no_interactive = args.no_interactive
    since = args.since
    until = args.until
    threads = args.threads
    verbose = args.verbose

    usernames = [norm_username(u) for u in usernames]

    profiles = {}
    stats_map = {}
    print("Checking users...")
    for u in usernames:
        prof = get_profile(u, verbose=verbose)
        if not prof:
            # improved error message & quick diagnostic
            print(f"ERROR: user '{u}' not found or API unreachable.")
            if verbose:
                print(f"[HINT] Try visiting in browser: https://www.chess.com/member/{u} or API endpoint: https://api.chess.com/pub/player/{u}")
            # do not exit yet; let user choose to continue or abort
            ans = None
            if not no_interactive:
                try:
                    ans = input(f"Continue despite missing user '{u}'? (y/N): ").strip().lower()
                except KeyboardInterrupt:
                    print("\naborted")
                    sys.exit(1)
            if ans != 'y':
                print("Aborting.")
                sys.exit(1)
        else:
            profiles[u] = prof
            st = get_stats(u)
            stats_map[u] = st
            pretty_print_profile(prof, st)
            print('---')

    if len(usernames) == 1:
        base_name = usernames[0]
    else:
        if group_name:
            base_name = sanitize_filename(group_name)
        elif no_interactive:
            base_name = '_'.join(usernames)
        else:
            try:
                inp = input('multiple usernames detected — enter a name for the output folder (or leave empty to use combined usernames): ').strip()
            except KeyboardInterrupt:
                print('\naborted')
                sys.exit(1)
            if inp:
                base_name = sanitize_filename(inp)
            else:
                base_name = '_'.join(usernames)

    out_base = os.path.join(outdir, base_name)
    ensure_dir(out_base)

    formats = [fmt]
    if fmt == 'all':
        formats = ['pgn', 'json', 'moves']

    print(f"Saving to: {out_base}")

    summaries = []
    for u in usernames:
        print(f"Processing {u} ...")
        s = process_user(u, out_base, wins_only, formats, since, until, threads=threads, verbose=verbose)
        summaries.append(s)
        print(f"-> saved {s.get('games_saved')} games ({s.get('dataset_rows',0)} position rows)")

    report_path = os.path.join(out_base, 'report.json')
    with open(report_path, 'w', encoding='utf8') as rf:
        json.dump({'summaries': summaries, 'params': vars(args)}, rf, indent=2)

    print('Done. Report saved to', report_path)

if __name__ == '__main__':
    main()
