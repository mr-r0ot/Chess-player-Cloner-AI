# Chess-player-Cloner-AI

```
PS D:\Project\ChessClone> python .\CopyAllGames.py witty_alien --wins-only --outdir dataset --format all
Checking users...
username: witty_alien
name: Volen Dyulgerov
url: https://www.chess.com/member/Witty_Alien
joined: 2015-09-23
ratings:
  chess_daily: 764
  chess960_daily: 1424
  chess_rapid: 2228
  chess_bullet: 2606
  chess_blitz: 2559
---
Saving to: dataset\witty_alien
Processing witty_alien ...
archives(witty_alien): 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 120/120 [27:49<00:00, 13.91s/it] 
-> saved 87107 games (4824641 position rows)
Done. Report saved to dataset\witty_alien\report.json
```



and

```

PS D:\Project\0SuperProjects\1AI\ChessClone> python Chess_Cloner_AI.py --dataset "D:\Project\0SuperProjects\1AI\ChessClone\dataset\witty_alien\witty_alien" \ --outmodel witty_clone.pt --vocab witty_vocab.json --epochs 6 --batch 128 --max-rows 100000
Expanding dataset paths...
Found 1 CSV(s):
  D:\Project\0SuperProjects\1AI\ChessClone\dataset\witty_alien\witty_alien\witty_alien_positions_moves.csv
Sampling up to 100000 rows from CSVs...
Loaded rows: 100000
Building move vocab...
vocab size: 1853
Encoding positions to tensors (this may take a few moments)...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100000/100000 [00:16<00:00, 6144.18it/s] 
Dataset prepared: samples: 100000 mem approx: 439.45 MB
Constructing model...
Training (CPU-only). epochs: 6 batch: 128
Epoch 1 avg loss: 10.2387
Epoch 2 avg loss: 8.1409                                                                                                                                                 
Epoch 3 avg loss: 7.0837                                                                                                                                                 
Epoch 4 avg loss: 6.4659                                                                                                                                                 
Epoch 5 avg loss: 5.9732                                                                                                                                                 
Epoch 6 avg loss: 5.5690                                                                                                                                                 
Saved model: witty_clone.pt vocab: witty_vocab.json
Done.
```
