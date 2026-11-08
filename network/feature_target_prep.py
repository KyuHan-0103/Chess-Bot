import json
import numpy as np
import pyarrow.dataset as ds
from core.notation import from_fen
from core.board import Chess
import os

MIN_PLY = 12
TRAIN_DIR = "2016_fen_eval/train"
OUT_DIR = '2016_fen_eval/prepped_train'
def position_key(fen : str):
    return " ".join(fen.split()[:4])
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    train = ds.dataset(TRAIN_DIR, format='parquet')
    filter = train.to_batches(
        columns=['fen', 'eval'],
        filter= (ds.field('ply') >= MIN_PLY) & (ds.field('eval').is_valid())
    )

    #Remove FENS that appear in val from train
    val_fens = ds.dataset('2016_fen_eval/val', format='parquet').to_table(columns=['fen'])
    val_keys = {position_key(f) for f in val_fens.column('fen').to_pylist()}
    dropped = 0
    features = []
    targets = []
    stm = []
    for batch in filter:
        fens = batch.column('fen').to_pylist()
        evals = batch.column('eval').to_numpy(zero_copy_only=False)
        game_indices = []
        game_evals = []
        game_stm = []
        for fen, ev in zip(fens, evals):
            if position_key(fen) in val_keys:
                dropped += 1
                continue

            chess_game : Chess = from_fen(fen)
            board = chess_game.chess_board
            indices = []

            for piece_idx in chess_game.white_pieces:
                s = piece_idx[0] * 8 + piece_idx[1]
                indices.append((board[piece_idx[0]][piece_idx[1]] + 5) * 64 + s)
            for piece_idx in chess_game.black_pieces:
                s = piece_idx[0] * 8 + piece_idx[1]
                indices.append((board[piece_idx[0]][piece_idx[1]] + 6) * 64 + s)
            game_indices.append(indices)
            game_evals.append(ev)
            game_stm.append(1 if fen.split()[1] == 'b' else 0)
        
        feats = np.full((len(game_indices), 32), 768, dtype=np.int16)
        for i, indices in enumerate(game_indices):
            feats[i, :len(indices)] = indices
        features.append(feats)
        targets.append(np.array(game_evals, dtype=np.int32))
        stm.append(np.array(game_stm, dtype=np.uint8))


    features = np.concatenate(features, axis=0)
    targets = np.concatenate(targets, axis=0)
    stm = np.concatenate(stm, axis=0)
    meta = {
        'source': TRAIN_DIR,
        'min_ply': MIN_PLY,
        'rows': len(features),
        'dropped_val_overlap' : dropped,
        "stm" : '1 = black to move : 0 = white to move',
        'eval_clamp' : None
        }
    np.save(f'{OUT_DIR}/features.npy', features)
    np.save(f'{OUT_DIR}/targets.npy', targets)
    np.save(f'{OUT_DIR}/stm.npy', stm)
    with open(f"{OUT_DIR}/meta.json", 'w') as f:
        json.dump(meta, f, indent=2)


if __name__ == '__main__':
    main()
