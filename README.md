CHESS BOT

A chess game you can play, and a chess engine that plays it.

Description

The engine is an alpha-beta search with a depth, node, and time budget. Any combination can be set and whichever runs out first stops the search.

Rather than one recursive descent to a fixed depth, the search uses iterative deepening: depth 1, then depth 2, and so on up to max_depth, keeping the best move from the last iteration that finished. Depth 1 always runs to completion, so the search always comes back with a legal move no matter how small the budget.

Moves are searched in this order:

the best move from the transposition table, or the best move from the previous iteration
captures, sorted by MVV-LVA (Most Valuable Victim, Least Valuable Attacker)
killer moves and history scores for quiet moves
promotions, which get a bonus on top of the above

Move lists are not filtered for legality up front. A move is played, and if it leaves your own king attacked then make_move undoes it and returns false. That is cheaper than working out pins ahead of time, and it means only one place in the codebase decides what is legal.

When the side to move is in check the search spends an extra ply there, so it does not cut off in the middle of a forcing sequence. At the leaves a quiescence search follows captures and promotions until the position is quiet, and generates full evasions when in check.

Evaluation

Two evaluators, interchangeable behind one interface:

HandEvaluator -- material plus piece-square tables, with separate midgame and endgame values blended by how much material is left on the board, so a king that hides behind pawns in the opening walks toward the centre in an endgame without any special case code for it.
NetworkEvaluator -- a trained neural network, 768 inputs to 256 to 32 to 1, evaluated in NumPy so the engine never imports torch.

The network's 768 inputs are 12 piece types by 64 squares, encoded from the side to move's point of view: the board flips vertically and the colours swap when it is Black's turn, so "my knight on f5" is one feature rather than two.

Requirements

Python 3 (developed on 3.12). The GUI needs tkinter and Pillow. The network evaluator needs numpy. Training additionally needs torch and pyarrow. The engine with hand evaluation, perft and the test suites need none of these.

Running

Everything runs from the repo root, since the packages are imported by name.

python3 main.py                      play against the engine
python3 -m testing.perft --all       move generation and board invariants
python3 -m testing.final_tests       engine, notation and arena regression suite
python3 -m testing.san_tests         SAN parser regression suite
python3 -m testing.verify_npz        net.npz matches best.pt
python3 -m arena.match --random      engine vs a random mover, as a smoke test
python3 -m arena.match --network     hand evaluation vs the network

In the GUI, click or drag a piece to move it. Press space and the engine plays a move for whoever is to move.

Training the network

Three stages, each a separate script, because each one is rerun for different reasons.

network/data_parser.py          lichess movetext -> (fen, eval, ply, result) parquet
network/feature_target_prep.py  fen -> feature indices + targets, as .npy
network/base_network.py         train, writing best.pt

Stage 1 is the expensive one and only reruns when new raw data arrives. Stage 2 holds every filter -- opening plies, eval clamp, train/val dedupe -- so changing a policy costs minutes rather than hours. Stage 3 trains.

Training data is Lichess games carrying Stockfish [%eval] annotations. Evals are stored from White's point of view in stage 1 and flipped to the side to move in stage 2, where the same flip is applied to the features.

The train/validation split is by game, decided before any game is expanded into positions, because positions from one game are near-duplicates of each other.

TESTING

perft counts the leaves of the legal move tree and compares against published values, which is what catches move generation bugs. Passing --all also checks that the zobrist hash, the piece sets, the phase, undo_move and the FEN round trip all stay consistent at every node of the tree.

final_tests covers everything above move generation: the search, the transposition table, the search budget, endgame detection, FEN and UCI notation, and the match harness.

san_tests covers the PGN layer, cross-checking SAN against the UCI parser ply by ply on whole games.

verify_npz checks that the exported NumPy weights evaluate identically to the PyTorch checkpoint, over named positions and random playouts.

arena/match.py plays two players against each other over a list of openings, swapping colours every game, and reports the score with an Elo estimate and a confidence interval. Two identical engines will score exactly 0.500, so a real comparison needs the two sides to actually differ. Ten games cannot resolve anything under about 150 Elo; 100 games gets you to roughly 60.

Currently, it has beaten chess.com's 2560 elo chess bot with both black and white (It has also beaten the preceding chess bots)

Not done yet

tapering the network evaluation the way the hand tables are tapered
training on game results blended with the Stockfish evals

Project layout

main.py       entry point (run with: python3 main.py)
arena/        scripts to test engines (match.py)
core/         the rules of chess (constants, board, zobrist, attacks, movegen, rules, notation)
engine/       deciding which move to play (search, evaluate, psqt, ordering, player)
network/      the neural network (san_parser, data_parser, feature_target_prep, base_network)
ui/           tkinter GUI
assets/       piece icons
testing/      perft, final_tests, san_tests, verify_npz
docs/         SPEC.md, PLANS.md