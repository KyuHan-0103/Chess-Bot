**CHESS BOT**

This program serves two purposes. One as a simple chess game you can play, and two a chess bot.

**Description**

The engine is an alpha-beta search with a depth, node, and time budget. Any combination can be set and whichever runs out first stops the search.

Rather than one recursive descent to a fixed depth, the search uses iterative deepening: depth 1, then depth 2, and so on up to max_depth, keeping the best move from the last iteration that finished. Depth 1 always runs to completion, so the search always comes back with a legal move no matter how small the budget.

Moves are searched in this order:

- the best move from the transposition table, or the best move from the previous iteration
- captures, sorted by MVV-LVA (Most Valuable Victim, Least Valuable Attacker)
- promotions, which get a bonus on top of the above

Move lists are not filtered for legality up front. A move is played, and if it leaves your own king attacked then make_move undoes it and returns false. That is cheaper than working out pins ahead of time, and it means only one place in the codebase decides what is legal.

When the side to move is in check the search spends an extra ply there, so it does not cut off in the middle of a forcing sequence. At the leaves a quiescence search follows captures and promotions until the position is quiet, which keeps the evaluation from being read halfway through a trade. Check evasions inside quiescence are not implemented yet.

The evaluation is material plus piece-square tables, and nothing else. Both have separate midgame and endgame values which are blended by how much material is left on the board, so a king that wants to hide behind pawns in the opening walks toward the centre in an endgame without any special case code for it.

**Requirements**

Python 3 (developed on 3.12). The GUI needs tkinter and Pillow. The engine and the test suites need neither.

**Running**

Everything runs from the repo root, since the packages are imported by name.

```
python3 main.py                    play against the engine
python3 -m testing.perft --all     move generation and board invariants
python3 -m testing.final_tests     engine, notation and arena regression suite
python3 -m arena.match --random    engine vs a random mover, as a smoke test
python3 -m arena.match             engine vs engine
```

In the GUI, click or drag a piece to move it. Press space and the engine plays a move for whoever is to move.

**TESTING**

perft counts the leaves of the legal move tree and compares against published values, which is what catches move generation bugs. Passing `--all` also checks that the zobrist hash, the piece sets, undo_move and the FEN round trip all stay consistent at every node of the tree.

final_tests covers everything above move generation: the search, the transposition table, the search budget, endgame detection, FEN and UCI notation, and the match harness.

arena/match.py plays two players against each other over a list of openings, swapping colours every game, and reports the score with an Elo estimate and a confidence interval. Two identical engines will score exactly 0.500, so a real comparison needs the two sides to actually differ.

Currently, it has beaten chess.com's 2300 elo chess bot with both black and white
(It has also beaten the preceding chess bots)

**Not done yet**

- check evasions inside the quiescence search
- killer move and history heuristics in the move ordering (the hooks are sitting in ordering.py)
- null move pruning and late move reductions
- SPRT, so a match can stop as soon as it has an answer instead of running a fixed number of games
- a UCI protocol adapter, which would let the bot run under cutechess or play on lichess
- tuning the evaluation against real game results instead of hand-picked numbers

**Project layout**

```
main.py       entry point (run with: python3 main.py)
arena/        scripts to test engines (match.py)
core/         the rules of chess (constants, board, zobrist, attacks, movegen, rules, notation)
engine/       deciding which move to play (search, evaluate, psqt, ordering)
ui/           tkinter GUI
assets/       piece icons
testing/      perft and final_tests suites
docs/         SPEC.md, PLANS.md
```