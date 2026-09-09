**CHESS BOT**

This program serves two purposes. One as a simple chess game you can play, and two a chess bot.

**Description**
The engine is currently an alpha-beta search engine. It has a depth, time, and node limit option
The move order which it is searched in is currently sorted by:
Saved moves from transposition table/best move from previous depths and then by MVV-LVA (Most Valuable Victim, Least Valuable Attacker)
Moves lists are not checked for legality, instead if they are illegal, when make_move runs, it will return false and then undo the move
Instead of a simple dfs recursive search, the search is done through iterative deepening: search one depth one layer at a time. (Starting at 1 all the way to max_depth capped by the time limit)
The engine also utilizes a quiescence search to follow capturing sequences and checking sequences
The engine evaluation comes from piece values and psqt (positional scoring for each piece). Both piece and psqt values have different values for midgame and endgame. Those values are blended as the phase of the game goes from midgame to endgame

**TESTING**
Currently, it has beaten chess.com's 2300 elo chess bot with both black and white
(It has also beaten the preceding chess bots)

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
