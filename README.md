**CHESS BOT**

This program serves two purposes. One as a simple chess game you can play, and two a chess bot.

Current state of chess bot:
The engine is currently just an alpha-beta search engine. Its depth and time limit are knobs that can be tweaked in ui/gui.py
The move order which it is searched in is currently sorted by:
Saved moves from transposition table/best move from previous depths and then by MVV-LVA (Most Valuable Victim, Least Valuable Attacker)
Instead of a simple dfs recursive search, the search is done through iterative deepening: search one depth one layer at a time. (Starting at 1 all the way to max_depth capped by the time limit)
The engine also utilizes a quiescence search to follow capturing sequences and checking sequences

**Project layout**
```
main.py       entry point (run with: python3 main.py)
core/         the rules of chess (constants, board, zobrist, attacks, movegen, rules)
engine/       deciding which move to play (search, evaluate, psqt, ordering)
ui/           tkinter GUI
assets/       piece icons
testing/      perft and final_tests suites
docs/         SPEC.md, PLANS.md
```
