**SPEC**

This file is the set of promises the code makes to itself. The README says what the program does, this says what has to stay true while you change it. Most of the bugs worth writing down were not wrong logic, they were a rule below quietly broken by a change above.

**The board**

`chess_board` is a list of 8 rows of 8 ints. Row 0 is rank 8 and row 7 is rank 1, which is the same order FEN writes ranks in, so placement parsing is a straight walk with no flipping. Column 0 is the a-file.

A square holds one signed int. The magnitude is the piece type (pawn 1 through king 6) and the sign is the side, positive for white. Zero is empty. `side_of`, `is_enemy` and `is_friendly` are the only things that should be reading that sign directly.

**Moves**

A move is a pair: an `(row, col)` tuple for where it starts, and a `Move(row, col, promotion)` for where it lands. `promotion` is 0 for everything that is not a promotion, and a piece type otherwise. Four separate moves share a destination square when a pawn promotes, which is why the promotion field exists at all rather than defaulting to a queen.

Castling is stored as the king's two-square move, and en passant as the pawn's move to the empty square. Neither carries a flag. make_move works out what kind of move it is from the board, so a move is always just a from-square, a to-square and a promotion.

**Authoritative state and derived state**

This is the important one.

Authoritative state is the board, the side to move, the castling rights, the en passant square and the clocks. Everything else on the Chess object is a cache computed from those:

```
white_pieces, black_pieces      where the pieces are
white_king_pos, black_king_pos  where the kings are
zobrist                         the position hash
phase                           how much material is left
```

Anything that installs a new position has to rebuild all of them. `from_fen` does, `Chess.__init__` does, and any future loader or test fixture has to as well. The reason this gets its own section is that a stale cache never crashes. A wrong `phase` just blends the evaluation tables slightly wrong. A stale piece set makes move generation quietly skip a piece that is there, or invent one that is not. You find it days later as "the engine plays badly from loaded positions" and go looking in the search.

`move_order`, `position_counts` and `game` are history rather than position. No FEN can carry them, so a loaded position starts with them empty.

**make_move and undo_move**

`make_move` returns true if the move was legal and false if it left the mover's own king attacked, in which case it has already undone itself. That makes it usable as a filter, and it is the only thing in the codebase that decides legality:

```
if not make_move(chess, origin, move):
    continue
```

It raises rather than returning false for two things that are bugs and not illegal moves: a move from an empty square, and a promotion with no piece type. Both raise before touching the board, so catching them leaves the position intact.

`undo_move` restores everything `make_move` changed, and restores rather than recomputes where it can. The hash and the phase are saved on the way in and put back on the way out, because recomputing them backwards is another chance to get them wrong. After an undo the position must be indistinguishable from before the move, which is what `perft --state` checks at every node.

**Draws and results**

`game_result` returns None while the game is live and the reason it ended otherwise. It is meant to be called after a move, to judge the position the mover just left.

Repetition and the fifty move rule are set by `make_move` as they happen, since only it knows the history. Checkmate, stalemate and insufficient material are worked out from the position. Insufficient material is checked first because it is O(1) on the piece sets and a position with insufficient material can never be checkmate anyway.

Insufficient material is deliberately conservative: K v K, K and one minor v K, and K+B v K+B with both bishops on one colour. KBN v K is a forced win and KNN v K is only a draw with best play, so neither counts. Claiming a draw that is not one corrupts a result, missing one only costs a little time.

**Notation**

`from_fen` and `to_fen` are inverses, with one asymmetry worth knowing. This engine only stores an en passant square when a capture is actually available, because that is the view `full_hash` takes and two positions differing only by an unusable en passant square really are the same position here. So a FEN carrying an en passant square no pawn can use comes back out as `-`. Round trips are stable after the first one.

Moves are written in UCI: from-square, to-square, and a promotion letter if there is one. `e2e4`, `e7e8q`, and `e1g1` for white castling kingside. `move_name` and `parse_uci_move` are inverses and are tested as a pair, because a round trip on its own cannot catch both halves being wrong in the same way. That is exactly the bug that shipped once.

A start FEN plus a list of UCI moves is a complete, replayable record of a game. That is the form the arena stores results in, and it is what turns a loss into a regression test.

**The player contract**

Anything that can choose a move implements it:

```
select_move(chess) -> (origin, Move), or None
```

Two rules every player keeps. It is handed the live board and must leave it exactly as it found it, so anything that makes a move in order to look at it undoes it. And None means this position has no legal move, meaning the game is over. It never means the player ran out of time. A player that cannot decide still has to return something legal.

Players may also implement `new_game` to forget what they learned, `last_score` to report what they thought of the position they just moved in, and `last_nodes` for how much work it took. All three are optional and default to doing nothing useful.

**The search contract**

`search(chess, max_depth, node_limit, time_limit)` returns a move and a score. The move is None only when the position has no legal move at all, never because the budget was tight, which is why depth 1 always runs to completion. Callers read None as game over, so that guarantee is the whole reason the signal is trustworthy.

Scores are from the point of view of the side to move, and the same is true of `self_evaluate`. That convention has to hold everywhere or nothing composes: negamax depends on it, the arena's adjudication depends on it, and any future training data has to use it too. A sign error here is a bug that trains happily to a loss floor while looking fine.

The board is left exactly as it was found, and a search asserts its own hash is unchanged before returning.

Two engines never share anything. Each owns its transposition table and its own search budget, so two players in one match cannot spend each other's nodes or read each other's scores. Anything held at module scope in the search is a bug waiting to be a fairness problem.

There are two sets of piece values on purpose. `PIECE_VALUES` in constants is for move ordering, where only the relative order matters. `MATERIAL` in psqt is what the tables were tuned against, and is what the evaluation uses. Using one for the other job is wrong in a way nothing will complain about.

**The arena**

`play_one_game` is about chess, `play_match` is about experimental design, and `aggregate` is about statistics. Keeping those separate is the point of the file, because they change for entirely different reasons.

A player is never handed a position whose game is already over, so termination is checked before anyone is asked to think. A player that returns an illegal move, returns None with legal moves available, or raises, is recorded as an error and that one game is abandoned. It is never scored as a loss, because a bug that looks like a plausible result is a bug you will not find.

Games may be ended early three ways: both sides agreeing the position is level for long enough, one side reporting a hopeless score for long enough, or a hard ply limit. The ply limit is a safety net and should almost never fire. If it does, the other two are mistuned and the numbers from that match are not worth reading.

Outcomes are always stored from white's point of view, since that is the only perspective that does not change halfway through a colour-swapped match. `score_for` does the flipping.

**What the tests guarantee**

perft is the floor. If the move generation is wrong then every number above it is meaningless, so it runs against published counts and, with `--all`, checks the hash, the piece sets, undo and the FEN round trip at every node.

final_tests covers the layer above. The rule it follows is that every bug that gets found once gets a test, named after what actually broke rather than the feature it lived in. A few of them exist purely because the old version passed while being badly wrong, which is worth remembering before deleting a test that looks redundant.