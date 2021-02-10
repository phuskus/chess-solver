import chess
import chess.engine


engine = chess.engine.SimpleEngine.popen_uci("stockfish.exe")

board = chess.Board()
while not board.is_game_over():
    result = engine.play(board, chess.engine.Limit(time=0.1))
    if board.turn:
        turnColor = "White"
    else:
        turnColor = "Black"

    print(f"{turnColor}: {result.move}")
    board.push(result.move)

engine.quit()