import chess
import chess.engine

def solve_chess_problem(board):
    move_list = []

    engine = chess.engine.SimpleEngine.popen_uci("stockfish.exe")
    while not board.is_game_over():
        result = engine.play(board, chess.engine.Limit(time=0.1))
        if board.turn:
            turnColor = "White"
        else:
            turnColor = "Black"

        move_list.append(f"{turnColor}: {result.move}")
        board.push(result.move)
    engine.quit()

    return move_list