import chess
import chess.engine
import PySimpleGUI as sg
from utils import *

def solve_chess_problem(board, turn):
    board.turn = turn
    engine = chess.engine.SimpleEngine.popen_uci("stockfish.exe")
    write_board_png_pretty(board, "temp_board.png", None)

    # Initialize GUI
    window = sg.Window("Chess Genius", location=(0, 0), layout=[
        [sg.Image(key="-IMAGE-")],
        [sg.Button("Next move", key="-NEXT_MOVE-"), sg.Text(key="-TURN_LABEL-", size=(40, 1)), sg.Text(key="-MOVE_LABEL-", size=(40, 1))]
    ], finalize=True)

    window["-IMAGE-"].update(filename="temp_board.png")
    if board.turn:
        window["-TURN_LABEL-"].update("Turn: White")
    else:
        window["-TURN_LABEL-"].update("Turn: Black")
    moveCount = 0
    window["-MOVE_LABEL-"].update(f"Move: {moveCount}")

    # Create an event loop
    while True:
        event, values = window.read()
        # End program if user closes window or
        # presses the OK button
        if event == sg.WIN_CLOSED:
            break

        if event == "-NEXT_MOVE-":
            result = engine.play(board, chess.engine.Limit(time=0.1))
            board.push(result.move)
            moveCount += 1
            if board.turn:
                window["-TURN_LABEL-"].update("Turn: White")
            else:
                window["-TURN_LABEL-"].update("Turn: Black")

            write_board_png_pretty(board, "temp_board.png", result.move)
            window["-IMAGE-"].update(filename="temp_board.png")
            window["-MOVE_LABEL-"].update(f"Move: {moveCount}")

            if board.is_checkmate():
                window["-NEXT_MOVE-"].update(disabled=True)
                window["-TURN_LABEL-"].update("Checkmate!")


    window.close()
    engine.quit()