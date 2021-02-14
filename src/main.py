import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib
matplotlib.rcParams['figure.figsize'] = 16,12

import time
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

from chess_solver import solve_chess_problem
from board_reader import *
from testing import *
import sys
import PySimpleGUI as sg

import chess.engine
def main():
    """
    arg1 - input PNG
    arg2 - turn [white | black]
    optional arg3 - chess AI max think time
    optional arg4 - opponent AI skill level [0 - 20]
    """
    if len(sys.argv) < 3:
        print("2 arguments are required: input png path and turn [white | black]. Optional: chess AI think time expressed in seconds, oppponent skill level [0 - 20]")
        return

    png_path = sys.argv[1]
    turn = sys.argv[2].lower()

    if len(sys.argv) < 4:
        think_time = 1.0
    else:
        try:
            think_time = float(sys.argv[3])
            if think_time <= 0:
                raise ValueError()
        except:
            print("Think time must be a positive number")
            return

    if len(sys.argv) < 5:
        opponent_skill = 20.0
    else:
        try:
            opponent_skill = float(sys.argv[4])
            if opponent_skill < 0 or opponent_skill > 20:
                raise ValueError
        except:
            print("Opponent skill must be a number between 0 and 20")
            return

    if not png_path.lower().endswith(".png"):
        print("Invalid png path!")
        return

    if turn != "white" and turn != "black":
        print("Turn must be 'white' or 'black'")
        return

    print("Reading board state from image...")
    chess_board = board_from_png(png_path)
    print("Done! Opening GUI...")
    solve_chess_problem(chess_board, turn == "white", think_time, opponent_skill)

def train_and_save():
    ann = train_model()
    ann.save("model1")

if __name__ == "__main__":
    main()