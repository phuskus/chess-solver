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


def main():
    """
    arg1 - input PNG
    arg2 - turn [white | black]
    arg3 - chess AI max think time
    """
    #sys.argv = ["", "chess_problems/mate_in_4_0.png", "white", 0.2]
    if len(sys.argv) < 4:
        print("3 arguments are required: input png path, turn [white | black] and chess AI think time expressed in seconds.")
        return

    png_path = sys.argv[1]
    turn = sys.argv[2].lower()
    think_time = sys.argv[3]

    if not png_path.lower().endswith(".png"):
        print("Invalid png path!")
        return

    if turn != "white" and turn != "black":
        print("Turn must be 'white' or 'black'")
        return

    chess_board = board_from_png(png_path)
    solve_chess_problem(chess_board, turn == "white", float(think_time))

def train_and_save():
    ann = train_model()
    ann.save("model1")

if __name__ == "__main__":
    main()