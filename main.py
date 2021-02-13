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
    """
    sys.argv = ["", "board.png", "white"]
    if len(sys.argv) < 3:
        print("2 arguments are required: input png path and turn [white | black].")
        return

    png_path = sys.argv[1]
    turn = sys.argv[2].lower()

    if not png_path.lower().endswith(".png"):
        print("Invalid png path!")
        return

    if turn != "white" and turn != "black":
        print("Turn must be 'white' or 'black'")
        return

    chess_board = board_from_png(png_path)
    solve_chess_problem(chess_board, turn == "white")

def train_and_save():
    ann = train_model()
    ann.save("model1")

def debug_image_transform():
    image_color = load_image("images/all_standalone_chess_pieces.png")
    image_transformed = transform_color_image(image_color)
    display_image(image_transformed, "Transformed")

    marked_image, regions = select_roi(image_color.copy(), image_transformed)
    display_image(marked_image, "Regions")

if __name__ == "__main__":
    main()