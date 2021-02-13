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

def main():
    pass

def train_and_save():
    ann = train_model()
    ann.save("model1")

def debug_image_transform():
    image_color = load_image("images/all_standalone_chess_pieces.png")
    image_transformed = transform_color_image(image_color)
    display_image(image_transformed, "Transformed")

    marked_image, regions = select_roi(image_color.copy(), image_transformed)
    display_image(marked_image, "Regions")

def solve_board(board, white=True):
    start_time = time.time()
    move_list = solve_chess_problem(board)
    end_time = time.time()
    seconds = end_time - start_time
    print(f'Solved in {seconds:.2f} seconds! Move list:')
    for move in move_list:
        print(move)

if __name__ == "__main__":
    main()