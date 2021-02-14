import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib
matplotlib.rcParams['figure.figsize'] = 16,12

import chess
import chess.svg
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

from board_reader import *

def run_tests():
    file_names = os.listdir("test_data")
    print("[ RUNNING TESTS ]")
    for file_name in file_names:
        if not file_name.endswith(".png"):
            continue
        board_name = file_name.split(".")[0]
        if not f"{board_name}.fen" in file_names:
            print(f"Fen file for {board_name} not found, skipping board test case")
            continue

        print(f"Running test case {file_name}... "),
        board = board_from_png(f"test_data/{file_name}")
        test_case_board = chess.Board()
        with open(f"test_data/{board_name}.fen") as file:
            fen_string = file.read()
        test_case_board.set_fen(fen_string)

        if board == test_case_board:
            print("PASS!")
        else:
            print("------------FAIL!")

def write_piece_from_board_array(board_array, file_path):
    board = chess_board_from_array(board_array)
    write_board_png(board, file_path)

    image_color = load_image(file_path)
    piece_region = get_first_region(image_color)
    save_image(file_path, piece_region)

def get_first_region(image_color):
    transformed = transform_color_image(image_color)
    marked_image, region_tuples = select_roi(image_color.copy(), transformed)
    return region_tuples[0][0]

def generate_training_data():
    if not os.path.isdir("training_data"):
        os.mkdir("training_data")

    for piece_name in alphabet:
        print(f"Generating training data for {piece_name}...")
        for i in range(8):
            for j in range(8):
                board_array = [[None for i in range(8)] for i in range(8)]
                board_array[i][j] = piece_name
                write_piece_from_board_array(board_array, f"training_data/{piece_name}_{i}_{j}.png")
    print("Done generating training data!")

def run_tests_sanity():
    file_names = os.listdir("test_sanity")
    print("[ RUNNING SANITY TESTS ]", end="")
    current_piece_name = None
    test_count = 0
    tests_passed = 0
    tests_failed = 0
    piece_test_count = 0
    piece_tests_passed = 0
    piece_tests_failed = 0
    for file_name in file_names:
        if not file_name.endswith(".png"):
            continue
        board_name = file_name.split(".")[0]
        tokens = board_name.split("_")
        piece_name = tokens[0] + "_" + tokens[1]
        if piece_name != current_piece_name:
            if piece_test_count > 0:
                print(f"\nTests ran: {piece_test_count}")
                print(f"Tests passed: {piece_tests_passed} ({(piece_tests_passed / piece_test_count) * 100.0:.2f}%)")
            print("")
            print(f"Running sanity tests for {piece_name}", end="")
            current_piece_name = piece_name

            piece_test_count = 0
            piece_tests_passed = 0
            piece_tests_failed = 0

        if not f"{board_name}.fen" in file_names:
            print(f"Fen file for {board_name} not found, skipping board test case")
            continue

        board = board_from_png(f"test_sanity/{file_name}")
        test_case_board = chess.Board()
        with open(f"test_sanity/{board_name}.fen") as file:
            fen_string = file.read()
        test_case_board.set_fen(fen_string)

        if board == test_case_board:
            print(".", end="")
            tests_passed += 1
            piece_tests_passed += 1
        else:
            print("x", end="")
            tests_failed += 1
            piece_tests_failed += 1
        test_count += 1
        piece_test_count += 1
    print("\n---------------")
    print(f"Total tests ran: {test_count}")
    print(f"Total tests passed: {tests_passed} ({(tests_passed / test_count) * 100.0:.2f}%)")

def generate_test_data(count):
    if not os.path.isdir("test_data"):
        os.mkdir("test_data")

    empty_cell_chance = 1.0
    decrement = empty_cell_chance / count
    print("[ GENERATING TEST DATA ]")
    for i in range(count):
        board = get_random_board(empty_cell_chance)
        file_name = f"test_data/board_{i}.png"
        fen_file_name = f"test_data/board_{i}.fen"

        fen_file = open(fen_file_name, "w")
        fen_file.write(board.fen())
        fen_file.close()

        write_board_png(board, file_name)
        print(f"Generated test board {file_name}, {fen_file_name}")

        empty_cell_chance -= decrement
    print("Done generating test data!")