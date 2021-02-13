import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
matplotlib.rcParams['figure.figsize'] = 16,12

import chess
import chess.svg
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
import imageio
import random
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

alphabet = ['black_queen', 'black_king', 'black_bishop', 'black_knight', 'black_rook', 'black_pawn',
                'white_queen', 'white_king', 'white_bishop', 'white_knight', 'white_rook', 'white_pawn']

def chess_piece_from_name(piece_name):
    name_to_piece = {
        'black_queen': chess.Piece(chess.QUEEN, chess.BLACK),
        'black_king': chess.Piece(chess.KING, chess.BLACK),
        'black_bishop': chess.Piece(chess.BISHOP, chess.BLACK),
        'black_knight': chess.Piece(chess.KNIGHT, chess.BLACK),
        'black_rook': chess.Piece(chess.ROOK, chess.BLACK),
        'black_pawn': chess.Piece(chess.PAWN, chess.BLACK),
        'white_queen': chess.Piece(chess.QUEEN, chess.WHITE),
        'white_king': chess.Piece(chess.KING, chess.WHITE),
        'white_bishop': chess.Piece(chess.BISHOP, chess.WHITE),
        'white_knight': chess.Piece(chess.KNIGHT, chess.WHITE),
        'white_rook': chess.Piece(chess.ROOK, chess.WHITE),
        'white_pawn': chess.Piece(chess.PAWN, chess.WHITE),
    }

    return name_to_piece[piece_name]

def chess_board_from_array(board_array):
    chess_board = chess.Board()
    chess_board.clear()
    for rowIndex in range(len(board_array)):
        for colIndex in range(len(board_array[rowIndex])):
            piece_name = board_array[rowIndex][colIndex]
            if piece_name is None:
                continue

            chess_piece = chess_piece_from_name(piece_name)
            chess_square = chess_square_from_coordinate(rowIndex, colIndex)
            # print(f"{piece_name} is a {chess_piece} and should be at {chess_square}")

            chess_board.set_piece_at(chess_square, chess_piece)
    return chess_board

def write_board_png(board, file_name):
    svg_text = chess.svg.board(board, size=1024)
    svg_file = open('in.svg', 'w')
    svg_file.write(svg_text)
    svg_file.close()

    svg = svg2rlg('in.svg')
    renderPM.drawToFile(svg, file_name)

    png = load_image(file_name)
    cropped_png = png[40:984, 40:984]
    save_image(file_name, cropped_png)

def write_board_png_pretty(board, file_name, last_move):
    svg_text = chess.svg.board(board=board, size=512, lastmove=last_move)
    svg_file = open('in.svg', 'w')
    svg_file.write(svg_text)
    svg_file.close()

    svg = svg2rlg('in.svg')
    renderPM.drawToFile(svg, file_name)


def chess_square_from_coordinate(rowIndex, colIndex):
    return 63 - (rowIndex * 8 + (7 - colIndex))

def transform_color_image(img):
    img = image_bin(image_gray(img))
    img = erode(img, (5, 5))
    #img = dilate(img, (5, 5))
    return img

def load_image(path):
    image = imageio.imread(path)
    return image

def save_image(path, image):
    imageio.imwrite(path, image)

def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret, image_bin = cv2.threshold(image_gs, 90, 255, cv2.THRESH_BINARY)
    return image_bin

def invert(image):
    return 255-image

def display_image(image, title, color=False):
    plt.title(title)
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')
    plt.show()

def dilate(image, kernel_block=(3,3)):
    kernel = np.ones(kernel_block) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)

def erode(image, kernel_block=(3,3)):
    kernel = np.ones(kernel_block) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)

def resize_region(region):
    return cv2.resize(region, (100, 100), interpolation=cv2.INTER_NEAREST)

def scale_to_range(image):
    return image/255

def matrix_to_vector(image):
    return image.flatten()

def prepare_for_ann(images):
    ready_for_ann = []
    for image in images:
        scale = scale_to_range(image)
        ready_for_ann.append(matrix_to_vector(scale))
    return ready_for_ann

def convert_output_old(outputs):
    nn_outputs = []
    for index in range(len(outputs)):
        output = np.zeros(len(outputs))
        output[index] = 1
        nn_outputs.append(output)
    return np.array(nn_outputs)

def convert_output(labels):
    nn_outputs = []
    for label in labels:
        output_array = np.zeros(len(alphabet))
        for i in range(len(alphabet)):
            if alphabet[i] == label:
                output_array[i] = 1
        nn_outputs.append(output_array)
    return nn_outputs

def create_ann(output_size):
    ann = Sequential()
    ann.add(Dense(128, input_dim=10000, activation='sigmoid'))
    ann.add(Dense(output_size, activation='sigmoid'))
    return ann


def train_ann(ann, X_train, y_train, epochs):
    X_train = np.array(X_train, np.float32)  # dati ulaz
    y_train = np.array(y_train, np.float32)  # zeljeni izlazi na date ulaze

    print("\nTraining started...")
    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)
    ann.fit(X_train, y_train, epochs=epochs, batch_size=1, verbose=1, shuffle=False)
    print("\nTraining completed...")
    return ann

def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]

def display_result(outputs, alphabet):
    result = []
    for output in outputs:
        result.append(f"{alphabet[winner(output)]}:")
    return result

def get_random_board(empty_cell_chance):
    board_array = [[ None for i in range(8) ] for i in range(8)]
    for i in range(len(board_array)):
        for j in range(len(board_array[i])):
            if random.random() < empty_cell_chance:
                board_array[i][j] = None
            else:
                random_index = random.randint(0, len(alphabet)-1)
                board_array[i][j] = alphabet[random_index]


    return chess_board_from_array(board_array)