import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import collections
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
import keras
matplotlib.rcParams['figure.figsize'] = 16,12

import chess
import chess.svg
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
import imageio
from tabulate import tabulate
import math
import time

from chess_solver import solve_chess_problem

def main():
    print('Detecting board state from PNG...')
    chess_board = board_from_png('./board.png')
    print('Solving chess problem...')
    start_time = time.time()
    move_list = solve_chess_problem(chess_board)
    end_time = time.time()
    seconds = end_time - start_time
    print(f'Solved in {seconds:.2f} seconds! Move list:')
    for move in move_list:
        print(move)

alphabet = ['black_queen', 'black_king', 'black_bishop', 'black_knight', 'black_rook', 'black_pawn',
                'white_queen', 'white_king', 'white_bishop', 'white_knight', 'white_rook', 'white_pawn']

def board_from_png(filePath):
    ann = keras.models.load_model('model1')

    test_color = load_image('./board.png')
    test = image_bin(image_gray(test_color))
    test = erode(test, (3, 3))
    test = dilate(test, (3, 3))

    marked_image, region_tuples = select_roi(test_color.copy(), test)
    # display_image(marked_image, 'Marked test image')

    test_inputs = prepare_for_ann(region[0] for region in region_tuples)
    result = ann.predict(np.array(test_inputs, np.float32))

    detected_pieces = []
    for i in range(len(test_inputs)):
        detected_pieces.append((alphabet[winner(result[i])], region_tuples[i][1]))
    #print("DETECTED CHESS PIECES:")
    #for piece in detected_pieces:
        #print(f"{piece[0]}: [ {piece[1]} ]")

    board_array = [[None for i in range(8)] for j in range(8)]
    image_size = test_color.shape
    cell_size = (image_size[0] / 8, image_size[1] / 8)
    for piece in detected_pieces:
        bounding_rect = piece[1]
        center_x = (bounding_rect[0] + bounding_rect[0] + bounding_rect[2]) / 2.0
        center_y = (bounding_rect[1] + bounding_rect[1] + bounding_rect[3]) / 2.0
        center_point = (center_x, center_y)

        cell_x = math.floor(float(center_point[0]) / cell_size[0])
        cell_y = math.floor(float(center_point[1]) / cell_size[1])
        board_array[cell_y][cell_x] = piece[0]

    #print(tabulate(board_array, tablefmt='grid'))

    chess_board = chess.Board()
    chess_board.clear()
    for rowIndex in range(len(board_array)):
        for colIndex in range(len(board_array[rowIndex])):
            piece_name = board_array[rowIndex][colIndex]
            if piece_name is None:
                continue

            chess_piece = chess_piece_from_name(piece_name)
            chess_square = chess_square_from_coordinate(rowIndex, colIndex)
            #print(f"{piece_name} is a {chess_piece} and should be at {chess_square}")

            chess_board.set_piece_at(chess_square, chess_piece)
    return chess_board

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

def chess_square_from_coordinate(rowIndex, colIndex):
    return 63 - (rowIndex * 8 + (7 - colIndex))

def train_model():
    image_color = load_image('images/all_standalone_chess_pieces.png')
    # image_color = load_image('./board.png')
    # display_image(image_color, "Original image")

    img = image_bin(image_gray(image_color))
    # display_image(img, "Binary image")

    # img_bin = erode(dilate(img))

    # display_image(img_bin, "Dilated and eroded")

    img = erode(img, (3, 3))
    img = dilate(img, (3, 3))
    # display_image(img, "Transformed")

    marked_image, region_tuples = select_roi(image_color.copy(), img)
    # display_image(selected_regions, "Regions of interest")

    inputs = prepare_for_ann(region[0] for region in region_tuples)
    outputs = convert_output(alphabet)
    ann = create_ann(output_size=12)
    ann = train_ann(ann, inputs, outputs, epochs=2000)

    return ann

def numbers_example():
    image_color = load_image('images/brojevi.png')
    img = image_bin(image_gray(image_color))
    img_bin = erode(dilate(img))
    selected_regions, numbers = select_roi(image_color.copy(), img_bin)
    display_image(selected_regions)

    alphabet = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    inputs = prepare_for_ann(numbers)
    outputs = convert_output(alphabet)
    ann = create_ann(output_size=10)
    ann = train_ann(ann, inputs, outputs, epochs=2000)

    test_color = load_image('images/test.png')
    test = image_bin(image_gray(test_color))
    test_bin = erode(dilate(test))
    selected_test, test_numbers = select_roi(test_color.copy(), test_bin)
    display_image(selected_test)

    test_inputs = prepare_for_ann(test_numbers)
    result = ann.predict(np.array(test_inputs, np.float32))
    print(display_result(result, alphabet))

def generate_board_png(board, file_name):
    svg_text = chess.svg.board(board, size=256)
    svg_file = open('in.svg', 'w')
    svg_file.write(svg_text)
    svg_file.close()

    svg = svg2rlg('in.svg')
    renderPM.drawToFile(svg, file_name)

def load_image(path):
    image = imageio.imread(path, )
    return image

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
    return cv2.resize(region, (28, 28), interpolation=cv2.INTER_NEAREST)


def select_roi(image_orig, image_bin):
    contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = []  # lista sortiranih regiona po X osi
    regions_array = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contour)
        if w > 15 and h > 18 and w < 60 and h < 60:
            # kopirati [y:y+h+1, x:x+w+1] sa binarne slike i smestiti u novu sliku
            # oznaciti region pravougaonikom na originalnoj slici sa rectangle funkcijom
            region = image_bin[y:y + h + 1, x:x + w + 1]
            regions_array.append([resize_region(region), (x, y, w, h)])
            cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)

    regions_array = sorted(regions_array, key=lambda x: x[1][0])
    #sorted_regions = [region[0] for region in regions_array]
    return image_orig, regions_array

def scale_to_range(image):
    return image/255

def matrix_to_vector(image):
    return image.flatten()

def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        scale = scale_to_range(region)
        ready_for_ann.append(matrix_to_vector(scale))
    return ready_for_ann

def convert_output(alphabet):
    nn_outputs = []
    for index in range(len(alphabet)):
        output = np.zeros(len(alphabet))
        output[index] = 1
        nn_outputs.append(output)
    return np.array(nn_outputs)

def create_ann(output_size):
    ann = Sequential()
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dense(output_size, activation='sigmoid'))
    return ann


def train_ann(ann, X_train, y_train, epochs):
    X_train = np.array(X_train, np.float32)  # dati ulaz
    y_train = np.array(y_train, np.float32)  # zeljeni izlazi na date ulaze

    print("\nTraining started...")
    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)
    ann.fit(X_train, y_train, epochs=epochs, batch_size=1, verbose=0, shuffle=False)
    print("\nTraining completed...")
    return ann

def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]

def display_result(outputs, alphabet):
    result = []
    for output in outputs:
        result.append(f"{alphabet[winner(output)]}:")
    return result


if __name__ == "__main__":
    main()