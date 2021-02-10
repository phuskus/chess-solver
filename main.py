import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import collections
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
matplotlib.rcParams['figure.figsize'] = 16,12

import chess
import chess.svg
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
import imageio

def main():
    image_color = load_image('images/standalone_chess_pieces.png')
    display_image(image_color, "Original image")

    img = image_bin(image_gray(image_color))
    display_image(img, "Binary image")

    img_bin = erode(dilate(img))
    display_image(img_bin, "Dilated and eroded")

    selected_regions, numbers = select_roi(image_color.copy(), img_bin)
    display_image(selected_regions, "Regions of interest")

    alphabet = ['pawn', 'king', 'knight', 'bishop', 'rook', 'queen']

    inputs = prepare_for_ann(numbers)
    outputs = convert_output(alphabet)
    ann = create_ann(output_size=6)
    ann = train_ann(ann, inputs, outputs, epochs=2000)

    test_color = load_image('./board.png')
    test = image_bin(image_gray(test_color))
    test_bin = erode(dilate(test))
    selected_test, test_numbers = select_roi(test_color.copy(), test_bin)
    display_image(selected_test, 'Test image')

    test_inputs = prepare_for_ann(test_numbers)
    result = ann.predict(np.array(test_inputs, np.float32))
    print(display_result(result, alphabet))

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
    ret, image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin

def invert(image):
    return 255-image

def display_image(image, title, color=False):
    plt.title = title
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')
    plt.show()

def dilate(image):
    kernel = np.ones((3, 3)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)

def erode(image):
    kernel = np.ones((3, 3)) # strukturni element 3x3 blok
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
        if area > 100 and h < 50 and h > 10 and w > 10:
            # kopirati [y:y+h+1, x:x+w+1] sa binarne slike i smestiti u novu sliku
            # oznaciti region pravougaonikom na originalnoj slici sa rectangle funkcijom
            region = image_bin[y:y + h + 1, x:x + w + 1]
            regions_array.append([resize_region(region), (x, y, w, h)])
            cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)

    regions_array = sorted(regions_array, key=lambda x: x[1][0])
    sorted_regions = [region[0] for region in regions_array]
    return image_orig, sorted_regions

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
        result.append(alphabet[winner(output)])
    return result


if __name__ == "__main__":
    main()