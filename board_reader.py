import keras
from utils import *
import math

def board_from_png(filePath):
    ann = keras.models.load_model('model1')

    test_color = load_image(filePath)
    test_transformed = transform_color_image(test_color)

    marked_image, region_tuples = select_roi(test_color.copy(), test_transformed)
    #display_image(marked_image, 'Marked test image')

    test_inputs = prepare_for_ann(region[0] for region in region_tuples)
    if len(test_inputs) == 0:
        result = []
    else:
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

    return chess_board_from_array(board_array)


def train_model():
    image_color = load_image('images/all_standalone_chess_pieces.png')
    img = transform_color_image(image_color)

    marked_image, region_tuples = select_roi(image_color.copy(), img)
    #display_image(marked_image, "Regions of interest")

    inputs = prepare_for_ann(region[0] for region in region_tuples)
    outputs = convert_output(alphabet)
    ann = create_ann(output_size=12)
    ann = train_ann(ann, inputs, outputs, epochs=2000)

    return ann


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