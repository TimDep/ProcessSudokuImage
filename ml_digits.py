import os
import cv2
import numpy as np
import pytesseract
from main import cell_image_directory, trained_model

def extractNumbersFromImage(img_grid):
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    tmp_sudoku = [[0 for i in range(9)] for j in range(9)]

    for i in range(9):
        for j in range(9):
            im = img_grid[i][j]

            im = cv2.normalize(im, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            res, im = cv2.threshold(im, 64, 255, cv2.THRESH_BINARY)

            image = cv2.resize(im, (32, 32))
            image = image[2:30,2:30]

            cell_filename = os.path.join(cell_image_directory, f"cell_{i}_{j}.jpg")
            cv2.imwrite(cell_filename, image)

            # digits model trained.
            img = np.asarray(image)
            img = img[4:img.shape[0] - 4, 4:img.shape[1] - 4]
            img = cv2.resize(img, (32, 32))
            img = img / 255
            image = img.reshape(1, 32, 32, 1)

            predictions = trained_model.predict(image)
            classIndex = np.argmax(predictions,axis=1)
            probabilityValue = np.amax(predictions)
            print(probabilityValue)

            if probabilityValue>0.65:
                data = classIndex[0]
            else:
                data = ""

            # OCR PYTESSERACT
            # data = pytesseract.image_to_string(image, config='--psm 10 --oem 3 -c tessedit_char_whitelist=123456789')

            #OCR KERAS // NOT WORKING
            # pipeline = keras_ocr.pipeline.Pipeline()
            # read_image = keras_ocr.tools.read(image)
            # data =pipeline.recognize([read_image])
            # print(data)

            tmp_sudoku[i][j] = data

    return tmp_sudoku
