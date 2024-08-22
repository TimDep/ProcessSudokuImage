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
            image = img[4:img.shape[0] - 4, 4:img.shape[1] - 4]
            image = cv2.resize(image, (32, 32))
            image = image / 255
            image = image.reshape(1, 32, 32, 1)

            # Ensure the image is in the proper format for PyTesseract
            ocr_image = cv2.resize(im, (28, 28))  # Resize to the expected size
            ocr_image = cv2.bitwise_not(ocr_image)  # Invert image to match Tesseract expectations

            number = pytesseract.image_to_string(ocr_image, config='--psm 10 --oem 3 -c tessedit_char_whitelist=123456789')

            predictions = trained_model.predict(image)
            classIndex = np.argmax(predictions,axis=1)
            probabilityValue = np.amax(predictions)
            print(probabilityValue)
            if probabilityValue > 0.65:
                numberModel = str(classIndex[0])
                print(numberModel)
                print(number)
                if numberModel in number:
                    data = numberModel
                else:
                    if not number:
                        data=""
                    else:
                        data = number[0]
            elif probabilityValue < 0.25:
                data= ""
            else:
                data=number

            data = data.strip().replace("\n", "").replace("\r", "")

            tmp_sudoku[i][j] = data

    return tmp_sudoku
