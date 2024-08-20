import os
import cv2
import numpy as np
import pytesseract
from main import cell_image_directory


def extractNumbersFromImage(img_grid):
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    tmp_sudoku = [[0 for i in range(9)] for j in range(9)]

    for i in range(9):
        for j in range(9):
            im = img_grid[i][j]

            im = cv2.normalize(im, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            res, im = cv2.threshold(im, 64, 255, cv2.THRESH_BINARY)

            # Fill everything that is the same colour (black) as top-left corner with white
            cv2.floodFill(im, None, (0, 0), 255)

            image = cv2.resize(im, (500, 500))

            # # Size of the image (assumed to be 30x30 based on your previous messages)
            # height, width = 500, 500
            #
            # # Define the width of the border
            # border_width = 20
            #
            # # Loop through the entire top and bottom border width
            # for x in range(width):
            #     for k in range(border_width):  # Covers 5 pixels depth
            #         cv2.floodFill(image, None, (x, k), 255)  # Top 5-pixel-wide border
            #         cv2.floodFill(image, None, (x, height - 1 - k), 255)  # Bottom 5-pixel-wide border
            #
            # # Loop through the entire left and right border width
            # for y in range(height):
            #     for k in range(border_width):  # Covers 5 pixels width
            #         cv2.floodFill(image, None, (k, y), 255)  # Left 5-pixel-wide border
            #         cv2.floodFill(image, None, (width - 1 - k, y), 255)  # Right 5-pixel-wide border

            cell_filename = os.path.join(cell_image_directory, f"cell_{i}_{j}.jpg")
            cv2.imwrite(cell_filename, image)

            # OCR
            data = pytesseract.image_to_string(image, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
            tmp_sudoku[i][j] = data.strip()

    return tmp_sudoku
