import os

import imutils
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from skimage.segmentation import clear_border

import ml_digits
import cv2
from keras.src.saving import load_model
from PIL import Image


app = Flask(__name__)

#globals
global cell_image_directory
global trained_model

trained_model = load_model('digits_model.keras')
cell_image_directory = "BoardCells"
if not os.path.exists(cell_image_directory):
    os.makedirs(cell_image_directory)

@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Save the uploaded file
        filename = 'sudoku.png'
        file.save(filename)
        image = cv2.imread(filename)

        # Process the image
        grayscale_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_img = cv2.GaussianBlur(grayscale_img.copy(), (5, 5), 0)  # ksize must be both positive and odd
        threshold_img = cv2.adaptiveThreshold(blurred_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11,
                                              2)

        inverted_threshold_img = cv2.bitwise_not(threshold_img)

        corners = findCornersRect(inverted_threshold_img)
        newimg = cropAndWarp(image, corners)

        processed_image_filename = 'processed_sudoku.png'
        cv2.imwrite(processed_image_filename, newimg)

        grid = extractCells(newimg)

        grid = extractDigits(grid)

        finalgrid= ml_digits.extractNumbersFromImage(grid)

        return jsonify({'message': 'Image processed successfully', 'grid': pd.Series(finalgrid).to_json(orient='values')}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def findCornersRect(img):
    contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)  # approx none means only save the 4 points of the corners
    # Contours is a Python list of all the contours in the image. Each individual contour is a Numpy array of (x,y) coordinates of boundary points of the object.
    contours = sorted(contours, key=cv2.contourArea, reverse=True)  # Sort by area, descending
    rect = contours[0]  # Largest image with all points on the contours

    # search for outer points
    bottom_right = max(range(len(rect)), key=lambda i: rect[i][0][0] + rect[i][0][1])
    bottom_left = min(range(len(rect)), key=lambda i: rect[i][0][0] - rect[i][0][1])
    top_right = max(range(len(rect)), key=lambda i: rect[i][0][0] - rect[i][0][1])
    top_left = min(range(len(rect)), key=lambda i: rect[i][0][0] + rect[i][0][1])

    return [rect[top_left][0], rect[top_right][0], rect[bottom_right][0], rect[bottom_left][0]]


def cropAndWarp(img, corners_rect):
    standard = np.array([corners_rect[0], corners_rect[1], corners_rect[2], corners_rect[3]], dtype='float32')

    # Get the longest side in the rectangle
    longest_side = max([
        distance_between(corners_rect[2], corners_rect[1]),  # bottom_right / top_right
        distance_between(corners_rect[0], corners_rect[3]),  # top_left / bottom_left
        distance_between(corners_rect[2], corners_rect[3]),  # bottom_right / bottom_left
        distance_between(corners_rect[0], corners_rect[1])  # top_left / top_right
    ])

    warped = np.array([[0, 0], [longest_side - 1, 0], [longest_side - 1, longest_side - 1], [0, longest_side - 1]],
                      dtype='float32')
    warp = cv2.getPerspectiveTransform(standard, warped)
    return cv2.warpPerspective(img, warp, (int(longest_side), int(longest_side)))


def distance_between(p1, p2):
    a = p2[0] - p1[0]
    b = p2[1] - p1[1]
    return np.sqrt((a ** 2) + (b ** 2))

def extractCells(img):
    grid = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grid = cv2.adaptiveThreshold(grid, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 1)

    edge_h = np.shape(grid)[0]
    edge_w = np.shape(grid)[1]
    celledge_h = edge_h // 9
    celledge_w = np.shape(grid)[1] // 9

    tempgrid = []
    for i in range(celledge_h, edge_h + 1, celledge_h):
        for j in range(celledge_w, edge_w + 1, celledge_w):
            rows = grid[i - celledge_h:i]
            tempgrid.append([rows[k][j - celledge_w:j] for k in range(len(rows))])

    finalgrid = []
    for i in range(9):
        row = []
        for j in range(9):
            cell = grid[i * celledge_h:(i + 1) * celledge_h, j * celledge_w:(j + 1) * celledge_w]
            row.append(cell)
        finalgrid.append(row)
    return finalgrid

def extractDigits(grid):
    tmp_sudoku = [[cell for cell in row] for row in grid]  # Start with the original grid

    for i in range(9):
        for j in range(9):
            cell = grid[i][j]
            thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            thresh = clear_border(thresh)

            # Find contours in the thresholded cell
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            if not cnts or len(cnts) == 0:  # Check if contours list is empty
                # Set cell to fully white if no contours found
                tmp_sudoku[i][j] = np.full_like(cell, 255)
                continue

            # Proceed with the largest contour
            c = max(cnts, key=cv2.contourArea)
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            (h, w) = thresh.shape
            percentFilled = cv2.countNonZero(mask) / float(w * h)

            # Skip cells with very small contours considered as noise
            if percentFilled < 0.03:
                tmp_sudoku[i][j] = np.full_like(cell, 255)
                continue

            # Apply the mask to the thresholded cell to isolate the digit
            digit = cv2.bitwise_and(thresh, thresh, mask=mask)
            digit = cv2.bitwise_not(digit)
            tmp_sudoku[i][j] = digit  # Store the digit image in the corresponding grid position

    return tmp_sudoku





if __name__ == '__main__':
    app.run(debug=True)
