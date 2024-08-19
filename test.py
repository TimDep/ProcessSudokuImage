import operator

import numpy as np
from flask import Flask, request, send_file, jsonify
import io
import cv2
from PIL import Image
import imutils


def upload_image():
    filename = "D:\Github\ProcessSudokuImage\sudoku.png"
    filenameBerg = "D:\Github\ProcessSudokuImage\sudoku2-1.jpg"

    # Read the image using OpenCV
    image = cv2.imread(filenameBerg)

    # Process the image (convert to grayscale in this example)
    grayscale_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    newimg, location = findSudokuBoard(grayscale_img, image)

    # Save the processed image to a temporary file
    processed_filename = 'sudoku_processed.png'
    cv2.imwrite(processed_filename, newimg)

def findSudokuBoard(gray, image):
    bfilter = cv2.bilateralFilter(gray, 5, 150, 5)
    edged = cv2.Canny(bfilter, 100, 200)
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE,
                                 cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    newimg = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 3)
    # Save the processed image to a temporary file
    processed_filename = 'contours.png'
    cv2.imwrite(processed_filename, newimg)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
    location = None
    # Finds rectangular contour
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 15, True)
        if len(approx) == 4:
            location = approx
            break
    result = get_perspective(image, location)
    return result, location


def get_perspective(img, location, height=900, width=900):
    """Takes an image and location of an interesting region.
    And return the only selected region with a perspective transformation"""
    pts1 = np.float32([location[0], location[3], location[1], location[2]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, matrix, (width+10, height+10))
    # result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)
    return result

if __name__ == '__main__':
    upload_image()
