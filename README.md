# ProcessSudokuImage (Image Processing + ML for Sudoku)

This project processes an input sudoku image, extracts the grid & digits, and transforms it into a JSON so the solver can use it. It may include digit recognition (ML) and image preprocessing steps.

## Features

- Preprocess image (grayscale, thresholding, contours)  
- Detect Sudoku grid boundaries  
- Segment cells & extract digit images  
- Recognize digits using a trained model (e.g. Keras / CNN)  
- Output a 2D array / JSON of initial puzzle  

## Tech Stack 

- Python  
- OpenCV (for image processing)  
- TensorFlow / Keras (for digit classification)   
- Jupyter notebook (for training on numbers)
