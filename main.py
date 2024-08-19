from flask import Flask, request, send_file, jsonify
import io
import cv2
from PIL import Image

app = Flask(__name__)

@app.route('/upload-image', methods=['POST'])
def upload_image():
    # if 'image' not in request.files:
    #     return jsonify({'error': 'No image part'}), 400
    #
    # file = request.files['image']
    #
    # if file.filename == '':
    #     return jsonify({'error': 'No selected file'}), 400

    try:
        # # Save the uploaded file
        # filename = 'sudoku.png'
        # file.save(filename)

        filename = "D:\Github\ProcessSudokuImage\sudoku.png"

        # Read the image using OpenCV
        image = cv2.imread(filename)

        # Process the image (convert to grayscale in this example)
        grayscale_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        threshold_img = cv2.adaptiveThreshold(grayscale_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

        # Save the processed image to a temporary file
        processed_filename = 'sudoku_processed2.png'
        cv2.imwrite(processed_filename, threshold_img)

        # Read the processed image file into a byte stream
        with open(processed_filename, 'rb') as f:
            byte_io = io.BytesIO(f.read())

        byte_io.seek(0)

        return send_file(byte_io, mimetype='image/png')

    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True)
