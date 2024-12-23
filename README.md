# Eye_Detection_model
This Python script detects eyes in an image using OpenCV’s Haar cascade. It highlights eyes with red circles, displays the result, and saves it as `output_with_eyes_detected.jpg`. Replace `"PATH_FILE"` with your image path.
This Python script is designed to detect eyes in an image using OpenCV and highlight them with red circles. It utilizes OpenCV’s pre-trained Haar cascade classifier (`haarcascade_eye.xml`), which is specifically designed for eye detection. The script processes the image, detects eye-like features, and saves the result with markings.

**Key Features**:
1. **Eye Detection**: The Haar cascade model detects eyes based on patterns in the image.
2. **Image Conversion**: The input image is converted to grayscale for more efficient and accurate processing.
3. **Marking Eyes**: Detected eyes are highlighted with red circles on the original image.
4. **Display and Save**: The processed image is displayed in a window and saved as `output_with_eyes_detected.jpg` for future use.

**Usage Instructions**:
1. Replace `"PATH_FILE"` in the script with the full path to your image (e.g., `"/Users/brejesh/Downloads/DSCN2252.JPG"`).
2. Save the script and run it in a Python environment:
   ```bash
   python eye_detection.py
   ```
3. The script will process the image, detect eyes, display the output, and save the image in the current directory.

**Requirements**:
- Install OpenCV by running:
  ```bash
  pip install opencv-python
  ```
This program is simple yet effective for detecting eyes in images with minimal setup.
