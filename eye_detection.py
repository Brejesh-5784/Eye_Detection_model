import cv2

def detect_eyes(image_path):
    # Load the pre-trained Haar cascade classifier for eye detection
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    if eye_cascade.empty():
        print("Error: Could not load Haar cascade for eye detection.")
        return

    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not load the image. Check the file path.")
        return

    # Convert the image to grayscale (required for Haar cascades)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect eyes in the image
    eyes = eye_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

    # Draw red circles around the detected eyes
    for (x, y, w, h) in eyes:
        center_x = x + w // 2
        center_y = y + h // 2
        radius = w // 2
        cv2.circle(image, (center_x, center_y), radius, (0, 0, 255), 2)

    # Display the image with detected eyes
    cv2.imshow('Eye Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optionally save the output image
    output_path = "output_with_eyes_detected.jpg"
    cv2.imwrite(output_path, image)
    print(f"Output image saved at {output_path}")

# Directly set the image path
image_file = "PATH_FILE"
detect_eyes(image_file)
