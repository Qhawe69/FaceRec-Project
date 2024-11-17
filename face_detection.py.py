import cv2
import os

# Path to the folder containing your images
image_folder = r"C:\Users\mhlon\Downloads\Images\Images"

# Load the Haar Cascade classifier for face detection
face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Helper function for face detection
def detect_faces(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Display the result
    cv2.imshow("Face Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Iterate over each image in the folder and perform face detection
for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        print(f"Processing image: {image_name}")
        detect_faces(image_path)
