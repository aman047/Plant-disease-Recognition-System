import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained MobileNetV2 model
model_path = "trained_mobilenet_model.keras"

try:
    model = load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Load and preprocess the test image
image_path = "Plant_Disease_Dataset/test/test/PotatoHealthy1.JPG"

try:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))  # Adjusted for MobileNetV2
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Display the test image
    plt.imshow(image)
    plt.axis("off")
    plt.title("Test Image")
    plt.show()

except Exception as e:
    print(f"Error loading image: {e}")
    exit()

# Perform prediction
predictions = model.predict(image_array)
predicted_class = np.argmax(predictions, axis=1)[0]

# Ensure class labels match `train_generator.class_indices`
class_labels = [
    "Apple Scab", "Apple Black Rot", "Apple Cedar Rust", "Apple Healthy",
    "Blueberry Healthy",
    "Cherry Powdery Mildew", "Cherry Healthy",
    "Corn Cercospora Leaf Spot", "Corn Common Rust", "Corn Northern Leaf Blight", "Corn Healthy",
    "Grape Black Rot", "Grape Esca", "Grape Leaf Blight", "Grape Healthy",
    "Orange Haunglongbing",
    "Peach Bacterial Spot", "Peach Healthy",
    "Pepper Bacterial Spot", "Pepper Healthy",
    "Potato Early Blight", "Potato Late Blight", "Potato Healthy",
    "Raspberry Healthy",
    "Soybean Healthy",
    "Squash Powdery Mildew",
    "Strawberry Leaf Scorch", "Strawberry Healthy",
    "Tomato Bacterial Spot", "Tomato Early Blight", "Tomato Late Blight", "Tomato Leaf Mold",
    "Tomato Septoria Leaf Spot", "Tomato Spider Mites", "Tomato Target Spot",
    "Tomato Yellow Leaf Curl Virus", "Tomato Mosaic Virus", "Tomato Healthy"
]

# Display the result
if 0 <= predicted_class < len(class_labels):
    print(f"Predicted Disease: {class_labels[predicted_class]}")
else:
    print("Error: Predicted class index is out of range.")
