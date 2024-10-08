import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import Model
from pdf2image import convert_from_path
from PIL import Image

# Load the VGG16 model with pre-trained weights, excluding the top fully connected layers
base_model = VGG16(weights='imagenet', include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)

def convert_specific_page_to_image(pdf_path, page_number):
    images = convert_from_path(pdf_path, first_page=page_number, last_page=page_number)
    return images[0]

def image_to_array(image):
    return np.array(image.convert('RGB'))  # Ensure 3 channels for RGB input

def extract_vgg16_features(image_array):
    # Resize image to 224x224 as required by VGG16
    img = keras_image.array_to_img(image_array)
    img = img.resize((224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Extract features
    features = model.predict(img_array)
    return features.flatten()

def compare_images_vgg16(image1, image2):
    features1 = extract_vgg16_features(image1)
    features2 = extract_vgg16_features(image2)
    
    # Compute the Euclidean distance between the feature vectors
    distance = np.linalg.norm(features1 - features2)
    return distance

def get_director(pdf_path, page_number1, page_number2):
    image1 = convert_specific_page_to_image(pdf_path, page_number1)
    image2 = convert_specific_page_to_image(pdf_path, page_number2)

    # Convert images to numpy arrays
    img_array1 = image_to_array(image1)
    img_array2 = image_to_array(image2)
    
    distance = compare_images_vgg16(img_array1, img_array2)
    print(f"Distance between page {page_number1} and page {page_number2} of the PDF: {distance}")
    return distance

# Example usage
#path = r"E:\account_opening\code_base_001\zip\account_opening\code\output_pdf_files"
#page_number1 = 3  # First page number to compare (1-based index)
#page_number2 = 4  # Second page number to compare (1-based index)

#get_director(pdf_path, page_number1, page_number2)





