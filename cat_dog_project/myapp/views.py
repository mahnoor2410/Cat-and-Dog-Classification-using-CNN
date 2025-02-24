from django.shortcuts import render
import base64
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import io

# ========================== Load the trained model ===========================

model_path = os.path.join(os.path.dirname(__file__), 'models', 'cat_dog_model.keras')
model = tf.keras.models.load_model(model_path)

# ========================== Image Conversion to Array ===========================

# Function to convert the uploaded image into a format the model can understand
def image_to_array(image_bytes: bytes, target_size=(180, 180)) -> np.ndarray: # img ko bytes(raw form) me lein , size set , return np array jo model k lea input data hoga
    image = Image.open(io.BytesIO(image_bytes)) # PIL img obj me convert kia
    image = image.resize(target_size)
    image_array = np.array(image)
    
    if len(image_array.shape) == 2:  # If the image is grayscale, convert it to RGB (3 channels)
        image_array = np.stack([image_array] * 3, axis=-1)
    
    if image_array.shape[-1] != 3:  # Ensure the image has 3 color channels (RGB)
        raise ValueError("Image must have 3 color channels (RGB).")
    
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def index(request):
    file_name, image, error, message = None, None, None, None
    probability_cat, probability_dog = None, None

    # Handle the image upload when the form is submitted (POST request)
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        file_name = uploaded_file.name  
        image_bytes = uploaded_file.read()  

        try:
            # Convert the image to a numpy array that the model can process
            img_array = image_to_array(image_bytes)
            predictions = model.predict(img_array)
            score = float(tf.keras.activations.sigmoid(predictions[0][0]))

            # Define a confidence threshold to ensure the prediction is either a cat or dog
            confidence_threshold = 0.7
            if score < confidence_threshold and score > 1 - confidence_threshold:
                message = "This image doesn't appear to be a cat or dog."
            else:
                if score > 0.5:
                    message = "Prediction: Dog"
                    probability_dog = round(100 * score, 2)  
                    probability_cat = round(100 * (1 - score), 2)  
                else:
                    message = "Prediction: Cat"
                    probability_cat = round(100 * (1 - score), 2) 
                    probability_dog = round(100 * score, 2)  
        except Exception as e:
            error = f"Error processing image: {str(e)}"

        # Convert the image to base64 to display it in the HTML page
        image_data = base64.b64encode(image_bytes).decode('utf-8')
        image = f"data:image/jpeg;base64,{image_data}"

    return render(request, 'index.html', {
        'file_name': file_name,  
        'image': image,  
        'error': error,  
        'message': message,  
        'probability_cat': probability_cat, 
        'probability_dog': probability_dog,  
    })
