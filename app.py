from flask import *
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import pickle
import cv2

app = Flask(__name__)

def image_processing(img):
    model = load_model('./model/plant_disease_classification_model.h5')
    image_labels = pickle.load(open('./model/plant_disease_label_transform.pkl', 'rb'))
    image = cv2.imread(img)
    query = cv2.resize(image, (32, 32))
    data = []
    image_array = np.array(query)
    data.append(image_array)
    
    np_image = np.array(image_array, dtype=np.float16) / 225.0
    np_image = np.expand_dims(np_image,0)
    predict_x=model.predict(np_image) 
    classes_x=np.argmax(predict_x,axis=1)
    prediction_class = image_labels.classes_[classes_x][0]
    return prediction_class

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        file_path = secure_filename(f.filename)
        f.save(file_path)
        # Make prediction
        result = image_processing(file_path)
        result = "Predicted Leaf Condition: " +result
        os.remove(file_path)
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)