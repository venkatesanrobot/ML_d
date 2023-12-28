from flask import Flask ,render_template,request
import os
from werkzeug.utils import secure_filename

import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import io

import keras.backend as K # F1 score metric custom object


UPLOAD_FOLDER = r"E:\ML_project\1st\ui\Images"
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def predict(image):  # to predict raw image input
    interpreter = tf.lite.Interpreter('ENet_model.tflite')
    interpreter.allocate_tensors()
    # get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img=cv2.imread(image)
    rdg=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(rdg, dsize=(160, 160))
  
    img = tf.keras.preprocessing.image.img_to_array(img)

    # Preprocess the image to required size and cast
    # input_shape = input_details[0]['shape']
    input_tensor = np.array(np.expand_dims(img, 0), dtype=np.float32)
    input_tensor = tf.keras.applications.efficientnet_v2.preprocess_input(input_tensor)
    # set the tensor to point to the input data to be inferred

    # Invoke the model on the input data
    interpreter.set_tensor(input_details[0]['index'], input_tensor)

    # Run the inference
    interpreter.invoke()
    output_details = interpreter.get_tensor(output_details[0]['index'])
    return output_details

@app.route('/', methods=[ 'POST','GET'])
def test_ui():
    if request.method == 'POST':  
        f = request.files['files-upload'] 
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print(f.filename)
        result = predict(UPLOAD_FOLDER+"\\"+str(f.filename)) # result is a probabilities array
        classes = ['cataract', 'diabetic retinopathy', 'glaucoma', 'normal']
        max_result = (np.max(result, axis=-1)) * 100 # max probability
        pred_prob = np.format_float_positional(max_result, precision=2) # format probability
        pred_class = classes[(np.argmax(result, axis=-1)[0])] # string
        print(pred_class)
        print(result)
        print(pred_prob)
        return render_template("succee.html",data_disease=pred_class)
    return render_template("file_path.html")



if __name__ == "__main__":
    app.run(debug=True)