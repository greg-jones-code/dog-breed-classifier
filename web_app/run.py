import os
import numpy as np
import cv2
import pickle
from flask import Flask, flash, render_template, request, redirect, url_for
from sklearn.datasets import load_files
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.models import load_model
from werkzeug.utils import secure_filename


app = Flask(__name__)
accepted_extensions = {'png', 'jpg', 'jpeg'}
upload_folder = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = upload_folder


def accepted_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in accepted_extensions


def face_detector(img_path: str) -> bool:

    '''
    Returns "True" if a face is detected in image stored at img_path.

    Inputs:
    img_path - file path to an image.

    Returns:
    "True" or "False".
    '''

    # extract pre-trained face detector
    face_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_alt.xml')
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


def path_to_tensor(img_path: str):

    '''
    Takes a string-valued file path to a colour image as input and returns a 4D tensor suitable for supplying to a Keras CNN.

    Inputs:
    img_path - file path to an image.

    Returns:
    4D tensor.
    '''

    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))

    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)

    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def ResNet50_predict_labels(img_path: str):

    '''
    Extract model predictions for image.

    Inputs:
    img_path - file path to an image.

    Returns:
    Numpy array whose i-th entry is the model's predicted probability that the image belongs to the i-th ImageNet category.
    '''

    # define ResNet50 model
    ResNet50_model = ResNet50(weights='imagenet')

    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))

    return np.argmax(ResNet50_model.predict(img))


def dog_detector(img_path: str) -> bool:

    '''
    Returns "True" if a dog is detected in the image stored at img_path.

    Inputs:
    img_path - file path to an image.

    Returns:
    "True" or "False".
    '''

    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))


def extract_Resnet50(tensor):

    '''
    Extract bottleneck features corresponding to ResNet50 CNN.

    Inputs:
    tensor - 4D tensor.

    Returns:
    Bottleneck features for ResNet50 model.
    '''

    return ResNet50(weights='imagenet', include_top=False).predict(preprocess_input(tensor))


def resnet50_predict_breed(img_path: str) -> str:

    '''
    Accepts a file path to an image and returns the dog breed that is predicted by the model.

    Inputs:
    img_path - file path to an image.

    Returns:
    predicted_breed - dog breed predicted by the model.
    '''

    # Load trained classifier
    resnet50_model = load_model('../saved_models/weights.best.resnet50.hdf5')

    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))

    # obtain predicted vector
    predicted_vector = resnet50_model.predict(bottleneck_feature)

    # load list of dog names
    dog_names = pickle.load(open('../dog_names.pkl', 'rb'))
    dog_names = [name.replace('_', ' ') for name in dog_names]

    # return dog breed that is predicted by the model
    predicted_breed = dog_names[np.argmax(predicted_vector)]

    return predicted_breed


def predict_breed(img_path: str) -> str:

    '''
    Accepts a file path to an image and first determines whether the image contains a human, dog, or neither:

        - if a dog is detected in the image, algorithm returns the predicted breed.
        - if a human is detected in the image, algorithm returns the resembling dog breed.
        - if neither is detected in the image, algorithm provides an error message.

    Inputs:
    img_path - file path to an image.

    Returns:
    Predicted breed and submitted image.
    '''

    # Determine whether image contains a dog, human, or neither and return predicted breed
    if dog_detector(img_path):
        predicted_breed_long = resnet50_predict_breed(img_path)
        predicted_breed = predicted_breed_long.split('.', 1)[1]
        message = f"Hello dog!\nYou are a {predicted_breed}"

    elif face_detector(img_path):
        resembling_breed_long = resnet50_predict_breed(img_path)
        resembling_breed = resembling_breed_long.split('.', 1)[1]
        message = f"Hello human!\nYou resemble a {resembling_breed}"

    else:
        message = "Error.\nSorry, I am unable to detect a dog or human in this image - please try a different image.\nRemember if your image is of a human, ensure that the face is clearly visible in order to help me with my prediction."

    return message


# Index webpage displays and receives user input image for model prediction
@app.route('/')
def upload_image():
    return render_template('upload_image.html')


# Handle user input image and displays model prediction
@app.route('/', methods=['POST'])
def classify_image():
    if request.method == 'POST':

        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part in uploaded image')
            return redirect(request.url)
        img = request.files['file']

        # If user does not select file, browser also
        # Submit an empty part without filename
        if img.filename == '':
            flash('No image uploaded')
            return redirect(request.url)

        if img and accepted_file(img.filename):
            filename = secure_filename(img.filename)
            img.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # Classify image using model and return predicted breed
            classification = predict_breed(app.config['UPLOAD_FOLDER'] + filename)

            return render_template(
                'upload_image.html',
                filename=filename,
                classification=classification
                )
        else:
            flash('Accepted image file types are: .png, .jpg, .jpeg')
            return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename='uploads/' + filename))


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
