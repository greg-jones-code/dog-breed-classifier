# dog-breed-classifier

## Table of Contents

1. [Project Definition](#defintion)
2. [Installation](#installation)
3. [File Descriptions](#files)
4. [Running the Code](#running)
5. [Analysis](#analysis)
6. [Methodology](#methodology)
7. [Results](#results)
8. [Conclusion](#conclusion)
9. [Licensing, Authors, and Acknowledgements](#licensing)

## Project Definition <a name="defintion"></a>

### Project Overview and Problem Statement

In this project, I develop a classification algorithm using CNNs capable of processing an image, identifying a canine or human face, and subsequently predicting either the dog's breed or the dog breed resembled by the human. I also develop a web application using Flask which utilises the dog classification algorithm to analyse an image uploaded by the user.

The algorithm will require a combination of models to perform different tasks, with many possible solutions for each. My algorithm uses the following models for each step:

- Human detection - Pre-trained face detector from OpenCV's implementation of [Haar feature-based cascade classifiers](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html).

- Dog detection - Pre-trained [ResNet-50](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006) model with weights that have been trained on [ImageNet](http://www.image-net.org/).

- Dog breed classifier - Pre-trained [ResNet-50](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006) model.

### Metrics

Model accuracy is used to assess the performance of the models and overall algorithm. Given that the data sets are well balanced (each breed of dog has a similar number of photos within the training dataset) this metric is the most appropriate measure of performance for the characteristics of this problem.

### Project Instructions

The web application can be run without needing to run the dog_app.ipynb notebook - please see [Section 4 - Running the Code](#running) for instructions on how to run the web application.

To run the dog_app.ipynb notebook and build a new dog breed classification algorithm and detection models then follow the below instructions:

1. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip). Unzip the folder and save the contents in a directory within the project repository (i.e. data/dog_images). Amend the file path within the 'Import Dog Dataset' section of the 'dog_app' notebook to suit.

2. Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip). Unzip the folder and save the contents in a directory within the project repository (i.e. data/lfw). Amend the file path within the 'Import Human Dataset' section of the 'dog_app' notebook to suit. If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder.

3. Download the [VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz) for the dog dataset. Place it in a directory within the project repository (i.e. bottleneck_features).

4. (Optional) __If you plan to install TensorFlow with GPU support on your local machine__, follow [the guide](https://www.tensorflow.org/install/) to install the necessary NVIDIA software on your system.  If you are using an EC2 GPU instance, you can skip this step.

5. (Optional) **If you are running the project on your local machine (and not using AWS)**, create (and activate) a new environment.

	- __Linux__ (to install with __GPU support__, change `requirements/dog-linux.yml` to `requirements/dog-linux-gpu.yml`):
	```
	conda env create -f requirements/dog-linux.yml
	source activate dog-breed-classifier
	```  
	- __Mac__ (to install with __GPU support__, change `requirements/dog-mac.yml` to `requirements/dog-mac-gpu.yml`):
	```
	conda env create -f requirements/dog-mac.yml
	source activate dog-breed-classifier
	```  
	**NOTE:** Some Mac users may need to install a different version of OpenCV
	```
	conda install --channel https://conda.anaconda.org/menpo opencv3
	```
	- __Windows__ (to install with __GPU support__, change `requirements/dog-windows.yml` to `requirements/dog-windows-gpu.yml`):  
	```
	conda env create -f requirements/dog-windows.yml
	activate dog-breed-classifier
	```

6. (Optional) **If you are running the project on your local machine (and not using AWS)** and Step 5 throws errors, try this __alternative__ step to create your environment.

	- __Linux__ or __Mac__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`):
	```
	conda create --name dog-breed-classifier python=3.5
	source activate dog-breed-classifier
	pip install -r requirements/requirements.txt
	```
	**NOTE:** Some Mac users may need to install a different version of OpenCV
	```
	conda install --channel https://conda.anaconda.org/menpo opencv3
	```
	- __Windows__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`):  
	```
	conda create --name dog-breed-classifier python=3.5
	activate dog-breed-classifier
	pip install -r requirements/requirements.txt
	```

7. (Optional) **If you are using AWS**, install Tensorflow.
```
sudo python3 -m pip install -r requirements/requirements-gpu.txt
```

8. Switch [Keras backend](https://keras.io/backend/) to TensorFlow.
	- __Linux__ or __Mac__:
		```
		KERAS_BACKEND=tensorflow python -c "from keras import backend"
		```
	- __Windows__:
		```
		set KERAS_BACKEND=tensorflow
		python -c "from keras import backend"
		```

9. (Optional) **If you are running the project on your local machine (and not using AWS)**, create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `dog-breed-classifier` environment.
```
python -m ipykernel install --user --name dog-breed-classifier --display-name "dog-breed-classifier"
```

10. Open the dog_app notebook.
```
jupyter notebook dog_app.ipynb
```

11. (Optional) **If you are running the project on your local machine (and not using AWS)**, before running code, change the kernel to match the dog-breed-classifier environment by using the drop-down menu (**Kernel > Change kernel > dog-breed-classifier**). Then, follow the instructions in the notebook.

12. Run the code within the dog_app notebook to obtain the new models.

## Installation <a name="installation"></a>

In order to run the project code you will need the standard libraries included within the Anaconda distribution of Python. The code should run with no issues using Python versions 3.*.

The additional packages required to run the web application can be found in requirements.txt within the web_app directory. To install them, navigate to the web_app directory and simply run:

`pip install -r requirements.txt`

## File Descriptions <a name="files"></a>

The repository is divided into numerous directories based on the functionality of the files. The files listed below are available within each respective section of this repository.

### Main repository

- `dog_app.ipynb` - Jupyter notebook containing code to build, train, test and save the face and dog detection models, as well as the dog breed classifier algorithm.

- `extract_bottleneck_features.py` - Python script containing the functions used to extract the bottleneck features for the pre-trained models.

- `dog_names.pkl` - Pickle file containing a list of the dog breeds included within the training dataset.

- `dog_img1` to `dog_img5` - Sample images containing different breeds of dog to test the final classifier algorithm.

- `human_img1` to `human_img5` - Sample images containing human faces to test the final classifier algorithm.

### web_app

- `run.py` - Python script to build the back-end of the web app using the Flask framework.

- `requirements.txt` - Text file containing a list of the packages required to run the web application.

- `templates/upload_image.html` - HTML file containing the front-end layout for the web app.

### saved_models

- `weights.best.VGG16.hdf5` - Model weights for the VGG-16 classifier with the best validation loss.

- `weights.best.from_scratch.hdf5` - Model weights for the custom built classifier with the best validation loss.

- `weights.best.resnet50.hdf5` - Model weights for the ResNet-50 classifier with the best validation loss.

### haarcascades

- `haarcascade_frontalface_alt.xml` - Pre-trained face detector from OpenCV's implementation of [Haar feature-based cascade classifiers](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) saved as an XML file.

## Running the Code <a name="running"></a>

To build, train and test the models used in the classification algorithm, run the dog_app.ipynb notebook.

To run the web application on a local machine and upload an image to be classified by the algorithm, follow the below instructions using the command line:

1. Ensure that you have all the necessary packages installed on your local machine.
2. Clone this repository to your local machine using the command ```git clone https://github.com/greg-jones-code/dog-breed-classifier.git```.
3. Navigate to the web_app directory.
4. Run the web application using the command ```python run.py```
5. Go to http://0.0.0.0:3001/
6. Upload an image of a dog or a human (ensure that the face is clearly visible and not obstructed) and push the 'Classify Image' button. The image will be analysed by the algorithm and the most resembling breed will be displayed along with the submitted image.

## Analysis <a name="analysis"></a>

An analysis of the datasets is provided within the project notebook.

The dog images dataset used to develop the dog detector and breed classifier contains 8,351 images of 133 different breeds. This has been split into training, validation and testing sets at a ratio of 80:10:10. This dataset is well balanced, having a similar number of images for each breed.

The human images dataset used to develop the human face detector model contains 13,233 colour images of humans with clearly presented faces.

## Methodology <a name="methodology"></a>

See project notebook for more information on the methods used.

### Data Preprocessing

In order to accurately detect human faces the RGB images from the human images input dataset were converted to grayscale using the colour conversion function available within OpenCV:

`cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`

In order to utilise pre-trained Keras CNNs for the dog detection and breed classifier algorithms, the input format required is a 4D array or tensor with shape:

(nb_samples, rows, columns, channels)

where `nb_samples` corresponds to the total number of images, and `rows`, `columns`, and `channels` correspond to the number of rows, columns, and channels for each image, respectively. Using Keras' image preprocessing functionality the images from the dog images and ImageNet input datasets were resized to 224x224 pixels, converted to an array, and subsequently resized to a 4D tensor. Since the dog images datasets contain colour images, each image has three channels. Therefore, the returned tensor will always have shape:

(1, 224, 224, 3)

Lastly, the 4D tensors of the RGB images contained within the dog images and ImageNet input datasets were converted to BGR by reordering the channels in order to prepare them as input for the pre-trained Keras models. All pre-trained models have an additional normalisation step where the mean pixel (expressed in RGB as [103.939,116.779,123.68] and calculated from all pixels in all the images in the input dataset) must be subtracted from every pixel in each image. This was implemented using the imported function `preprocess_input` which is available [here](https://github.com/fchollet/keras/blob/master/keras/applications/imagenet_utils.py).

### Implementation

A pre-trained face detector from OpenCV's implementation of [Haar feature-based cascade classifiers](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) was used to detect human faces in images. The custom function takes a string-valued file path to an image as input, converts the image from RGB to grayscale, and uses the `detectMultiScale` function to execute the pre-trained face detector. The function returns True if a human face is detected in an image and False otherwise.

A pre-trained [ResNet-50](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006) model with weights that have been trained on [ImageNet](http://www.image-net.org/) was used to detect dogs within images. The custom function takes a string-valued file path to an image as input, converts the image to a 4D tensor and reformats it for input into a pre-trained Keras model. The model predictions are extracted using the `predict` method, which returns an array whose i-th entry is the model's predicted probability that the image belongs to the i-th ImageNet category. By taking the argmax of the predicted probability vector, we obtain an integer corresponding to the model's predicted object class, which we can identify with an object category through the use of this [dictionary](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a). Given that the categories corresponding to dogs appear in an uninterrupted sequence and correspond to dictionary keys 151-268, if the integer obtained is a value between 151 and 268 (inclusive), then the image is predicted to contain a dog and the function returns True (and False otherwise).

A pre-trained [ResNet-50](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006) model was created with transfer learning using bottleneck features and used to classify breeds from images of dogs. The custom function takes a string-valued file path to an image as input, converts the image to a 4D tensor, extracts the bottleneck features as input to the pre-trained Keras model. The model predictions are extracted using the `predict` method and the argmax of this prediction vector is used to return the index of the predicted dog breed. This is then converted to the breed name using the dog_names dictionary, which is returned as the output of the function.

These three custom functions are implemented within the dog breed classifier algorithm which accepts a string-valued file path to an image as input and first determines whether the image contains a human, dog, or neither. Then,

- if a dog is detected in the image, the algorithm returns the predicted breed.
- if a human is detected in the image, the algorithm returns the resembling dog breed.
- if neither is detected in the image, the algorithm returns an error message and tips to improve performance.

## Results <a name="results"></a>

A thorough evaluation and validation of the model performance is provided in the project notebook, including potential future improvements.

### Face detection model

The face detection model managed to successfully detect human faces in 100% of the human images, however, it also detected human faces in 11% of the dog images. There is therefore room for improvement with the face detection algorithm but it still gives an acceptable level of performance.

The choice of algorithm unfortunately necessitates that only images with a clear view of the face can be accepted. This is a reasonable expectation to pose on the user, especially given the high success rate of the algorithm when presented with a clear view of a face. However, if users are unable to provide a clear view of their face, potentially due to religious beliefs, then another approach to detecting humans that could be explored would be using an algorithm that identifies facial features such as eyes, nose or mouth.

![face detector](https://github.com/greg-jones-code/dog-breed-classifier/blob/main/readme_images/face_detector.png)

### Dog detection models

The dog detection model performs very well, managing to successfully detect dogs in 100% of the dog images whilst also correctly identifying that there were no dogs in the human images dataset.

![dog detector](https://github.com/greg-jones-code/dog-breed-classifier/blob/main/readme_images/dog_detector.png)

### Dog breed classifiers

Assigning breeds to dogs from images is exceptionally challenging. Even a human would have great difficulty in distinguishing between a Brittany and a Welsh Springer Spaniel.

![dog breed classifier](https://github.com/greg-jones-code/dog-breed-classifier/blob/main/readme_images/dog_breed_classifier.png)

The final CNN architecture chosen for the dog breed classifier was created with transfer learning using ResNet-50 bottleneck features. This consists of a GlobalAveragePooling layer and a fully connected Dense layer, which uses a softmax function with 133 output nodes (1 for each breed of dog contained within the training data) and returns a vector containing the probabilities that the input image belongs to each class. I have also included a dropout layer to prevent overfitting.

After testing each of the pre-trained models, it was decided that ResNet50 provided the optimal solution to this problem due to the combination of high accuracy and minimal activation map size. The VGG-16 model only obtained a test accuracy of approximately 44%, whereas the ResNet-50 model was able to classify dog breeds with a test accuracy in excess of 80%.

### Web application

The web application allows a user to upload an image of their choice and once submitted (by clicking on the 'classify image' button) provides the user with a message identifying the canine or human face, and predicting either the dog's breed or the dog breed resembled by the human.

![web app home](https://github.com/greg-jones-code/dog-breed-classifier/blob/main/readme_images/web_app_home.png)

The application performs very well when analysing images of dogs - correctly identifying the German Shepherd below!

![web app dog](https://github.com/greg-jones-code/dog-breed-classifier/blob/main/readme_images/web_app_dog.png)

When it comes to images of humans, the algorithms accuracy is open to interpretation. Do you think this person looks like a Pointer?

![web app human](https://github.com/greg-jones-code/dog-breed-classifier/blob/main/readme_images/web_app_human.png)

We think so, but not sure that the Pointer agrees!

![web app pointer](https://github.com/greg-jones-code/dog-breed-classifier/blob/main/readme_images/web_app_pointer.png)

## Conclusion <a name="conclusion"></a>

In general, the dog breed classifier algorithm performs very well and has exceeded my initial expectations for this project. It correctly identifies whether an image contains either a dog or a human, and has a high accuracy when classifying the breed of a dog. In addition the algorithm also provides compelling resembling breeds for the human images.

Some potential future improvements that could be made to the algorithm are:

- The algorithm occasionally misclassifies very similar looking breeds. This could be improved by increasing the size of the training data set so that the breed classifier is able to more clearly distinguish between these similar breeds.

- Increasing the depth of the CNNs used in the algorithm could help to improve the accuracy of classification. This improved performance would however have to be balanced against the reduction in computational performance (i.e. time taken to analyse an image).

- Augmentation of the images within the training datasets through transformations such as random scaling, cropping and flipping could help the models generalise better, leading to better accuracy.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to [Udacity](https://www.udacity.com/) for providing the project datasets to train and validate the classifier.
