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

### Project overview

In this project, I develop a classification algorithm using CNNs capable of processing an image, identifying a canine or human face, and subsequently predicting either the dog's breed or the dog breed resembled by the human. I also develop a web application using Flask which utilises the dog classification algorithm to analyse an image uploaded by the user.

The algorithm will require a combination of models to perform different tasks, with many possible solutions for each. My algorithm uses the following models for each step:

- Human detection - Pre-trained face detector from OpenCV's implementation of [Haar feature-based cascade classifiers](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html).

- Dog detection - Pre-trained [ResNet-50](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006) model with weights that have been trained on [ImageNet](http://www.image-net.org/).

- Dog breed classifier - 

Input datasets can be found at:

Background information such as the problem domain, the project origin, and related data sets or input data is provided.

Problem statement - The problem which needs to be solved is clearly defined. A strategy for solving the problem, including discussion of the expected solution, has been made.

Metrics - Metrics used to measure performance of a model or result are clearly defined. Metrics are justified based on the characteristics of the problem.

## Installation <a name="installation"></a>

In order to run the ETL and ML pipeline code you will need the standard libraries included within the Anaconda distribution of Python. The code should run with no issues using Python versions 3.*.

install opencv, keras, tensorflow

## File Descriptions <a name="files"></a>

The repository is divided into two directories based on the functionality of the files - model and app. The files listed below are available within each respective section of this repository.

Model:

- `train_classifier.py` - machine learning pipeline script that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model that uses the cleaned message data from process_data.py to predict classifications for 36 different categories (multi-output classification). This file also exports the trained model as a pickle file.

- This is saved as an XML file within the haarcascades directory.

Web App:

- `run.py` - script to build the back-end of the web app using the Flask framework.

- `master.html` - html file containing the front-end layout for the web app.

- `go.html` - html file containing the front-end layout for the search bar within the web app.

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

Data Exploration - Features and calculated statistics relevant to the problem have been reported and discussed related to the dataset, and a thorough description of the input space or input data has been made. Abnormalities or characteristics about the data or input that need to be addressed have been identified.

Data Visualization - Build data visualizations to further convey the information associated with your data exploration journey. Ensure that visualizations are appropriate for the data values you are plotting.

## Methodology <a name="methodology"></a>

See notebook for more information on the methods used.

## Results <a name="results"></a>

Model Evaluation and Validation - If a model is used, the following should hold: The final model’s qualities — such as parameters — are evaluated in detail. Some type of analysis is used to validate the robustness of the model’s solution.

Justification - The final results are discussed in detail.
Exploration as to why some techniques worked better than others, or how improvements were made are documented.

## Conclusion <a name="conclusion"></a>

Reflection - Student adequately summarizes the end-to-end problem solution and discusses one or two particular aspects of the project they found interesting or difficult.

Improvement - Discussion is made as to how at least one aspect of the implementation could be improved. Potential solutions resulting from these improvements are considered and compared/contrasted to the current solution.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to [Udacity](https://www.udacity.com/) for providing the project datasets to train and validate the classifier.
