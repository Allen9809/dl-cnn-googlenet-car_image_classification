# Car Image Classification Using Transfer Learning

This project focuses on classifying car images using the Stanford Car Dataset, leveraging transfer learning with a pretrained model from ImageNet. Below are the key components of the project:

- **dl_cnn_car_classification.ipynb**: This Jupyter notebook is dedicated to developing the car image classification model. It involves data preprocessing, model fine-tuning, and training processes.
  
- **test.ipynb**: This Jupyter notebook is used for testing the app. It verifies the functionality and accuracy of the trained model on various test images.

- **app.py**: This Python script contains the code for the Streamlit app. It sets up the interface for users to upload car images and get classification results.

- **car_classes.txt**: This file lists the car classes, mapping the index to the specific car model. It is used by the model to interpret and display the classification results.

- **car_model.pth**: This file is the saved model artifact, containing the trained weights and architecture necessary for making predictions on new car images.

- **sample.jpeg**: This is a sample image provided for testing the app. It can be used to verify that the classification pipeline is working correctly.

With these components, you can develop, test, and deploy a robust car image classification application.

We focus on classifying car images using the Stanford Car Dataset, which consists of 16,185 images categorized into 196 classes based on Make, Model, and Year. The dataset is divided into 8,144 training images and 8,041 testing images.

Given the challenge of limited images per class, the project employs transfer learning. A pretrained model from ImageNet is fine-tuned on the car dataset to develop a robust car classifier. This involves adjusting all layers of the model and replacing the final fully connected layer to suit the specific classification requirements.

After fine-tuning the model and training it on the entire dataset, the model and necessary reference files were downloaded locally. A user-friendly interface was then created using Streamlit for local deployment, providing an intuitive platform to interact with the classifier.
