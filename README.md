# dl-cnn-googlenet-car_image_classification
This project focuses on car image classification using the Stanford Car Dataset, which includes 16,185 images categorized into 196 classes based on Make, Model, and Year. The dataset is split into 8,144 training images and 8,041 testing images.

Training a deep learning model directly on this dataset is challenging due to the limited number of images per class. To address this, the project utilizes transfer learning—a technique where a pretrained model from ImageNet is we fine-tuned on the car dataset. We aim to develop a car classifier by fine-tuning a pretrained model. All layers will be adjusted, and the final fully connected layer will be replaced for specific classification tasks.
