This project focuses on classifying car images using the Stanford Car Dataset, which consists of 16,185 images categorized into 196 classes based on Make, Model, and Year. The dataset is divided into 8,144 training images and 8,041 testing images.

Given the challenge of limited images per class, the project employs transfer learning. A pretrained model from ImageNet is fine-tuned on the car dataset to develop a robust car classifier. This involves adjusting all layers of the model and replacing the final fully connected layer to suit the specific classification requirements.

After fine-tuning the model and training it on the entire dataset, the model and necessary reference files were downloaded locally. A user-friendly interface was then created using Streamlit for local deployment, providing an intuitive platform to interact with the classifier.
