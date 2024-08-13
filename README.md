# Automated-Garbage-Classification
Automated Garbage Classification Using Deep Learning
Objective:
Develop a robust deep learning model to classify different types of garbage, aiming to improve recycling efficiency and reduce environmental impact.
Dataset
Source: Custom Dataset

The dataset includes images of garbage categorized into seven types:

Paper
Plastic
Cardboard
Glass
Metal
Biological waste
E-waste
Business Questions
How can we accurately classify different types of garbage using deep learning?
How can we deploy the model for real-world applications?
Key Features
Image data: Various categories of garbage images.
Business Approach
1. Data Loading and Exploration
Load the dataset and explore the structure and contents.
2. Data Preprocessing
Resize images.
Normalize pixel values.
3. Model Building
Define the CNN model.
Compile the model.
4. Model Training
Train the model using the training data.
5. Model Evaluation
Evaluate the model's performance on the validation data.
Plot training and validation accuracy and loss.
6. Model Comparison
Compare different pre-trained models.
Select the best-performing model.
7. Prediction and Confusion Matrix
Make predictions on the validation data.
Generate and plot the confusion matrix.
8. Deployment and Application Link
Link: https://huggingface.co/spaces/NehaMaw/GarbageClassifier_Capstone
Key Insights:
The model was deployed on Hugging Face, making it accessible for users to interact with and test the model's predictions. The deployment enables stakeholders to evaluate the model in real-world scenarios and understand its potential for practical applications in waste classification.
Conclusion
This project aimed to classify different types of waste using convolutional neural networks (CNNs) and transfer learning techniques. By leveraging pre-trained models such as DenseNet121, we were able to achieve high accuracy and generalization performance.

Key Achievements:
Data Preprocessing:

Successfully preprocessed the dataset with consistent image dimensions and normalization.
Ensured a balanced dataset for robust training.
Model Training

Implemented and trained a custom CNN model.
Applied transfer learning with several pre-trained models including VGG16, ResNet50, InceptionV3, DenseNet121, and EfficientNetB0.
Model Evaluation::

Both VGG16 and DenseNet121 demonstrated high accuracy, with DenseNet121 showing slightly better generalization and stability.
Common misclassifications were observed due to visual similarities and variations in lighting and angles.
DenseNet121 achieved a validation accuracy of 91.42%, indicating robust performance for garbage classification tasks.
Model Deployment:

Successfully deployed the model on Hugging Face for public access and practical application.
