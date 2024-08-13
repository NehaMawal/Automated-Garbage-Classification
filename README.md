# Project: Automated Garbage Classification Using Deep Learning

### Overview
* Leverages Convolutional Neural Networks (CNNs) and transfer learning to classify garbage images into seven categories: plastic, metal, paper, glass, cardboard, e-waste, and biological waste.
* Aims to enhance waste sorting efficiency and recycling processes by accurately categorizing various types of garbage.

### Key Features
* **Model Development**: Developed, trained, and fine-tuned five pre-trained CNN architectures using TensorFlow/Keras, with DenseNet121 achieving the highest validation accuracy of 91.42%.
* **Data Preprocessing**: Applied essential preprocessing techniques to ensure dataset consistency and improve model generalization.
* **Deployment**: Successfully deployed the top-performing DenseNet121 model on the Hugging Face platform, making it accessible for real-world applications and user interaction.

### Dataset
* **Dataset1 Link**: [Kaggle - Trash Type Image Dataset](https://www.kaggle.com/datasets/farzadnekouei/trash-type-image-dataset)
* **Dataset2 Link**: [Kaggle - Garbage Classification PZA](https://www.kaggle.com/code/phyoezawaung/garbage-classification-pza/input)
* Includes labeled images across seven categories, underwent preprocessing including resizing, normalization, and augmentation for training suitability.

### Model Training
* Explored and fine-tuned the following CNN architectures:
  * **VGG16**
  * **ResNet50**
  * **InceptionV3**
  * **DenseNet121** *(Top Performer with 91.42% validation accuracy)*
  * **EfficientNetB0*
* DenseNet121 was selected for deployment due to its superior performance.

### Deployment
* **Model Deployment**: Successfully deployed on Hugging Face, enabling interaction and testing in real-world scenarios.
* **Access the Model**: [Model on Hugging Face](https://huggingface.co/spaces/NehaMaw/GarbageClassifier_Capstone))

### Business Questions
* How can we accurately classify different types of garbage using deep learning?
* How can we deploy the model for real-world applications?

### Conclusion
* Demonstrates the potential of deep learning in automating garbage classification.
* Deployment on a public platform invites further exploration and validation, contributing to more efficient waste management practices.
