# Breast Cancer Classification with Neural Networks

This project aims to classify breast cancer tumors as either malignant or benign using neural networks. The dataset used is the Breast Cancer Wisconsin (Diagnostic) Data Set, which contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. 

## Dataset Overview

The dataset is available through scikit-learn's built-in datasets and comprises 569 instances with 30 features each. These features include measures like radius, texture, perimeter, area, smoothness, compactness, concavity, and more, computed from the digitized images. Each instance is labeled as either malignant (0) or benign (1).

## Implementation Steps

1. **Data Preparation**:
    - Load the Breast Cancer Wisconsin dataset.
    - Create a DataFrame containing the features and the target variable.
    
2. **Data Preprocessing**:
    - Standardize the features using `StandardScaler`.
    
3. **Neural Network Model**:
    - Build a neural network using TensorFlow and Keras.
    - Define the input layer with 30 neurons (one for each feature).
    - Add a hidden layer with 20 neurons and ReLU activation function.
    - Add the output layer with 2 neurons and sigmoid activation function.
    
4. **Model Training**:
    - Compile the model with 'adam' optimizer and 'sparse_categorical_crossentropy' loss function.
    - Train the model using the training data for 15 epochs.
    
5. **Model Evaluation**:
    - Visualize the training and validation accuracy and loss.
    - Evaluate the model's performance on the test data using accuracy and loss metrics.
    
6. **Model Evaluation**:
    - Make predictions on the test data.
    - Calculate and visualize the confusion matrix.
    - Generate a classification report including precision, recall, and F1-score.
    - Calculate and plot the ROC curve and calculate the ROC-AUC score.
    
7. **Predictive System**:
    - Build a predictive system to classify new instances of breast cancer tumors.
    - Standardize the input data.
    - Make predictions and interpret the results as either malignant or benign.
    
8. **Model Saving**:
    - Save the trained model for future use.

## How to Use

1. Ensure you have the necessary libraries installed (`numpy`, `pandas`, `matplotlib`, `scikit-learn`, `tensorflow`, `keras`).
2. Run the provided Python script to execute the breast cancer classification system.
3. Modify the input data in the predictive system section to classify new instances of breast cancer tumors.
4. Interpret the model's predictions as either malignant or benign.

## Conclusion

This project demonstrates the use of neural networks for breast cancer classification, achieving high accuracy and providing valuable insights for medical diagnosis. The trained model can be used to classify new instances of breast cancer tumors, aiding medical professionals in decision-making and improving patient care.
