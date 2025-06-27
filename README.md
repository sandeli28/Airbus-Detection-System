Airbus Ship Detection Using Deep Learning — Project Summary This project explores the application of deep learning techniques, specifically Convolutional Neural Networks (CNNs), to detect the presence of ships in satellite imagery. Accurate ship detection from satellite images is crucial in domains such as maritime traffic monitoring, illegal fishing detection, naval security, and environmental protection.

The foundation of this work is inspired by the Airbus Ship Detection Challenge, which provides real-world satellite imagery and aims to push the boundaries of computer vision techniques applied in remote sensing.

📌 Objective The primary objective of the project is to:

Develop a robust binary classification model that can determine whether a given satellite image contains a ship or not.

Understand the data preparation pipeline for satellite imagery.

Set a foundation for advanced applications like object localization or semantic segmentation.

📊 Dataset and Preprocessing The dataset includes:

Thousands of satellite images (256x256 pixels) taken over oceans and coastal regions.

Corresponding segmentation masks provided in Run-Length Encoding (RLE) format which indicate the exact pixels where ships are located.

Data preprocessing involved:

Decoding the RLE masks to generate binary mask images.

Categorizing images into “ship” and “no-ship” classes.

Resizing images for computational efficiency.

Normalizing pixel values for faster model convergence.

A balanced dataset was maintained to avoid bias towards shipless images, which were more frequent in the original data.

🧠 Model Architecture The classification model is built using a Convolutional Neural Network (CNN) architecture. The architecture includes:

Multiple convolutional layers with ReLU activation for feature extraction.

MaxPooling layers to reduce dimensionality and retain essential features.

Dropout layers to prevent overfitting.

Dense (fully connected) layers for classification.

A final output layer with a sigmoid activation function for binary output.

The model was trained using:

Binary Cross-Entropy Loss.

Adam Optimizer.
Evaluation metrics including accuracy, loss, and visual inspection of predictions.

📈 Results and Analysis The CNN model achieved strong performance in accurately detecting ship presence in test images.

Accuracy and loss curves indicate proper convergence and good generalization on unseen data.

Qualitative analysis (visual comparison) confirmed that the model was correctly distinguishing between images with and without ships.

This proves the effectiveness of CNNs for image classification tasks even with relatively small image sizes and varied visual conditions (e.g., clouds, waves).

📦 Technologies Used Python: Primary language for implementation.

TensorFlow & Keras: For model building and training.

OpenCV: Image manipulation and preprocessing.

NumPy & Pandas: Data manipulation and numerical operations.

Matplotlib: Visualization of results, including sample images and training graphs
