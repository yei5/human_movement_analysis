# Human activity recognition - delivery 1

1. **Introduction**

Human Activity Recognition refers to the process of using machine learning algorithms and sensor data from various devices to detect and categorize human activities such as walking, running, and cooking. This project aims to develop a system that can accurately recognize and classify 5 different human activities (walk towards the camera, walk away from the camera, turn around, sit down, stand up) using data collected through video recordings and pose estimation techniques using artificial intelligence.

2. **Context**

The project is being developed as part of the "Inteligencia Artificial I" course at Universidad Icesi, Cali, Colombia. The goal is to create a tool that can analyze human movement in real time and automatically classify specific actions. The system will utilize MediaPipe Pose for landmark detection, extracting numerical features (angles, inclinations, velocities) to feed into supervised learning models such as SVM, Random Forest, or XGBoost.

3. **Research Questions**

The main research questions that this project aims to address are:

* Which features extracted from pose estimation data are most relevant for accurately classifying human activities?
* How do different machine learning algorithms (e.g., SVM, Random Forest, XGBoost) perform in classifying human activities based on pose estimation data?
* What is the impact of data preprocessing techniques (e.g., normalization, feature scaling) on the performance of activity recognition models?
* How the data privacy and ethical considerations be addressed when collecting and using video data for human activity recognition?

4. **Problem type**

This is a supervised multi-class classification problem where the goal is to classify human activities into one of the predefined categories based on the features extracted from pose estimation data.
The model will learn from labeled samples of human movements and predict activity labels for unseen sequences.

5. **Metodology(CRISP-DM adaptation)**

The project will follow an adapted CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology, which includes the following phases:

* Business understanding: 

    - Define the relevance of recognizing simple human movements in real time and the potential social or academic impact.
    - Identify potential ethical considerations and data privacy concerns related to video data collection.
  
* Data understanding:
  
  - Collect video recordings of individuals performing the 5 target activities on different conditions and angles.
  - Understand the structure and format of the collected video data.
  - Explore the data to identify patterns, anomalies, and potential challenges.
  

* Data preparation:

    - Extract pose landmarks from the video data using MediaPipe Pose.
    - Clean and preprocess the extracted data, including handling missing values and outliers.
    - Engineer relevant features (e.g., angles, inclinations, velocities) from the pose landmarks.
    - Split the data into training and testing sets.

* Modeling: 
    - Select and implement machine learning algorithms (SVM, Random Forest, XGBoost) for activity classification.
    - Train the models using the training dataset.
    - Tune hyperparameters to optimize model performance.4

* Evaluation:
    - Evaluate the models using metrics such as accuracy, precision, recall, and F1-score.
    - Compare the performance of different models and select the best-performing one.
    - Validate the model's performance on the testing dataset.

* Deployment:
    - Develop a real-time application that utilizes the trained model to recognize human activities from live video input.
    - Ensure the application is user-friendly and provides accurate activity recognition results.
    - Monitor the system's performance and make necessary updates based on user feedback and new data.

5. **Evaluation metrics**

The performance of the machine learning models will be evaluated using the following metrics:

* Accuracy: The proportion of correctly classified instances out of the total instances.
* Precision: The proportion of true positive predictions out of all positive predictions made by the model.
* Recall: The proportion of true positive predictions out of all actual positive instances.
* F1-score: The harmonic mean of precision and recall, providing a balanced measure of model performance.
* Confusion Matrix: A table that summarizes the performance of the classification model by showing the true positive, true negative, false positive, and false negative predictions for each class.
* ROC Curve: A graphical representation of the model's performance across different classification thresholds, illustrating the trade-off between true positive rate and false positive rate.

6. **Ethical considerations**

Since this project involves collecting and analyzing video data of human subjects, that may include personally identifiable information (PII), it is crucial to address ethical considerations related to data privacy and informed consent. This project is developed in an academic context, therefore it is essential to adhere to ethical guidelines and best practices for handling sensitive data. In order to ensure ethical compliance, the following measures will be implemented:

* Ensure informed consent is obtained from all participants involved in video data collection.
* Anonymize video data to protect the privacy of individuals.
* Since the data that will be used in this project focus on pose and movement, avoid collecting or storing any personally identifiable information (PII). 

7. **Potential dataset expanding strategies**

If the initial dataset is insufficient for training the selected machine learning models, some strategies to expand the dataset may include:

* Data Augmentation: Apply techniques such as rotation, scaling, and flipping to create variations of existing video recordings.

* Synthetic data generation: Use computer graphics or simulation tools to generate synthetic video data of human activities.

* Potential Artificial Data Sources: Explore publicly available datasets related to human activity recognition that may complement the collected data.

* Artificial intelligence techniques: Utilize generative models (e.g., GANs) to create synthetic pose data based on the existing dataset.

If any of these strategies are implemented, it will be documented in future project deliveries. This is just a proposal for the first delivery. Additionally, the feasibility and effectiveness of these strategies will be evaluated based on the project's requirements and constraints.

8. **Next steps**

The next steps for the project include:

* Data preparation: With the exploratory data analysis completed, the next step includes cleaning and preprocessing the data, as well as feature engineering to extract relevant features from the pose landmarks.
* Model selection and training: After data preparation, the focus will shift to selecting appropriate machine learning algorithms and training models using the prepared dataset.
* Model evaluation: Once the models are trained, they will be evaluated using appropriate metrics to assess their performance in classifying human activities.
* Data augmentation (if necessary): If the dataset is found to be insufficient, data augmentation techniques may be explored to enhance the dataset.
* Result analysis and interpretation: Analyze the results obtained from the models and interpret their implications for human activity recognition.
* Deployment planning: Finally, plan for the deployment of the trained model in a real-time application for human activity recognition.
* Real-time application development: Finally, the project will aim to develop a real-time application that utilizes the trained model for human activity recognition from live video input.





