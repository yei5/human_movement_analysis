# Human activity recognition - delivery 1

1. **Introduction**

Human Activity Recognition refers to the process of using machine learning algorithms and sensor data from various devices to detect and categorize human activities such as walking, running, and cooking. This project aims to develop a system that can accurately recognize and classify 5 different human activities (walk towards the camera, walk away from the camera, turn around, sit down, stand up) using data collected through video recordings and pose estimation techniques using artificial intelligence.

2. **Context**

Human Activity Recognition (HAR) is a growing research area that focuses on enabling machines to automatically identify human actions from sensor or video data. It plays an essential role in applications such as healthcare, sports analysis, smart home systems, and human–computer interaction. Early work in HAR mainly relied on wearable sensors such as accelerometers and gyroscopes to capture motion signals for classifying activities like walking, running, or sitting [1], [4], [5]. While these sensor-based systems achieved reliable results, they often required participants to wear multiple devices, limiting usability and scalability.

To overcome these constraints, researchers have shifted toward vision-based approaches, which use cameras and computer vision algorithms to recognize activities from video. Jalal and Kamal [2] proposed one of the early real-time systems using depth silhouettes for recognizing actions in smart home environments, demonstrating the potential of non-intrusive sensing for daily activity monitoring. Similarly, Soomro et al. [3] introduced UCF101, a large public dataset containing 101 human action classes recorded in natural settings, which became a benchmark for training and evaluating vision-based models.

Recent advances in pose estimation and deep learning have further improved HAR accuracy and generalization. The OpenPose framework [6] provided a robust real-time solution for detecting multiple human body poses simultaneously through Part Affinity Fields (PAFs), allowing for precise joint localization and movement analysis. Complementing this, MediaPipe [7] introduced an efficient and lightweight framework for building perception pipelines capable of running on mobile and embedded devices, facilitating real-time activity analysis without specialized hardware.

By leveraging pose estimation tools like OpenPose and MediaPipe, this project aims to build a non-intrusive, real-time human activity recognition system that can classify simple actions (walking, turning, sitting, standing) using only camera input. The combination of established computer vision techniques and supervised learning algorithms provides a foundation for exploring how classical machine learning can achieve reliable classification based on pose landmarks while adhering to principles of accessibility, reproducibility, and ethical data handling.
3. **Research Questions**

The main research question this project aims to address is:
* Can we accurately classify simple human activities (walking towards the camera, walking away from the camera, turning around, sitting down, standing up) using pose estimation data extracted from video recordings?

Sub-questions include:
* Which machine learning algorithms (e.g., SVM, Random Forest, XGBoost) are most effective for classifying human activities based on pose landmarks?
* What features derived from pose landmarks (e.g., joint angles, relative distances) contribute most to accurate activity recognition?

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

6. **Evaluation metrics**

The performance of the machine learning models will be evaluated using the following metrics:

* Accuracy: The proportion of correctly classified instances out of the total instances.
* Precision: The proportion of true positive predictions out of all positive predictions made by the model.
* Recall: The proportion of true positive predictions out of all actual positive instances.
* F1-score: The harmonic mean of precision and recall, providing a balanced measure of model performance.
* Confusion Matrix: A table that summarizes the performance of the classification model by showing the true positive, true negative, false positive, and false negative predictions for each class.
* ROC Curve: A graphical representation of the model's performance across different classification thresholds, illustrating the trade-off between true positive rate and false positive rate.

7. **Data description and exploratory analysis**

The dataset for this project consists of video recordings of individuals performing five distinct human activities: walking towards the camera, walking away from the camera, turning around, sitting down, and standing up. The videos were captured in various environments and lighting conditions to ensure diversity and robustness in the data.

This videos were collected from 2 volunteers who consented to participate in the study. Each participant was recorded performing the activities multiple times to capture variations in movement patterns. The recordings were made using standard video cameras, ensuring that the data collection process was non-intrusive and respectful of participants' privacy. The videos will not be shared or published, and raw data video files will be kept offline to protect the privacy of individuals.

The videos were processed using the MediaPipe Pose estimation framework to extract 66 key body landmarks for each frame. These landmarks include critical points such as the nose, eyes, shoulders, elbows, wrists, hips, knees, and ankles. The extracted landmarks provide a comprehensive representation of human posture and movement.

The exploratory data analysis indicates that the dataset includes 4143 frames with 60 extracted features representing pose landmarks, biomechanical metrics, and derived measures. The five activities (walk_forward, walk_back, turn_around, sit_down, and stand_up) are balanced enough for model training, though walking actions dominate, with approximately 24% walk_back and 22% walk_forward. This suggests that the dataset is representative but slightly skewed toward locomotion actions, which can influence model bias toward those classes. However, the amount of videos per activity is the same it's their length what changes

No missing values were detected, confirming that the preprocessing and landmark extraction steps were performed correctly. The boxplots reveal distinct distributions of biomechanical features across activities. For example, the left and right knee angles show higher mean values during turn_around and walk_back, while lower averages appear in walk_forward and sit_down. This difference supports the hypothesis that knee flexion and extension angles provide meaningful discriminative power for classifying actions involving different leg dynamics. The trunk lateral inclination shows a wide range, shifting from negative in sitting and standing to positive in walking actions, indicating changes in torso orientation and stability across activities. Person height also varies systematically, with lower average height in seated frames, confirming that the extracted features correctly reflect posture and vertical movement.

The correlation matrix identified multiple highly correlated variables among positional coordinates and their visibilities, which is expected since many landmark coordinates move in parallel during a single action. Features such as left_hip_x, right_hip_y, or ankle coordinates correlate strongly, implying redundancy that can be reduced later through feature selection or dimensionality reduction to avoid overfitting. However, the main derived features such as angles, trunk inclination, and height appear to have lower intercorrelations, supporting their relevance as independent predictors.

Overall, the analysis demonstrates that the dataset is clean, balanced enough for initial modeling, and that the biomechanical variables exhibit clear variation across activities. The next steps for the report should highlight these findings as evidence that the features are meaningful, the dataset quality is adequate, and that subsequent phases can focus on normalization, feature selection, and testing model performance on classification tasks.


8. **Ethical considerations**

This project involves recording and analyzing videos of human participants, which raises issues of privacy, consent, and responsible AI use. To ensure ethical compliance, the development follows internationally recognized frameworks such as the General Data Protection Regulation (GDPR) [10] and the OECD Principles on Artificial Intelligence [11], which emphasize fairness, transparency, and accountability in data-driven systems.

All participants will provide informed consent before data collection, in line with the ethical standards for research involving human subjects as described in the Belmont Report [12]. Video data will be anonymized to prevent identification of individuals, focusing solely on body landmarks rather than facial or biometric information.

Moreover, this project adheres to the UNESCO Recommendation on the Ethics of Artificial Intelligence [13], promoting human-centered and socially beneficial AI. The system will not be used for surveillance or any form of monitoring beyond its academic scope. All data will be securely stored, and access will be limited to project members.

These measures collectively ensure that the project aligns with the principles of privacy protection, consent, and ethical integrity while advancing research in computer vision and human activity recognition. Accordingly, the following ethical guidelines will be followed:

* Ensure informed consent is obtained from all participants involved in video data collection.
* Anonymize video data to protect the privacy of individuals.
* Raw data video files will be kept offline and won't be shared or published.
* Since the data that will be used in this project focus on pose and movement, avoid collecting or storing any personally identifiable information (PII). 

9. **Potential dataset expanding strategies**

If the initial dataset is insufficient for training the selected machine learning models, some strategies to expand the dataset may include:

* Data Augmentation: Apply techniques such as rotation, scaling, and flipping to create variations of existing video recordings.

* Synthetic data generation: Use computer graphics or simulation tools to generate synthetic video data of human activities.

* Potential Artificial Data Sources: Explore publicly available datasets related to human activity recognition that may complement the collected data.

* Artificial intelligence techniques: Utilize generative models (e.g., GANs) to create synthetic pose data based on the existing dataset.

If any of these strategies are implemented, it will be documented in future project deliveries. This is just a proposal for the first delivery. Additionally, the feasibility and effectiveness of these strategies will be evaluated based on the project's requirements and constraints.

10. **Next steps**

The next steps for the project include:

* Data preparation: With the exploratory data analysis completed, the next step includes cleaning and preprocessing the data, as well as feature engineering to extract relevant features from the pose landmarks.
* Model selection and training: After data preparation, the focus will shift to selecting appropriate machine learning algorithms and training models using the prepared dataset.
* Model evaluation: Once the models are trained, they will be evaluated using appropriate metrics to assess their performance in classifying human activities.
* Data augmentation (if necessary): If the dataset is found to be insufficient, data augmentation techniques may be explored to enhance the dataset.
* Result analysis and interpretation: Analyze the results obtained from the models and interpret their implications for human activity recognition.
* Deployment planning: Finally, plan for the deployment of the trained model in a real-time application for human activity recognition.
* Real-time application development: Finally, the project will aim to develop a real-time application that utilizes the trained model for human activity recognition from live video input.


**References**

[1] M. Zeng, L. T. Nguyen, B. Yu, O. J. Mengshoel, J. Zhu, P. Wu, “Convolutional Neural Networks for Human Activity Recognition using Mobile Sensors,” Proc. 6th Int. Conf. Mobile Computing, Applications and Services, 2014.

[2] A. Jalal and S. Kamal, "Real-time life logging via a depth silhouette-based human activity recognition system for smart home services," 
in *Proc. 2014 11th IEEE Int. Conf. Advanced Video and Signal Based Surveillance (AVSS)*, Seoul, South Korea, 2014, pp. 74–80, 
doi: 10.1109/AVSS.2014.6918647.

[3] K. Soomro, A. R. Zamir, M. Shah, “UCF101: A Dataset of 101 Human Actions Classes from Videos in the Wild,” CRCV-TR-12-01, 2012.

[4] H. Bayat, M. Pomplun, D. Tran, “A Study on Human Activity Recognition Using Accelerometer Data from Smartphones,” Procedia Computer Science, vol. 34, 2014.

[5] M. Shoaib, S. Bosch, O. D. Incel, H. Scholten, P. Havinga, “Fusion of Smartphone Motion Sensors for Physical Activity Recognition,” Sensors, vol. 14, 2014.  

[6] Z. Cao, T. Simon, S.-E. Wei, Y. Sheikh, “OpenPose: Realtime Multi-Person 2D Pose Estimation Using Part Affinity Fields,” IEEE CVPR, 2017.  

[7] F. Lugaresi et al., “MediaPipe: A Framework for Building Perception Pipelines,” arXiv preprint arXiv:1906.08172, 2019.  

[8] European Union, “General Data Protection Regulation (GDPR),” Regulation (EU) 2016/679, 2018.

[9] OECD, “OECD Principles on Artificial Intelligence,” 2019.

[10] National Commission for the Protection of Human Subjects of Biomedical and Behavioral Research, “The Belmont Report,” 1979.

[11] UNESCO, “Recommendation on the Ethics of Artificial Intelligence,” Paris, 2021.






