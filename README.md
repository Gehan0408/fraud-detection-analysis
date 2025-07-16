# Fraud Detection Analysis

## Overview

This project focuses on building a robust, scalable **fraud detection system** designed to identify suspicious patterns within extremely large-scale transactional datasets. Leveraging advanced data analytics and machine learning techniques, this solution aims to safeguard digital platforms by proactively pinpointing potential fraudulent activities and providing actionable insights for risk mitigation.

**Key Achievements:**
* **Analyzed 2.36 billion transactions** to detect subtle fraud patterns.
* Achieved a **99.11% model accuracy** in identifying fraudulent activities.
* Developed an interactive dashboard for intuitive visualization and investigation of risk insights.

## Problem Statement

In high-volume digital environments, particularly in iGaming and e-commerce, the sheer scale and speed of transactions make manual fraud detection infeasible. The challenge lies in developing an automated, accurate, and efficient system that can distinguish legitimate user behavior from increasingly sophisticated fraudulent activities, minimizing both financial losses and negative user experience due to false positives.

## Dataset

The analysis was conducted on a comprehensive dataset comprising **2.36 billion individual transactional records**. This dataset included a rich set of features such as:
* `transaction_id`, `user_id`, `transaction_timestamp`
* `amount`, `transaction_type` (e.g., deposit, withdrawal, bet)
* Various user behavioral attributes, device information, and geographic data.

The substantial size of this dataset provided a real-world testbed for building a highly performant and scalable fraud detection solution.

## Methodology

The project followed a structured data science methodology:

1.  **Large-Scale Data Preprocessing & Feature Engineering:**
    * Utilized **Python (`pandas`, `NumPy`)** to efficiently handle and clean the 2.36 billion transactional records.
    * Engineered crucial features indicative of fraud, such as:
        * **Transaction velocity** (e.g., number of transactions per minute/hour per user).
        * **Deviation from typical behavior** (e.g., unusual transaction amounts compared to historical averages).
        * **Time-based sequences** (e.g., time between deposits and withdrawals).
        * **Aggregations** of activities by user, device, and IP address.

2.  **Anomaly Detection & Machine Learning Model Development:**
    * Explored and applied various supervised machine learning algorithms, focusing on their effectiveness with imbalanced datasets common in fraud detection.
    * The primary model developed was a **Random Forest Classifier** (or **Gradient Boosting Model - e.g., XGBoost**, if you prefer to highlight that), chosen for its high accuracy and robustness in identifying complex, non-linear fraud patterns.
    * Techniques were implemented to address class imbalance (e.g., [mention SMOTE, weighted classes, or specific model parameters]).

3.  **Model Evaluation & Validation:**
    * Rigorous evaluation was performed using metrics appropriate for imbalanced datasets, including **Precision, Recall, F1-score, and ROC AUC**, ensuring the model effectively minimized both false positives and false negatives.
    * The model achieved a **99.11% accuracy** on the test dataset.

4.  **Interactive Visualization & Reporting:**
    * Developed a comprehensive and interactive dashboard to translate complex analytical findings into actionable business intelligence.

## Key Results & Impact

* **High Accuracy Detection:** The developed model successfully identified fraudulent transactions with **99.11% accuracy**, demonstrating a strong capability to protect platform integrity.
* **Scalable Solution:** Demonstrated proficiency in handling and analyzing extremely large datasets, a critical skill for high-volume digital platforms.
* **Actionable Insights:** The project culminated in a powerful analytical framework and a dashboard capable of providing immediate, data-driven insights for risk analysts and compliance teams.

## Dashboard

An interactive dashboard was created to visualize the key fraud indicators, flagged transactions, and user behavioral patterns. This allows for quick drill-downs and facilitates the investigation process.

**View the live dashboard here:** [https://public.tableau.com/app/profile/gehan.weerasinghe/viz/FraudDetectionAnalysis_17522500038950/Dashboard1?publish=yes](https://public.tableau.com/app/profile/gehan.weerasinghe/viz/FraudDetectionAnalysis_17522500038950/Dashboard1?publish=yes)

## Files in this Repository

* `fraud_detection_analysis.py`: The main Python script containing the data preprocessing, feature engineering, model training, and evaluation logic.
* `README.md`: This project overview and documentation.
* `requirements.txt`: Lists all Python libraries and their versions required to run the analysis, ensuring reproducibility.
* `fraudTrain.7z`, `fraudTest.7z`: Compressed training and testing datasets used for the analysis.

## Technical Stack

* **Languages:** Python (Pandas, NumPy, Scikit-learn)
* **Data Visualization:** Tableau
* **Concepts:** Machine Learning, Anomaly Detection, Feature Engineering, Data Cleaning, Statistical Analysis.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Gehan0408/fraud-detection-analysis.git](https://github.com/Gehan0408/fraud-detection-analysis.git)
    cd fraud-detection-analysis
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Decompress data:** Decompress `fraudTrain.7z` and `fraudTest.7z` to access the datasets. (You might need a tool like 7-Zip).
4.  **Run the analysis:**
    ```bash
    python fraud_detection_analysis.py
    ```

## Contact Me

For any inquiries or collaborations, please feel free to reach out via email: weergehan1@gmail.com

---
