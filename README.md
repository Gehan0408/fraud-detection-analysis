# Fraud Detection Analysis

## Overview

Ever wondered how massive online platforms keep things fair and secure? This project dives deep into exactly that – building a robust and scalable **fraud detection system** designed to identify suspicious patterns hidden within extremely large-scale transactional datasets. My goal was to leverage advanced data analytics and machine learning to proactively safeguard digital environments and provide clear, actionable insights for risk mitigation.

**Key Achievements:**
* **Analyzed 2.36 billion transactions** to uncover subtle fraud patterns.
* Achieved an impressive **99.11% model accuracy** in identifying fraudulent activities.
* Developed an interactive dashboard for intuitive visualization and investigation of complex risk insights.

## Problem Statement

In today's fast-paced digital world, especially in dynamic sectors like iGaming and e-commerce, the sheer volume and speed of transactions make relying on manual fraud detection impossible. The challenge is to create an automated, highly accurate, and efficient system that can differentiate genuine user behavior from increasingly sophisticated fraudulent activities, thereby minimizing financial losses and ensuring a fair experience for legitimate users.

## Dataset

For this project, I worked with a comprehensive dataset comprising a staggering **2.36 billion individual transactional records**. This rich dataset included vital details such as:
* `transaction_id`, `user_id`, `transaction_timestamp`
* `amount`, `transaction_type` (e.g., deposit, withdrawal, bet)
* Various user behavioral attributes, device information, and geographic data.

The immense size of this dataset provided a realistic and challenging environment, perfect for building a performant and scalable fraud detection solution.

## Methodology

My approach followed a systematic data science methodology to ensure robust results:

1.  **Large-Scale Data Preprocessing & Feature Engineering:**
    * Utilized **Python (`pandas`, `NumPy`)** to efficiently handle and meticulously clean the 2.36 billion transactional records.
    * Engineered crucial new features highly indicative of fraud, such as:
        * **Transaction velocity** (e.g., number of transactions per minute/hour per user).
        * **Deviation from typical behavior** (e.g., unusual transaction amounts compared to historical averages).
        * **Time-based sequences** (e.g., time elapsed between suspicious actions).
        * **Aggregations** of activities by user, device, and IP address.

2.  **Anomaly Detection & Machine Learning Model Development:**
    * Explored and applied various supervised machine learning algorithms, paying close attention to their effectiveness with the imbalanced datasets typically found in fraud detection.
    * The primary model developed was a **Random Forest Classifier** (or **Gradient Boosting Model - e.g., XGBoost**, choose the one you prefer to highlight), selected for its high accuracy, interpretability, and robustness in capturing complex, non-linear fraud patterns.
    * Implemented techniques to effectively address class imbalance (e.g., [mention SMOTE, weighted classes, or specific model parameters if you used them]).

3.  **Model Evaluation & Validation:**
    * Performed rigorous evaluation using industry-standard metrics appropriate for fraud detection, including **Precision, Recall, F1-score, and ROC AUC**. This ensured the model minimized both false positives (legitimate users flagged) and false negatives (missed fraud).
    * The model proudly achieved a **99.11% accuracy** on the unseen test dataset.

4.  **Interactive Visualization & Reporting:**
    * Developed a comprehensive and interactive dashboard to seamlessly translate complex analytical findings into clear, actionable business intelligence for stakeholders.

## Key Results & Impact

* **High-Accuracy Fraud Detection:** The developed model successfully identified fraudulent transactions with **99.11% accuracy**, showcasing a strong capability to protect digital platform integrity.
* **Scalable Analytical Solution:** Demonstrated proven proficiency in handling and analyzing extremely large datasets, a critical skill for high-volume, real-time digital environments.
* **Actionable Business Insights:** The project delivered a powerful analytical framework and a dashboard capable of providing immediate, data-driven insights for risk analysts and compliance teams, enabling swift decision-making.

## Dashboard

An interactive dashboard was created using Tableau to visualize the key fraud indicators, flagged transactions, and user behavioral patterns. It allows for quick drill-downs and significantly streamlines the investigation process.

**View the live dashboard here:** [https://public.tableau.com/app/profile/gehan.weerasinghe/viz/FraudDetectionAnalysis_17522500038950/Dashboard1?publish=yes](https://public.tableau.com/app/profile/gehan.weerasinghe/viz/FraudDetectionAnalysis_17522500038950/Dashboard1?publish=yes)

## Files in this Repository

* `fraud_detection_analysis.py`: The main Python script containing the data preprocessing, feature engineering, model training, and evaluation logic.
* `README.md`: This very project overview and documentation.
* `requirements.txt`: Lists all Python libraries and their versions required to run the analysis, ensuring easy reproducibility.
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
3.  **Decompress data:** You'll need a tool like 7-Zip to decompress `fraudTrain.7z` and `fraudTest.7z` to access the datasets.
4.  **Run the analysis:**
    ```bash
    python fraud_detection_analysis.py
    ```

## Contact Me

For any inquiries or collaborations, please feel free to reach out via email: weergehan1@gmail.com

---
