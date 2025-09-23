# loan prediction

## 📖 Introduction  
This project describes the dataset used for the *Loan Default Prediction* task, explains where the data comes from, lists the project contributors, and provides a complete explanation of each file in the dataset.  

The main goal of the project is to build predictive models that estimate the probability of a client defaulting on a loan (*binary target*).  

The dataset contains three files:  
1. *application_data.csv*  
2. *previous_application.csv*  
3. *columns_description.csv*  

The workflow involves combining these files, performing preprocessing, feature engineering, and exploratory data analysis to create robust predictive models.  

---

## 🎯 Objective  
The objective of this case study is to apply *Exploratory Data Analysis (EDA)* techniques in a real-world business scenario within the banking and financial services domain.  
The focus is to develop a basic understanding of *risk analytics* by analyzing customer data and their previous loan applications.  

By combining demographic, financial, and historical loan information, the study aims to identify key factors that influence the risk of default.  
This will help in minimizing financial losses for the bank by improving lending decisions and building a stronger risk assessment framework.  

---

## 📊 Dataset Overview  

The dataset comprises information about *loan applicants* and their *credit histories*.  

### Primary responsibilities for each file  
- *application_data.csv* — main data: client-level features and the TARGET label (0 = non-default, 1 = default). Used for preprocessing, training, and evaluating models.  
- *previous_application.csv* — historical loan applications of the same clients. Used to create additional features (counts, averages, ratios) summarizing past behavior for each SK_ID_CURR.  
- *columns_description.csv* — column dictionary: definitions and explanations of columns across the dataset. Used as a reference during analysis.  

### Main Workflow  
1. Start with application_data.csv for cleaning, encoding, and baseline modeling.  
2. Aggregate previous_application.csv by SK_ID_CURR to create historical features, then merge them into the main table.  
3. Use columns_description.csv as the authoritative reference for column meanings and units.  

---

## 📂 File-by-File Breakdown  

### 1️⃣ application_data.csv — Main Application Data  
*Purpose*: Core dataset of current loan applications with client attributes and default label.  
*Target column*: TARGET — 1 = default, 0 = repaid.  

*Top 10 important columns*  

| Column              | Description                                    |
|---------------------|------------------------------------------------|
| SK_ID_CURR          | Unique client identifier.                      |
| TARGET              | Default indicator (1 = default, 0 = repaid).   |
| AMT_INCOME_TOTAL    | Total annual income of the client.             |
| AMT_CREDIT          | Credit amount of the loan.                     |
| AMT_ANNUITY         | Loan annuity (installment) amount.             |
| NAME_CONTRACT_TYPE  | Type of contract (Cash loans, Revolving loans).|
| CODE_GENDER         | Client gender.                                 |
| NAME_EDUCATION_TYPE | Education level of the client.                 |
| NAME_FAMILY_STATUS  | Marital status of the client.                  |
| EXT_SOURCE_3        | External risk score (very predictive feature). |

---

### 2️⃣ previous_application.csv — Historical Applications  
*Purpose*: Contains all previous loan applications of the same clients. Use to create historical features per SK_ID_CURR.  

*Top 10 important columns*  

| Column              | Description                                           |
|---------------------|-------------------------------------------------------|
| SK_ID_CURR          | Client identifier (link to application data).         |
| SK_ID_PREV          | Previous application identifier.                      |
| AMT_APPLICATION     | Amount of credit applied for.                         |
| AMT_CREDIT          | Amount of credit approved.                            |
| NAME_CONTRACT_TYPE  | Type of previous loan.                                |
| NAME_CONTRACT_STATUS| Status of the application (Approved, Refused…).       |
| PRODUCT_COMBINATION | Product combination offered.                          |
| DAYS_DECISION       | Days relative to current application decision.        |
| RATE_DOWN_PAYMENT   | Rate of down payment for the loan.                    |
| CHANNEL_TYPE        | Channel through which the loan was applied (branch, internet, etc.). |

---

### 3️⃣ columns_description.csv — Column Dictionary  
*Purpose*: Provides a human-readable description of each column across all files.  
Not used for modeling directly but essential for understanding variables.  

*Top 10 example columns*  

| Column name         | Meaning                                          |
|---------------------|--------------------------------------------------|
| SK_ID_CURR          | Client identifier used across files.             |
| TARGET              | Binary indicator of default.                     |
| AMT_CREDIT          | Amount of credit requested/approved.             |
| AMT_INCOME_TOTAL    | Client’s total annual income.                    |
| NAME_CONTRACT_STATUS| Application status in previous_application.csv.|
| EXT_SOURCE_3        | External risk score 3.                           |
| CODE_GENDER         | Gender of the client.                            |
| NAME_FAMILY_STATUS  | Marital status.                                  |
| NAME_EDUCATION_TYPE | Education level.                                 |
| CHANNEL_TYPE        | Channel type of previous applications.           |

---

## ✅ Conclusion  
This dataset provides a comprehensive view of both *current* and *historical loan applications*.  
By performing thorough *EDA, feature engineering, and predictive modeling, we can identify the **key drivers of default risk*, thereby supporting smarter and safer lending decisions for financial institutions.
