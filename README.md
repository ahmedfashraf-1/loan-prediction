<h1 align="center"><b>📊 Loan Defaulter Prediction</b></h1>

### 1. 📖 Introduction  
This project describes the dataset used for the *Loan Default Prediction* task, explains where the data comes from, lists the project contributors, and provides a complete explanation of each file in the dataset.  

The main goal of the project is to build predictive models that estimate the probability of a client defaulting on a loan (*binary target*).  

The dataset contains three files:  
1. *application_data.csv*  
2. *previous_application.csv*  
3. *columns_description.csv*  

The workflow involves combining these files, performing preprocessing, feature engineering, and exploratory data analysis to create robust predictive models.  

---

### 💻 Contributers

| Name            | GitHub Profile                                                   |
|-----------------|------------------------------------------------------------------|
| Ahmed Ashraf    | [Ahmed Ashraf](https://github.com/ahmedfashraf-1)                |
| Malak Ahmed     | [Malak Ahmed](https://github.com/Malak-A7med)                    |
| Tasneem Hussein | [Tasneem Hussein](https://github.com/tasneemhussein12)           |
| Mohamed Sheta   | [Mohamed Sheta](https://github.com/Mohamed-Sheta)                |
| Ossama Ayman    | [Ossama Ayman](https://github.com/OssamaAyman)                   |

---

### 2. 🎯 Objective

The objective of this case study is to apply Exploratory Data Analysis (EDA) techniques in a real-world business scenario within the banking and financial services domain.  
The focus is to develop a basic understanding of risk analytics by analyzing customer data and their previous loan applications.  
By combining demographic, financial, and historical loan information, the study aims to identify key factors that influence the risk of default. This will help in minimizing financial losses for the bank by improving lending decisions and building a stronger risk assessment framework.

---

### 3. 📂 Dataset Overview

The dataset comprises information about loan applicants and their credit histories. Use the files together to perform thorough EDA (exploratory data analysis), create aggregated historical features, and build robust predictive models.

**Here is The Columns Description:**  
[columns_description.csv (Google Drive Link)](https://drive.google.com/file/d/14OPssUiciOdcfhzXwMsviVBg1058gIbw/view?usp=sharing)

#### Primary responsibilities for each file

- **application_data.csv** — main data: client-level features and the TARGET label (0 = non-default, 1 = default). This is the file you will use for preprocessing, training, and evaluating models.
- **previous_application.csv** — historical loan applications from the same clients. Use it to create additional features (counts, averages, ratios) summarizing past behaviour for each SK_ID_CURR.
- **columns_description.csv** — column dictionary: definitions and explanations of columns across the dataset. Use this as a reference throughout analysis.

#### ⚡ Main workflow

1. Start with `application_data.csv` for cleaning, encoding, and baseline modeling.
2. Aggregate `previous_application.csv` by SK_ID_CURR to create historical features, then merge them into the main table for improved predictions.
3. Use `columns_description.csv` as the authoritative reference for column meanings and units.

---

### 5. ✅ Conclusion  
This dataset provides a comprehensive view of both *current* and *historical loan applications*.  
By performing thorough *EDA, feature engineering, and predictive modeling, we can identify the **key drivers of default risk*, thereby supporting smarter and safer lending decisions for financial institutions.

---
>📂**Data Source:** [Kaggle — Loan Defaulter dataset](https://www.kaggle.com/datasets/gauravduttakiit/loan-defaulter)  

>📦**Repository Link:** [ahmedfashraf-1/loan-prediction](https://github.com/ahmedfashraf-1/loan-prediction)
