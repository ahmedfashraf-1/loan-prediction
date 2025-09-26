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

### 2. 💻 Contributers

| Name            | GitHub Profile                                                   |
|-----------------|------------------------------------------------------------------|
| Ahmed Ashraf    | [Ahmed Ashraf](https://github.com/ahmedfashraf-1)                |
| Malak Ahmed     | [Malak Ahmed](https://github.com/Malak-A7med)                    |
| Tasneem Hussein | [Tasneem Hussein](https://github.com/tasneemhussein12)           |
| Mohamed Sheta   | [Mohamed Sheta](https://github.com/Mohamed-Sheta)                |
| Ossama Ayman    | [Ossama Ayman](https://github.com/Ossama-Ayman)                  |

---

### 3. 🎯 Project Objective

The objective of this project is to use Machine Learning techniques on real-world banking and financial data to predict the likelihood of loan default. By analyzing demographic, financial, and historical loan information, the study aims to identify key factors influencing risk, enabling the bank to make informed lending decisions, minimize financial losses, and build a robust risk assessment framework. The insights from this analysis will also support the development of more accurate predictive models and enhance overall risk analytics

---

### 4. 📂 Dataset Overview

The dataset comprises information about loan applicants and their credit histories. Use the files together to perform thorough EDA (exploratory data analysis), create aggregated historical features, and build robust predictive models.

<div style="border: 2px solid #fff; padding: 16px; border-radius: 8px; background: #222; color: #fff; margin-bottom: 1.5em;">
<b>Here is The Columns Description:</b><br>
<a href="https://drive.google.com/file/d/14OPssUiciOdcfhzXwMsviVBg1058gIbw/view?usp=sharing" style="color:#fff; text-decoration:underline;">columns_description.csv (Google Drive Link)</a>
</div>

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

>📑**Google Colab Notebook:** [Loan Defaulter Prediction Notebook](https://colab.research.google.com/drive/16tdhFYxIqNBiy_HznSF4_-874vzx9R-E?usp=drive_link#scrollTo=PdX_qoeKZi4h)
