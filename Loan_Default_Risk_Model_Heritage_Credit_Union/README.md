# Loan Default Risk Model — Heritage Credit Union
**By Aisha Nomani, Sai Han Shan Pha, Luis Alay Guerrero (Fernando), Aurelia Vo, and An Chu**

---

## Overview

This is a team project where we built a cost-sensitive binary classification model to assess loan default risk for a credit union using historical borrower data. The goal wasn't just to predict defaults accurately — it was to make the model useful for an actual underwriting decision, where missing a real default costs a lot more than flagging a loan that would have been fine.

---

## Approach

- **Exploratory data analysis and feature engineering** to find the key drivers of default risk
- **Model comparison** — we trained and compared logistic regression, decision trees, random forest, and gradient boosting, all evaluated with cross-validation
- **Cost-sensitive threshold tuning** — instead of optimizing for raw accuracy, we tuned the classification threshold around the real business cost trade-off between false negatives (missed defaults) and false positives (good loans flagged), so the model minimizes total financial loss and stays within underwriting capacity

---

## Result

The final model supports risk-based loan review decisions and is projected to reduce annual default losses by approximately **$450K**.

---

## Files

| File | What it is |
|------|-----------|
| `LoanDefault_TheBacksideUndergrads_FinalProject.ipynb` | The full notebook — EDA, feature engineering, model training, and threshold optimization |
| `Executive_Summary.pdf` | A short write-up of the findings and recommendation |
