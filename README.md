# E-Commerce Recommendation System Analysis

This repository contains Jupyter notebooks and datasets for comparing traditional and hybrid recommendation systems using Amazon review data.

---

## 🛠️ Steps

### 1. Data Preprocessing
- **Dataset**: [Amazon Review](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews) (568k reviews → 219k after filtering).
- **Key Features**:
  - `ProductId`, `UserId`, `Score` (1-5), `Time` (Unix timestamp)
  - `Summary`, `Text` (review content), `HelpfulnessNumerator/Denominator`
- **Cleaning Steps**:
  - Handle missing values (`ProfileName`, `Summary`)
  - Filter sparse users/items (≥5 interactions)
  - Create `HelpfulnessRatio = HelpfulnessNumerator / (Denominator + 1e-6)`
  - Convert `Time` to datetime

### 2. Model Implementation
| Model                | Tools Used          | Key Features                          |
|----------------------|---------------------|---------------------------------------|
| Collaborative Filtering | Surprise (KNNBasic) | User-based similarity (MSD metric)    |
| Matrix Factorization | Surprise (SVD)      | 100 latent factors                    |
| Hybrid Model         | VADER Sentiment     | 60% CF + 30% sentiment + 10% helpfulness |

### 3. Evaluation Metrics
- **RMSE**: Root Mean Square Error for rating predictions
- **Precision@10**: Relevance of top 10 recommendations
- **Cold-Start RMSE**: Performance on users/items absent in training data

---

## 📊 Interpretation of Findings

### Performance Comparison
| Model          | RMSE   | Precision@10 | Cold-Start RMSE |
|----------------|--------|--------------|------------------|
| Collaborative  | 1.282  | 0.757        | 1.324            |
| Matrix Fact.   | 1.236  | 0.761        | 1.310            |
| Hybrid         | 1.215  | 0.761        | 1.274            |

### Key Insights
- **Hybrid Model**: 4.8% RMSE improvement over CF, 1.7% over MF.
- **Cold-Start**: Hybrid reduced errors by 3.8% vs. CF but underperforms state-of-the-art models (e.g., Review-Based MF achieves 12% improvement).
- **Limitations**: Static weighting (60/30/10) and VADER’s simplistic sentiment analysis.

---

## 🚀 How to Use
1. Install dependencies:
   ```bash
   pip install pandas numpy surprise nltk matplotlib
   python -m nltk.downloader vader_lexicon
