# Recommender Systems Toolkit

A comprehensive recommendation engine implementing collaborative filtering, content-based filtering, and hybrid approaches for personalized recommendations.

## Description

The Recommender Systems Toolkit provides a unified interface for building recommendation systems using various algorithms. It supports user-based and item-based collaborative filtering, content-based recommendations, matrix factorization techniques, and hybrid methods to deliver personalized recommendations.

## Key Features

- **Collaborative Filtering**
  - User-based collaborative filtering with similarity metrics
  - Item-based collaborative filtering
  - Cosine similarity and Pearson correlation
  - Adjustable neighborhood sizes

- **Content-Based Filtering**
  - Feature-based item recommendations
  - TF-IDF for text-based features
  - Cosine similarity for content matching
  - User profile building from interaction history

- **Matrix Factorization**
  - Singular Value Decomposition (SVD)
  - Non-negative Matrix Factorization (NMF)
  - Latent factor models
  - Dimensionality reduction for sparse matrices

- **Hybrid Methods**
  - Weighted combination of multiple approaches
  - Cascading recommenders
  - Feature augmentation
  - Ensemble recommendations

- **Evaluation Metrics**
  - Precision and recall at K
  - Mean Average Precision (MAP)
  - Normalized Discounted Cumulative Gain (NDCG)
  - Coverage and diversity metrics
  - Root Mean Square Error (RMSE)

## Technologies Used

- **Python 3.x**
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **scikit-learn** - Machine learning algorithms and metrics
- **SciPy** - Sparse matrices and similarity computations

## Installation

```bash
# Clone the repository
cd /home/user/BrillConsulting/DataScience/RecommenderSystems

# Install required packages
pip install numpy pandas scikit-learn scipy
```

## Usage Examples

### User-Based Collaborative Filtering

```python
from recommender_systems import RecommenderSystem
import numpy as np
import pandas as pd

# Create sample user-item rating matrix
ratings_data = {
    'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
    'item_id': [101, 102, 103, 101, 104, 102, 103, 105, 101, 105],
    'rating': [5, 3, 4, 4, 5, 3, 5, 4, 5, 3]
}
ratings_df = pd.DataFrame(ratings_data)

# Initialize recommender
recommender = RecommenderSystem()

# Fit the model
recommender.fit(ratings_df, user_col='user_id', item_col='item_id', rating_col='rating')

# Get user-based recommendations
recommendations = recommender.collaborative_filtering_user(
    user_id=1,
    n_recommendations=5,
    n_neighbors=3
)

print(f"Top recommendations for user 1: {recommendations['item_ids']}")
print(f"Predicted ratings: {recommendations['predicted_ratings']}")
```

### Item-Based Collaborative Filtering

```python
# Get item-based recommendations
item_recommendations = recommender.collaborative_filtering_item(
    user_id=2,
    n_recommendations=5,
    n_neighbors=3
)

print(f"Item-based recommendations: {item_recommendations['item_ids']}")
print(f"Similarity scores: {item_recommendations['similarity_scores']}")
```

### Content-Based Filtering

```python
# Create item features for content-based filtering
item_features = pd.DataFrame({
    'item_id': [101, 102, 103, 104, 105],
    'genre': ['Action', 'Comedy', 'Action', 'Drama', 'Action'],
    'year': [2020, 2019, 2021, 2020, 2022]
})

# Fit content-based recommender
recommender.fit_content_based(
    item_features,
    item_col='item_id',
    feature_cols=['genre', 'year']
)

# Get content-based recommendations
content_recs = recommender.content_based_filtering(
    user_id=1,
    n_recommendations=5
)

print(f"Content-based recommendations: {content_recs['item_ids']}")
print(f"Similarity scores: {content_recs['content_scores']}")
```

### Matrix Factorization (SVD)

```python
# Apply SVD matrix factorization
svd_result = recommender.matrix_factorization_svd(
    n_factors=10,
    random_state=42
)

# Get recommendations using latent factors
svd_recommendations = recommender.predict_svd(
    user_id=3,
    n_recommendations=5
)

print(f"SVD recommendations: {svd_recommendations['item_ids']}")
print(f"Predicted ratings: {svd_recommendations['predicted_ratings']}")
print(f"Explained variance ratio: {svd_result['explained_variance_ratio']:.3f}")
```

### Hybrid Recommender

```python
# Combine multiple recommendation approaches
hybrid_recs = recommender.hybrid_recommendations(
    user_id=1,
    n_recommendations=5,
    weights={
        'collaborative': 0.5,
        'content': 0.3,
        'svd': 0.2
    }
)

print(f"Hybrid recommendations: {hybrid_recs['item_ids']}")
print(f"Combined scores: {hybrid_recs['combined_scores']}")
print(f"Method contributions: {hybrid_recs['method_contributions']}")
```

### Evaluation Metrics

```python
# Evaluate recommendation quality
test_data = ratings_df.sample(frac=0.2, random_state=42)

metrics = recommender.evaluate(
    test_data,
    k=5,
    metrics=['precision', 'recall', 'ndcg', 'rmse']
)

print(f"Precision@5: {metrics['precision@5']:.3f}")
print(f"Recall@5: {metrics['recall@5']:.3f}")
print(f"NDCG@5: {metrics['ndcg@5']:.3f}")
print(f"RMSE: {metrics['rmse']:.3f}")
print(f"Coverage: {metrics['coverage']:.3f}")
print(f"Diversity: {metrics['diversity']:.3f}")
```

## Demo Instructions

Run the comprehensive demo to see all features in action:

```bash
python recommender_systems.py
```

The demo will:
1. Generate synthetic user-item interaction data
2. Apply all recommendation methods (user-based CF, item-based CF, content-based, SVD, hybrid)
3. Generate recommendations for sample users
4. Evaluate recommendation quality with multiple metrics
5. Compare different approaches
6. Display performance metrics and top recommendations

## Output Examples

**Console Output:**
```
Recommender Systems Toolkit Demo
======================================================================

Generating synthetic user-item interaction data...
Total users: 100
Total items: 50
Total ratings: 1000
Sparsity: 80.0%

1. User-Based Collaborative Filtering
----------------------------------------------------------------------
User: 42
Top 5 Recommendations: [23, 45, 12, 38, 7]
Predicted Ratings: [4.8, 4.6, 4.5, 4.4, 4.3]

2. Item-Based Collaborative Filtering
----------------------------------------------------------------------
User: 42
Top 5 Recommendations: [23, 12, 45, 31, 7]
Average Similarity: 0.782

3. Content-Based Filtering
----------------------------------------------------------------------
User: 42
Top 5 Recommendations: [15, 23, 8, 45, 29]
Content Match Scores: [0.95, 0.92, 0.89, 0.87, 0.85]

4. Matrix Factorization (SVD)
----------------------------------------------------------------------
Number of latent factors: 20
Explained variance: 85.3%
Top 5 Recommendations: [23, 45, 12, 15, 7]
Predicted Ratings: [4.9, 4.7, 4.6, 4.5, 4.4]

5. Hybrid Recommender
----------------------------------------------------------------------
User: 42
Top 5 Recommendations: [23, 45, 12, 15, 38]
Combined Scores: [4.85, 4.68, 4.52, 4.38, 4.25]

Evaluation Metrics
----------------------------------------------------------------------
Method               Precision@5  Recall@5     NDCG@5       RMSE
----------------------------------------------------------------------
User-based CF        0.642        0.423        0.721        0.892
Item-based CF        0.638        0.418        0.715        0.905
Content-based        0.521        0.352        0.634        1.124
SVD                  0.678        0.445        0.753        0.845
Hybrid               0.692        0.461        0.768        0.831

Coverage: 78.5%
Diversity: 0.856
```

## Author

**Brill Consulting**

---

For more information about the algorithms and methodologies, see the inline documentation in `recommender_systems.py`.
