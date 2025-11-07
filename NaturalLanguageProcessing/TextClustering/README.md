# Advanced Text Clustering System v2.0

**Author:** BrillConsulting | **Version:** 2.0 - K-Means, DBSCAN, Hierarchical
**Methods:** K-Means, DBSCAN, Agglomerative, Spectral Clustering

## Overview

Cluster documents into semantic groups using TF-IDF vectorization and multiple clustering algorithms. Perfect for document organization, topic discovery, and content categorization.

## Features

- **Multiple Algorithms:** K-Means, DBSCAN, Hierarchical, Spectral
- **TF-IDF Vectorization:** Smart document representation
- **Automatic K:** DBSCAN finds optimal number of clusters
- **Visualization:** 2D PCA plots
- **Evaluation Metrics:** Silhouette score, Davies-Bouldin index
- **Large Scale:** Handle 10,000+ documents

## Installation

```bash
pip install scikit-learn numpy matplotlib
```

## Quick Start

```python
from text_clustering import TextClusterer

# Initialize
clusterer = TextClusterer(method='kmeans', n_clusters=5)

# Cluster documents
result = clusterer.cluster(documents)

# Results
print(f"Found {result['num_clusters']} clusters")
for cluster_id, docs in result['clusters'].items():
    print(f"Cluster {cluster_id}: {len(docs)} documents")
```

## Methods Comparison

| Method | Speed | Quality | Auto-K | Best For |
|--------|-------|---------|--------|----------|
| K-Means | ⚡⚡⚡ | ⭐⭐⭐ | ❌ | Known # clusters |
| DBSCAN | ⚡⚡ | ⭐⭐⭐⭐ | ✅ | Density-based |
| Hierarchical | ⚡ | ⭐⭐⭐⭐ | ❌ | Tree structure |
| Spectral | ⚡ | ⭐⭐⭐⭐⭐ | ❌ | Complex shapes |

## Usage Examples

### 1. News Article Clustering

```python
news_articles = load_news()
clusterer = TextClusterer('kmeans', n_clusters=10)

result = clusterer.cluster(news_articles)

# Analyze clusters
for cid, docs in result['clusters'].items():
    print(f"\nCluster {cid}:")
    for doc in docs[:3]:
        print(f"  - {doc['text'][:80]}...")
```

### 2. Customer Feedback Grouping

```python
feedback = load_customer_feedback()
clusterer = TextClusterer('dbscan')  # Auto finds clusters

result = clusterer.cluster(feedback)

# DBSCAN may find noise (cluster -1)
for cid, docs in result['clusters'].items():
    if cid == -1:
        print(f"Noise: {len(docs)} uncategorized")
    else:
        print(f"Topic {cid}: {len(docs)} feedbacks")
```

### 3. Research Paper Organization

```python
papers = [paper['abstract'] for paper in research_papers]
clusterer = TextClusterer('hierarchical', n_clusters=8)

result = clusterer.cluster(papers)

# Assign topics
topic_names = ['AI', 'ML', 'CV', 'NLP', 'Robotics', 'Theory', 'Applications', 'Other']
for cid, docs in result['clusters'].items():
    print(f"{topic_names[cid]}: {len(docs)} papers")
```

### 4. Email Categorization

```python
emails = load_emails()
clusterer = TextClusterer('kmeans', n_clusters=5)

result = clusterer.cluster([e['body'] for e in emails])

# Categorize
categories = ['Work', 'Personal', 'Spam', 'Promotions', 'Social']
for idx, email in enumerate(emails):
    cluster_id = result['labels'][idx]
    email['category'] = categories[cluster_id]
```

### 5. Product Review Grouping

```python
reviews = get_product_reviews()
clusterer = TextClusterer('dbscan')

result = clusterer.cluster([r['text'] for r in reviews])

# Find common complaints/praises
for cid, docs in result['clusters'].items():
    if len(docs) > 10:  # Significant cluster
        print(f"Common theme {cid}: {len(docs)} reviews")
```

## Command Line

```bash
# Basic clustering
python text_clustering.py \
    --file documents.txt \
    --method kmeans \
    --n-clusters 5

# Auto-find clusters with DBSCAN
python text_clustering.py \
    --file texts.txt \
    --method dbscan

# With visualization
python text_clustering.py \
    --file articles.txt \
    --method hierarchical \
    --n-clusters 8 \
    --visualize
```

## Choosing the Right Method

### Use K-Means when:
- ✅ You know the number of clusters
- ✅ Clusters are roughly spherical
- ✅ Need fast processing
- ✅ Documents evenly distributed

### Use DBSCAN when:
- ✅ Don't know number of clusters
- ✅ Have noise/outliers
- ✅ Clusters have varying densities
- ✅ Arbitrary cluster shapes

### Use Hierarchical when:
- ✅ Want hierarchical structure
- ✅ Need dendrogram visualization
- ✅ Small to medium datasets
- ✅ Multiple granularity levels

### Use Spectral when:
- ✅ Complex cluster shapes
- ✅ Non-convex clusters
- ✅ Highest quality needed
- ✅ Can afford computation time

## Visualization

```python
# 2D PCA visualization
clusterer.visualize(texts, result['labels'], save_path='clusters.png')

# Interactive plot
import plotly.express as px
from sklearn.decomposition import PCA

X = clusterer.vectorizer.fit_transform(texts)
X_2d = PCA(n_components=2).fit_transform(X.toarray())

fig = px.scatter(x=X_2d[:, 0], y=X_2d[:, 1], 
                 color=[str(l) for l in result['labels']],
                 hover_data=[texts])
fig.show()
```

## Evaluation Metrics

```python
from sklearn.metrics import silhouette_score, davies_bouldin_score

X = clusterer.vectorizer.fit_transform(texts)
labels = result['labels']

# Silhouette Score (higher is better, -1 to 1)
sil_score = silhouette_score(X, labels)
print(f"Silhouette: {sil_score:.3f}")

# Davies-Bouldin Index (lower is better)
db_score = davies_bouldin_score(X.toarray(), labels)
print(f"Davies-Bouldin: {db_score:.3f}")
```

## Finding Optimal K

```python
from sklearn.metrics import silhouette_score

# Elbow method
inertias = []
silhouettes = []

for k in range(2, 11):
    clusterer = TextClusterer('kmeans', n_clusters=k)
    result = clusterer.cluster(texts)
    
    X = clusterer.vectorizer.fit_transform(texts)
    silhouettes.append(silhouette_score(X, result['labels']))
    inertias.append(clusterer.model.inertia_)

# Plot and find elbow
optimal_k = silhouettes.index(max(silhouettes)) + 2
print(f"Optimal K: {optimal_k}")
```

## Best Practices

### 1. Preprocessing
```python
# Clean texts before clustering
texts = [clean_text(t) for t in raw_texts]

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text
```

### 2. Feature Engineering
```python
# Custom TF-IDF parameters
clusterer.vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.8,
    ngram_range=(1, 2),
    stop_words='english'
)
```

### 3. Handle Imbalanced Clusters
```python
# Filter small clusters
significant_clusters = {
    cid: docs 
    for cid, docs in result['clusters'].items() 
    if len(docs) > 10
}
```

## Troubleshooting

**Problem:** All documents in one cluster
**Solution:** Increase n_clusters, adjust DBSCAN eps parameter

**Problem:** Too many small clusters
**Solution:** Decrease n_clusters, adjust DBSCAN min_samples

**Problem:** Poor quality clusters
**Solution:** Improve text preprocessing, use more features, try different method

---

**BrillConsulting** - Advanced NLP Solutions
