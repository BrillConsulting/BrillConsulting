"""
Recommender Systems Toolkit
============================

Comprehensive recommendation algorithms and evaluation:
- Collaborative filtering (user-based, item-based)
- Matrix factorization (SVD)
- Content-based filtering
- Hybrid recommender systems
- Evaluation metrics (RMSE, MAE, Precision@K, Recall@K)
- Cold start handling
- Top-N recommendations

Author: Brill Consulting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class RecommenderSystem:
    """Comprehensive recommender system toolkit."""

    def __init__(self, random_state: int = 42):
        """
        Initialize recommender system.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        self.user_item_matrix = None
        self.user_similarity = None
        self.item_similarity = None
        self.predictions = None

    def create_user_item_matrix(self, ratings_df: pd.DataFrame,
                                user_col: str = 'user_id',
                                item_col: str = 'item_id',
                                rating_col: str = 'rating') -> np.ndarray:
        """
        Create user-item rating matrix from ratings dataframe.

        Args:
            ratings_df: DataFrame with user-item ratings
            user_col: Name of user ID column
            item_col: Name of item ID column
            rating_col: Name of rating column

        Returns:
            User-item matrix (users x items)
        """
        self.ratings_df = ratings_df.copy()
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col

        # Create pivot table
        self.user_item_matrix = ratings_df.pivot_table(
            index=user_col,
            columns=item_col,
            values=rating_col,
            fill_value=0
        ).values

        self.user_ids = ratings_df[user_col].unique()
        self.item_ids = ratings_df[item_col].unique()
        self.n_users, self.n_items = self.user_item_matrix.shape

        return self.user_item_matrix

    def user_based_collaborative_filtering(self, user_item_matrix: Optional[np.ndarray] = None,
                                          k_neighbors: int = 10) -> np.ndarray:
        """
        User-based collaborative filtering recommendations.

        Args:
            user_item_matrix: User-item rating matrix (if None, uses stored matrix)
            k_neighbors: Number of similar users to consider

        Returns:
            Predicted ratings matrix
        """
        if user_item_matrix is None:
            user_item_matrix = self.user_item_matrix

        # Calculate user similarity (cosine similarity)
        # Replace zeros with NaN for proper similarity calculation
        matrix_for_similarity = user_item_matrix.copy().astype(float)
        matrix_for_similarity[matrix_for_similarity == 0] = np.nan

        # Compute cosine similarity
        self.user_similarity = self._compute_similarity(user_item_matrix)

        # Make predictions
        predictions = np.zeros_like(user_item_matrix, dtype=float)

        for user_idx in range(self.n_users):
            # Get k most similar users
            similar_users = np.argsort(self.user_similarity[user_idx])[::-1][1:k_neighbors+1]

            for item_idx in range(self.n_items):
                if user_item_matrix[user_idx, item_idx] == 0:  # Only predict for unrated items
                    # Get ratings from similar users
                    similar_ratings = user_item_matrix[similar_users, item_idx]
                    similar_weights = self.user_similarity[user_idx, similar_users]

                    # Filter out users who haven't rated this item
                    mask = similar_ratings > 0
                    if np.sum(mask) > 0:
                        weighted_sum = np.sum(similar_ratings[mask] * similar_weights[mask])
                        similarity_sum = np.sum(similar_weights[mask])
                        predictions[user_idx, item_idx] = weighted_sum / (similarity_sum + 1e-10)

        self.predictions = predictions
        return predictions

    def item_based_collaborative_filtering(self, user_item_matrix: Optional[np.ndarray] = None,
                                          k_neighbors: int = 10) -> np.ndarray:
        """
        Item-based collaborative filtering recommendations.

        Args:
            user_item_matrix: User-item rating matrix (if None, uses stored matrix)
            k_neighbors: Number of similar items to consider

        Returns:
            Predicted ratings matrix
        """
        if user_item_matrix is None:
            user_item_matrix = self.user_item_matrix

        # Calculate item similarity (cosine similarity)
        self.item_similarity = self._compute_similarity(user_item_matrix.T)

        # Make predictions
        predictions = np.zeros_like(user_item_matrix, dtype=float)

        for user_idx in range(self.n_users):
            for item_idx in range(self.n_items):
                if user_item_matrix[user_idx, item_idx] == 0:  # Only predict for unrated items
                    # Get k most similar items that user has rated
                    user_rated_items = np.where(user_item_matrix[user_idx] > 0)[0]

                    if len(user_rated_items) > 0:
                        # Get similarities with rated items
                        similarities = self.item_similarity[item_idx, user_rated_items]
                        top_k_idx = np.argsort(similarities)[::-1][:k_neighbors]

                        top_similar_items = user_rated_items[top_k_idx]
                        top_similarities = similarities[top_k_idx]

                        # Calculate weighted average
                        if np.sum(top_similarities) > 0:
                            weighted_sum = np.sum(user_item_matrix[user_idx, top_similar_items] * top_similarities)
                            similarity_sum = np.sum(top_similarities)
                            predictions[user_idx, item_idx] = weighted_sum / (similarity_sum + 1e-10)

        self.predictions = predictions
        return predictions

    def matrix_factorization_svd(self, user_item_matrix: Optional[np.ndarray] = None,
                                 n_factors: int = 20) -> Dict:
        """
        Matrix factorization using SVD.

        Args:
            user_item_matrix: User-item rating matrix (if None, uses stored matrix)
            n_factors: Number of latent factors

        Returns:
            Dictionary with predictions and factor matrices
        """
        if user_item_matrix is None:
            user_item_matrix = self.user_item_matrix

        # Center the matrix by subtracting user means
        user_ratings_mean = np.mean(user_item_matrix, axis=1, keepdims=True)
        user_ratings_mean[user_ratings_mean == 0] = 0  # Keep zeros as zeros
        matrix_centered = user_item_matrix - user_ratings_mean

        # SVD
        U, sigma, Vt = svds(matrix_centered, k=min(n_factors, min(self.n_users, self.n_items) - 1))

        # Convert sigma to diagonal matrix
        sigma = np.diag(sigma)

        # Reconstruct predictions
        predictions = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean

        # Clip predictions to valid rating range
        predictions = np.clip(predictions, 0, np.max(user_item_matrix))

        self.predictions = predictions
        self.user_factors = U
        self.item_factors = Vt.T
        self.singular_values = sigma

        return {
            'predictions': predictions,
            'user_factors': U,
            'item_factors': Vt.T,
            'singular_values': sigma,
            'n_factors': n_factors
        }

    def content_based_filtering(self, item_features: np.ndarray,
                                user_item_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Content-based filtering using item features.

        Args:
            item_features: Item feature matrix (items x features)
            user_item_matrix: User-item rating matrix (if None, uses stored matrix)

        Returns:
            Predicted ratings matrix
        """
        if user_item_matrix is None:
            user_item_matrix = self.user_item_matrix

        # Normalize item features
        item_features_norm = normalize(item_features, axis=1)

        # Calculate item similarity based on features
        item_similarity = cosine_similarity(item_features_norm)

        # Make predictions
        predictions = np.zeros_like(user_item_matrix, dtype=float)

        for user_idx in range(self.n_users):
            # Get items rated by user
            rated_items = np.where(user_item_matrix[user_idx] > 0)[0]

            if len(rated_items) > 0:
                # For each unrated item
                for item_idx in range(self.n_items):
                    if user_item_matrix[user_idx, item_idx] == 0:
                        # Calculate similarity with rated items
                        similarities = item_similarity[item_idx, rated_items]
                        ratings = user_item_matrix[user_idx, rated_items]

                        # Weighted average
                        if np.sum(np.abs(similarities)) > 0:
                            predictions[user_idx, item_idx] = np.sum(similarities * ratings) / np.sum(np.abs(similarities))

        self.predictions = predictions
        return predictions

    def hybrid_recommender(self, collaborative_predictions: np.ndarray,
                          content_predictions: np.ndarray,
                          alpha: float = 0.5) -> np.ndarray:
        """
        Hybrid recommender combining collaborative and content-based filtering.

        Args:
            collaborative_predictions: Predictions from collaborative filtering
            content_predictions: Predictions from content-based filtering
            alpha: Weight for collaborative filtering (1-alpha for content-based)

        Returns:
            Hybrid predictions
        """
        hybrid_predictions = alpha * collaborative_predictions + (1 - alpha) * content_predictions
        self.predictions = hybrid_predictions
        return hybrid_predictions

    def top_n_recommendations(self, user_idx: int, n: int = 10,
                             predictions: Optional[np.ndarray] = None) -> List[Tuple[int, float]]:
        """
        Get top N recommendations for a user.

        Args:
            user_idx: User index
            n: Number of recommendations
            predictions: Prediction matrix (if None, uses stored predictions)

        Returns:
            List of (item_idx, predicted_rating) tuples
        """
        if predictions is None:
            predictions = self.predictions

        # Get user's predictions
        user_predictions = predictions[user_idx]

        # Get items not yet rated
        unrated_items = np.where(self.user_item_matrix[user_idx] == 0)[0]

        # Get top N from unrated items
        unrated_predictions = user_predictions[unrated_items]
        top_n_idx = np.argsort(unrated_predictions)[::-1][:n]

        recommendations = [(unrated_items[i], unrated_predictions[i]) for i in top_n_idx]

        return recommendations

    def evaluate_rmse(self, true_ratings: np.ndarray,
                     predicted_ratings: np.ndarray) -> float:
        """
        Calculate Root Mean Squared Error.

        Args:
            true_ratings: True ratings matrix
            predicted_ratings: Predicted ratings matrix

        Returns:
            RMSE value
        """
        # Only consider non-zero true ratings
        mask = true_ratings > 0
        if np.sum(mask) == 0:
            return 0.0

        rmse = np.sqrt(mean_squared_error(true_ratings[mask], predicted_ratings[mask]))
        return rmse

    def evaluate_mae(self, true_ratings: np.ndarray,
                    predicted_ratings: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error.

        Args:
            true_ratings: True ratings matrix
            predicted_ratings: Predicted ratings matrix

        Returns:
            MAE value
        """
        # Only consider non-zero true ratings
        mask = true_ratings > 0
        if np.sum(mask) == 0:
            return 0.0

        mae = mean_absolute_error(true_ratings[mask], predicted_ratings[mask])
        return mae

    def evaluate_precision_recall_at_k(self, true_ratings: np.ndarray,
                                       predicted_ratings: np.ndarray,
                                       k: int = 10,
                                       threshold: float = 3.5) -> Dict:
        """
        Calculate Precision@K and Recall@K.

        Args:
            true_ratings: True ratings matrix
            predicted_ratings: Predicted ratings matrix
            k: Number of top recommendations
            threshold: Rating threshold for relevance

        Returns:
            Dictionary with precision and recall
        """
        precisions = []
        recalls = []

        for user_idx in range(true_ratings.shape[0]):
            # True relevant items (rated >= threshold)
            relevant_items = np.where(true_ratings[user_idx] >= threshold)[0]

            if len(relevant_items) == 0:
                continue

            # Top K predicted items
            top_k_items = np.argsort(predicted_ratings[user_idx])[::-1][:k]

            # Recommended relevant items
            relevant_recommended = np.intersect1d(top_k_items, relevant_items)

            # Precision@K
            precision = len(relevant_recommended) / k if k > 0 else 0

            # Recall@K
            recall = len(relevant_recommended) / len(relevant_items) if len(relevant_items) > 0 else 0

            precisions.append(precision)
            recalls.append(recall)

        return {
            'precision_at_k': np.mean(precisions) if precisions else 0.0,
            'recall_at_k': np.mean(recalls) if recalls else 0.0,
            'k': k
        }

    def handle_cold_start_user(self, user_features: np.ndarray,
                               existing_user_features: np.ndarray,
                               k_neighbors: int = 5) -> np.ndarray:
        """
        Handle cold start problem for new users using demographic info.

        Args:
            user_features: Features of new user
            existing_user_features: Features of existing users
            k_neighbors: Number of similar users to consider

        Returns:
            Predicted ratings for new user
        """
        # Find similar users based on features
        similarities = cosine_similarity(user_features.reshape(1, -1), existing_user_features)[0]
        top_k_users = np.argsort(similarities)[::-1][:k_neighbors]

        # Average ratings from similar users
        predictions = np.mean(self.user_item_matrix[top_k_users], axis=0)

        return predictions

    def handle_cold_start_item(self, item_features: np.ndarray,
                               existing_item_features: np.ndarray,
                               k_neighbors: int = 5) -> np.ndarray:
        """
        Handle cold start problem for new items using content features.

        Args:
            item_features: Features of new item
            existing_item_features: Features of existing items
            k_neighbors: Number of similar items to consider

        Returns:
            Predicted ratings for new item across all users
        """
        # Find similar items based on features
        similarities = cosine_similarity(item_features.reshape(1, -1), existing_item_features)[0]
        top_k_items = np.argsort(similarities)[::-1][:k_neighbors]

        # Average ratings for similar items
        predictions = np.mean(self.user_item_matrix[:, top_k_items], axis=1)

        return predictions

    def _compute_similarity(self, matrix: np.ndarray) -> np.ndarray:
        """Compute cosine similarity matrix."""
        # Handle zero vectors
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero

        matrix_normalized = matrix / norms
        similarity = np.dot(matrix_normalized, matrix_normalized.T)

        return similarity

    def visualize_user_item_matrix(self, user_item_matrix: Optional[np.ndarray] = None,
                                   title: str = "User-Item Rating Matrix") -> plt.Figure:
        """
        Visualize user-item rating matrix as heatmap.

        Args:
            user_item_matrix: User-item matrix (if None, uses stored matrix)
            title: Plot title

        Returns:
            Matplotlib figure
        """
        if user_item_matrix is None:
            user_item_matrix = self.user_item_matrix

        fig, ax = plt.subplots(figsize=(12, 8))

        # Limit display to first 50 users and items for visibility
        display_matrix = user_item_matrix[:min(50, self.n_users), :min(50, self.n_items)]

        sns.heatmap(display_matrix, cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Rating'})

        ax.set_xlabel('Item Index', fontsize=12)
        ax.set_ylabel('User Index', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()
        return fig

    def visualize_prediction_errors(self, true_ratings: np.ndarray,
                                   predicted_ratings: np.ndarray) -> plt.Figure:
        """
        Visualize prediction errors.

        Args:
            true_ratings: True ratings matrix
            predicted_ratings: Predicted ratings matrix

        Returns:
            Matplotlib figure
        """
        # Get non-zero ratings
        mask = true_ratings > 0
        true_vals = true_ratings[mask]
        pred_vals = predicted_ratings[mask]

        errors = true_vals - pred_vals

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Error distribution
        axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        axes[0].set_xlabel('Prediction Error (True - Predicted)', fontsize=11)
        axes[0].set_ylabel('Frequency', fontsize=11)
        axes[0].set_title('Error Distribution', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # True vs Predicted scatter
        axes[1].scatter(true_vals, pred_vals, alpha=0.3, s=20)
        axes[1].plot([true_vals.min(), true_vals.max()],
                    [true_vals.min(), true_vals.max()],
                    'r--', linewidth=2, label='Perfect Prediction')
        axes[1].set_xlabel('True Rating', fontsize=11)
        axes[1].set_ylabel('Predicted Rating', fontsize=11)
        axes[1].set_title('True vs Predicted Ratings', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        return fig


def demo():
    """Demonstrate recommender system toolkit."""
    np.random.seed(42)

    print("Recommender Systems Toolkit Demo")
    print("=" * 70)

    # Generate synthetic ratings data
    print("\nGenerating synthetic user-item ratings...")
    n_users = 100
    n_items = 50
    n_ratings = 1000

    # Generate random ratings
    user_ids = np.random.choice(n_users, n_ratings)
    item_ids = np.random.choice(n_items, n_ratings)
    ratings = np.random.choice([1, 2, 3, 4, 5], n_ratings, p=[0.1, 0.15, 0.2, 0.3, 0.25])

    ratings_df = pd.DataFrame({
        'user_id': user_ids,
        'item_id': item_ids,
        'rating': ratings
    })

    # Remove duplicates
    ratings_df = ratings_df.drop_duplicates(subset=['user_id', 'item_id'])

    print(f"Users: {n_users}")
    print(f"Items: {n_items}")
    print(f"Ratings: {len(ratings_df)}")
    print(f"Sparsity: {(1 - len(ratings_df) / (n_users * n_items)) * 100:.2f}%")

    # Split into train and test
    train_size = int(0.8 * len(ratings_df))
    train_df = ratings_df.iloc[:train_size]
    test_df = ratings_df.iloc[train_size:]

    # Initialize recommender
    recommender = RecommenderSystem(random_state=42)

    # Create user-item matrix
    train_matrix = recommender.create_user_item_matrix(train_df)
    print(f"\nTrain matrix shape: {train_matrix.shape}")

    # Create test matrix for evaluation
    test_matrix = pd.DataFrame(0, index=range(n_users), columns=range(n_items))
    for _, row in test_df.iterrows():
        test_matrix.loc[row['user_id'], row['item_id']] = row['rating']
    test_matrix = test_matrix.values

    # 1. User-based Collaborative Filtering
    print("\n1. User-based Collaborative Filtering")
    print("-" * 70)
    user_cf_predictions = recommender.user_based_collaborative_filtering(k_neighbors=10)
    rmse_user = recommender.evaluate_rmse(test_matrix, user_cf_predictions)
    mae_user = recommender.evaluate_mae(test_matrix, user_cf_predictions)
    print(f"RMSE: {rmse_user:.4f}")
    print(f"MAE: {mae_user:.4f}")

    metrics_user = recommender.evaluate_precision_recall_at_k(test_matrix, user_cf_predictions, k=10)
    print(f"Precision@10: {metrics_user['precision_at_k']:.4f}")
    print(f"Recall@10: {metrics_user['recall_at_k']:.4f}")

    # 2. Item-based Collaborative Filtering
    print("\n2. Item-based Collaborative Filtering")
    print("-" * 70)
    item_cf_predictions = recommender.item_based_collaborative_filtering(k_neighbors=10)
    rmse_item = recommender.evaluate_rmse(test_matrix, item_cf_predictions)
    mae_item = recommender.evaluate_mae(test_matrix, item_cf_predictions)
    print(f"RMSE: {rmse_item:.4f}")
    print(f"MAE: {mae_item:.4f}")

    metrics_item = recommender.evaluate_precision_recall_at_k(test_matrix, item_cf_predictions, k=10)
    print(f"Precision@10: {metrics_item['precision_at_k']:.4f}")
    print(f"Recall@10: {metrics_item['recall_at_k']:.4f}")

    # 3. Matrix Factorization (SVD)
    print("\n3. Matrix Factorization (SVD)")
    print("-" * 70)
    svd_result = recommender.matrix_factorization_svd(n_factors=20)
    svd_predictions = svd_result['predictions']
    rmse_svd = recommender.evaluate_rmse(test_matrix, svd_predictions)
    mae_svd = recommender.evaluate_mae(test_matrix, svd_predictions)
    print(f"Number of factors: {svd_result['n_factors']}")
    print(f"RMSE: {rmse_svd:.4f}")
    print(f"MAE: {mae_svd:.4f}")

    metrics_svd = recommender.evaluate_precision_recall_at_k(test_matrix, svd_predictions, k=10)
    print(f"Precision@10: {metrics_svd['precision_at_k']:.4f}")
    print(f"Recall@10: {metrics_svd['recall_at_k']:.4f}")

    # 4. Content-based Filtering
    print("\n4. Content-based Filtering")
    print("-" * 70)
    # Generate random item features
    n_features = 10
    item_features = np.random.randn(n_items, n_features)
    content_predictions = recommender.content_based_filtering(item_features)
    rmse_content = recommender.evaluate_rmse(test_matrix, content_predictions)
    mae_content = recommender.evaluate_mae(test_matrix, content_predictions)
    print(f"Item features: {n_features}")
    print(f"RMSE: {rmse_content:.4f}")
    print(f"MAE: {mae_content:.4f}")

    # 5. Hybrid Recommender
    print("\n5. Hybrid Recommender (Collaborative + Content-based)")
    print("-" * 70)
    hybrid_predictions = recommender.hybrid_recommender(
        item_cf_predictions,
        content_predictions,
        alpha=0.7
    )
    rmse_hybrid = recommender.evaluate_rmse(test_matrix, hybrid_predictions)
    mae_hybrid = recommender.evaluate_mae(test_matrix, hybrid_predictions)
    print(f"Alpha (collaborative weight): 0.7")
    print(f"RMSE: {rmse_hybrid:.4f}")
    print(f"MAE: {mae_hybrid:.4f}")

    metrics_hybrid = recommender.evaluate_precision_recall_at_k(test_matrix, hybrid_predictions, k=10)
    print(f"Precision@10: {metrics_hybrid['precision_at_k']:.4f}")
    print(f"Recall@10: {metrics_hybrid['recall_at_k']:.4f}")

    # 6. Top-N Recommendations
    print("\n6. Top-N Recommendations for User 0")
    print("-" * 70)
    recommendations = recommender.top_n_recommendations(user_idx=0, n=10, predictions=svd_predictions)
    print(f"{'Rank':<6} {'Item ID':<10} {'Predicted Rating':<20}")
    print("-" * 70)
    for rank, (item_id, rating) in enumerate(recommendations, 1):
        print(f"{rank:<6} {item_id:<10} {rating:<20.4f}")

    # 7. Cold Start Handling
    print("\n7. Cold Start Handling")
    print("-" * 70)

    # New user with features
    new_user_features = np.random.randn(n_features)
    existing_user_features = np.random.randn(n_users, n_features)
    cold_user_predictions = recommender.handle_cold_start_user(
        new_user_features,
        existing_user_features,
        k_neighbors=5
    )
    print(f"Cold start user - Top 5 predictions: {np.sort(cold_user_predictions)[::-1][:5]}")

    # New item with features
    new_item_features = np.random.randn(n_features)
    cold_item_predictions = recommender.handle_cold_start_item(
        new_item_features,
        item_features,
        k_neighbors=5
    )
    print(f"Cold start item - Average predicted rating: {np.mean(cold_item_predictions[cold_item_predictions > 0]):.4f}")

    # Visualizations
    print("\n8. Generating Visualizations")
    print("-" * 70)

    # User-Item Matrix
    fig1 = recommender.visualize_user_item_matrix(title='Training User-Item Matrix (First 50x50)')
    fig1.savefig('recommender_user_item_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Saved recommender_user_item_matrix.png")
    plt.close()

    # Prediction Errors
    fig2 = recommender.visualize_prediction_errors(test_matrix, svd_predictions)
    fig2.savefig('recommender_prediction_errors.png', dpi=300, bbox_inches='tight')
    print("✓ Saved recommender_prediction_errors.png")
    plt.close()

    # Performance Comparison
    print("\n9. Performance Summary")
    print("-" * 70)
    print(f"{'Method':<30} {'RMSE':<12} {'MAE':<12} {'Precision@10':<15} {'Recall@10':<15}")
    print("-" * 70)
    print(f"{'User-based CF':<30} {rmse_user:<12.4f} {mae_user:<12.4f} {metrics_user['precision_at_k']:<15.4f} {metrics_user['recall_at_k']:<15.4f}")
    print(f"{'Item-based CF':<30} {rmse_item:<12.4f} {mae_item:<12.4f} {metrics_item['precision_at_k']:<15.4f} {metrics_item['recall_at_k']:<15.4f}")
    print(f"{'Matrix Factorization (SVD)':<30} {rmse_svd:<12.4f} {mae_svd:<12.4f} {metrics_svd['precision_at_k']:<15.4f} {metrics_svd['recall_at_k']:<15.4f}")
    print(f"{'Content-based':<30} {rmse_content:<12.4f} {mae_content:<12.4f}")
    print(f"{'Hybrid':<30} {rmse_hybrid:<12.4f} {mae_hybrid:<12.4f} {metrics_hybrid['precision_at_k']:<15.4f} {metrics_hybrid['recall_at_k']:<15.4f}")

    print("\n" + "=" * 70)
    print("✓ Recommender Systems Demo Complete!")
    print("=" * 70)


if __name__ == '__main__':
    demo()
