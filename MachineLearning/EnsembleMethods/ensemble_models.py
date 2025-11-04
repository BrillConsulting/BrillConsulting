"""
Ensemble Methods
Author: BrillConsulting
Description: Bagging, Boosting, and Stacking ensembles
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier,
                              AdaBoostClassifier, GradientBoostingClassifier,
                              VotingClassifier, StackingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import argparse


class EnsembleAnalyzer:
    """Ensemble methods system"""

    def __init__(self):
        self.models = {}
        self.results = {}

    def bagging_ensemble(self, X_train, y_train, X_test, y_test):
        """Bagging with Decision Trees"""
        print("🔧 Bagging Ensemble...")

        model = BaggingClassifier(
            estimator=DecisionTreeClassifier(),
            n_estimators=50,
            random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        self.models['Bagging'] = model
        self.results['Bagging'] = {'accuracy': accuracy}

        print(f"  Accuracy: {accuracy:.4f}\n")
        return model

    def boosting_ensemble(self, X_train, y_train, X_test, y_test):
        """Gradient Boosting"""
        print("🔧 Gradient Boosting...")

        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        self.models['Boosting'] = model
        self.results['Boosting'] = {'accuracy': accuracy}

        print(f"  Accuracy: {accuracy:.4f}\n")
        return model

    def voting_ensemble(self, X_train, y_train, X_test, y_test):
        """Voting Classifier"""
        print("🔧 Voting Ensemble...")

        estimators = [
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
            ('lr', LogisticRegression(max_iter=1000, random_state=42))
        ]

        model = VotingClassifier(estimators=estimators, voting='soft')
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        self.models['Voting'] = model
        self.results['Voting'] = {'accuracy': accuracy}

        print(f"  Accuracy: {accuracy:.4f}\n")
        return model

    def stacking_ensemble(self, X_train, y_train, X_test, y_test):
        """Stacking Classifier"""
        print("🔧 Stacking Ensemble...")

        estimators = [
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42))
        ]

        model = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000),
            cv=5
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        self.models['Stacking'] = model
        self.results['Stacking'] = {'accuracy': accuracy}

        print(f"  Accuracy: {accuracy:.4f}\n")
        return model

    def compare_results(self):
        """Compare all ensemble methods"""
        comparison = pd.DataFrame([
            {'Model': name, 'Accuracy': results['accuracy']}
            for name, results in self.results.items()
        ])

        comparison = comparison.sort_values('Accuracy', ascending=False)
        return comparison


def main():
    parser = argparse.ArgumentParser(description='Ensemble Methods')
    parser.add_argument('--data', type=str, required=True, help='CSV data')
    parser.add_argument('--target', type=str, required=True, help='Target column')

    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.data)
    X = df.drop(columns=[args.target])
    y = df[args.target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Run ensembles
    analyzer = EnsembleAnalyzer()
    analyzer.bagging_ensemble(X_train, y_train, X_test, y_test)
    analyzer.boosting_ensemble(X_train, y_train, X_test, y_test)
    analyzer.voting_ensemble(X_train, y_train, X_test, y_test)
    analyzer.stacking_ensemble(X_train, y_train, X_test, y_test)

    # Compare
    print("📊 Ensemble Comparison:")
    print("=" * 40)
    comparison = analyzer.compare_results()
    print(comparison.to_string(index=False))
    print("=" * 40)


if __name__ == "__main__":
    main()
