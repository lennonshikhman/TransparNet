# decision_tree_student.py

import numpy as np
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from config import PCA_VARIANCE_PERCENT, random_seed


def apply_pca(train_feats, val_feats=None, test_feats=None):
    """
    Reduces dimensionality using PCA while preserving a specified variance percentage.
    """
    pca = PCA(n_components=PCA_VARIANCE_PERCENT, random_state=random_seed)
    train_reduced = pca.fit_transform(train_feats)
    val_reduced = pca.transform(val_feats) if val_feats is not None else None
    test_reduced = pca.transform(test_feats) if test_feats is not None else None
    return train_reduced, val_reduced, test_reduced, pca


def train_decision_tree(features, teacher_preds):
    """
    Performs grid search to train and tune a decision tree classifier.
    """
    param_grid = {
        'max_depth': [8],
        'min_samples_leaf': [1],
        'min_samples_split': [2],
        'ccp_alpha': [0]
    }

    dt = DecisionTreeClassifier(random_state=random_seed)
    grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(features, teacher_preds)
    print(f"Best Decision Tree Params: {grid_search.best_params_}")
    return grid_search.best_estimator_


def evaluate_student(student_model, features_test, labels_test, teacher_preds_test):
    """
    Computes and prints student accuracy and fidelity.
    """
    student_preds = student_model.predict(features_test)
    student_acc = accuracy_score(labels_test, student_preds) * 100
    fidelity = accuracy_score(teacher_preds_test, student_preds) * 100
    print(f"Student Accuracy on Test Set: {student_acc:.2f}%")
    print(f"Fidelity to Teacher Predictions: {fidelity:.2f}%")
    return student_preds, student_acc, fidelity
