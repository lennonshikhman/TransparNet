# app.py

import torch
from data.prepare_data import create_dataloaders
from models.resnet_teacher import build_resnet50_teacher, train_teacher
from models.feature_extraction import get_feature_extractor, extract_features
from models.decision_tree_student import apply_pca, train_decision_tree, evaluate_student
from visualize.decision_tree_plot import plot_decision_tree, get_used_features
from visualize.confusion import plot_confusion_matrix, compute_kappa
from visualize.tsne import plot_tsne
from utils.helpers import format_elapsed_time
from config import device, IMAGENETTE_CLASSES, TREE_VISUALIZATION_PATH
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    total_start = time.time()

    # 1. Load Data
    print("\n>>> Loading and preparing data...")
    train_loader, val_loader, test_loader, dataset_test = create_dataloaders()

    # 2. Build & Train Teacher
    print("\n>>> Building and training teacher model...")
    teacher = build_resnet50_teacher()
    teacher = train_teacher(teacher, train_loader, val_loader, device)

    # 3. Feature Extraction
    print("\n>>> Extracting features...")
    extractor = get_feature_extractor(teacher)
    feats_train, logits_train = extract_features(extractor, train_loader, teacher, device)
    feats_val, logits_val = extract_features(extractor, val_loader, teacher, device)
    feats_test, logits_test = extract_features(extractor, test_loader, teacher, device)

    # 4. Train Student Model
    print("\n>>> Training decision tree student model...")
    feats_train_pca, feats_val_pca, feats_test_pca, pca_model = apply_pca(feats_train, feats_val, feats_test)
    teacher_preds_train = logits_train.argmax(axis=1)
    teacher_preds_test = logits_test.argmax(axis=1)

    student = train_decision_tree(feats_train_pca, teacher_preds_train)

    # 5. Evaluation
    print("\n>>> Evaluating student model...")
    y_true_test = [label for _, label in dataset_test]
    y_pred_test, student_acc, fidelity = evaluate_student(student, feats_test_pca, y_true_test, teacher_preds_test)

    print("\n>>> Confusion Matrix and Kappa Score")
    plot_confusion_matrix(y_true_test, y_pred_test, class_names=IMAGENETTE_CLASSES)
    kappa = compute_kappa(y_true_test, y_pred_test)
    print(f"Cohen's Kappa Score: {kappa:.4f}")

    # 6. Visualize Tree
    print("\n>>> Visualizing decision tree...")
    used_feats = get_used_features(student)
    feature_names = [f"PC {i}" for i in range(student.n_features_in_)]
    plot_decision_tree(student, feature_names, IMAGENETTE_CLASSES, save_path=str(TREE_VISUALIZATION_PATH))

    # 7. t-SNE
    print("\n>>> Plotting t-SNE projection of features...")
    plot_tsne(feats_test_pca, y_true_test, class_names=IMAGENETTE_CLASSES)

    print("\n✅ Pipeline complete. Total time:", format_elapsed_time(time.time() - total_start))


if __name__ == '__main__':
    main()