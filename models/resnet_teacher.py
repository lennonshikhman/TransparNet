# resnet_teacher.py

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import accuracy_score
from config import NUM_CLASSES, INITIAL_LR, WARMUP_EPOCHS, NUM_EPOCHS, LABEL_SMOOTHING, MOMENTUM, WEIGHT_DECAY


def build_resnet50_teacher():
    """
    Loads a pretrained ResNet50 model, freezes early layers, and modifies the classifier for transfer learning.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze deeper layers
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace the classifier head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, NUM_CLASSES)
    )
    return model


def get_lr(epoch):
    """
    Computes the learning rate using linear warmup followed by cosine annealing.
    """
    if epoch < WARMUP_EPOCHS:
        return INITIAL_LR * (epoch + 1) / WARMUP_EPOCHS
    return INITIAL_LR * 0.5 * (1 + math.cos(math.pi * (epoch - WARMUP_EPOCHS) / (NUM_EPOCHS - WARMUP_EPOCHS)))


def train_teacher(model, train_loader, val_loader, device):
    """
    Trains the teacher model using transfer learning and evaluates on validation set each epoch.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.SGD(model.parameters(), lr=INITIAL_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler()

    model.train()
    for epoch in range(NUM_EPOCHS):
        lr = get_lr(epoch)
        for g in optimizer.param_groups:
            g['lr'] = lr

        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images.to(device))
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        val_acc = accuracy_score(all_labels, all_preds) * 100
        print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | LR: {lr:.5f} | Loss: {running_loss:.4f} | Val Acc: {val_acc:.2f}%")
        model.train()

    return model
