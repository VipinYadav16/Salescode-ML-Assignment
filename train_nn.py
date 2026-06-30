import os
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from dataset_loader import load_image_paths_and_labels
from utils import get_logger, set_global_seed

LOGGER = get_logger(__name__)
SEED = 42
ARTIFACT_DIR = Path("artifacts")
MODEL_ONNX_PATH = ARTIFACT_DIR / "mobilenet_recapture.onnx"
METADATA_PATH = ARTIFACT_DIR / "mobilenet_metadata.json"
NUM_FOLDS = 5
EPOCHS = 15
BATCH_SIZE = 8
LR = 1e-3

class RecaptureDataset(Dataset):
    """Custom Dataset for loading images from paths list."""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image in RGB
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.float32)

def build_model():
    """Build MobileNetV2 with frozen backbone and custom classification head."""
    # Load pre-trained MobileNetV2
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)
    
    # Freeze feature extractor layers
    for param in model.features.parameters():
        param.requires_grad = False
        
    # Replace classifier head for binary classification
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 1) # Raw logit output
    )
    return model

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images).squeeze(1)
            scores = torch.sigmoid(outputs)
            preds = (scores >= 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_scores.extend(scores.cpu().numpy())
            
    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_scores)
    )

def main():
    set_global_seed(SEED)
    ARTIFACT_DIR.mkdir(exist_ok=True)
    
    # Load dataset
    image_paths, labels = load_image_paths_and_labels("dataset")
    image_paths = np.array(image_paths)
    labels = np.array(labels)
    LOGGER.info("Loaded %d images for training.", len(image_paths))

    # Define transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    # 5-Fold Stratified Cross-Validation
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    
    oof_labels = np.zeros(len(labels))
    oof_preds = np.zeros(len(labels))
    oof_scores = np.zeros(len(labels))
    
    fold_metrics = []
    best_overall_f1 = 0.0
    best_model_state = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(image_paths, labels), 1):
        LOGGER.info("--- Training Fold %d/%d ---", fold, NUM_FOLDS)
        
        train_sub = RecaptureDataset(image_paths[train_idx], labels[train_idx], transform=train_transform)
        val_sub = RecaptureDataset(image_paths[val_idx], labels[val_idx], transform=val_transform)
        
        train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=False)
        
        model = build_model().to(device)
        criterion = nn.BCEWithLogitsLoss()
        
        # Optimize only classifier parameters
        optimizer = optim.Adam(model.classifier.parameters(), lr=LR, weight_decay=1e-4)
        
        best_fold_f1 = 0.0
        best_fold_state = None
        
        for epoch in range(1, EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_true, val_pred, val_score = evaluate_model(model, val_loader, device)
            
            val_acc = accuracy_score(val_true, val_pred)
            val_f1 = f1_score(val_true, val_pred, zero_division=0)
            
            if val_f1 > best_fold_f1:
                best_fold_f1 = val_f1
                best_fold_state = {k: v.cpu() for k, v in model.state_dict().items()}
                
            # Log progress every 5 epochs
            if epoch % 5 == 0 or epoch == EPOCHS:
                LOGGER.info("Epoch %d/%d | Train Loss: %.4f | Val Acc: %.4f | Val F1: %.4f", 
                            epoch, EPOCHS, train_loss, val_acc, val_f1)
                
        # Load best fold weights
        model.load_state_dict({k: v.to(device) for k, v in best_fold_state.items()})
        val_true, val_pred, val_score = evaluate_model(model, val_loader, device)
        
        oof_labels[val_idx] = val_true
        oof_preds[val_idx] = val_pred
        oof_scores[val_idx] = val_score
        
        fold_acc = accuracy_score(val_true, val_pred)
        fold_f1 = f1_score(val_true, val_pred, zero_division=0)
        fold_auc = roc_auc_score(val_true, val_score) if len(np.unique(val_true)) > 1 else 1.0
        
        fold_metrics.append({
            "fold": fold,
            "accuracy": fold_acc,
            "f1": fold_f1,
            "auc": fold_auc
        })
        LOGGER.info("Fold %d Best Metrics -> Acc: %.4f | F1: %.4f | AUC: %.4f", fold, fold_acc, fold_f1, fold_auc)
        
        # Save overall best state dict
        if fold_f1 > best_overall_f1:
            best_overall_f1 = fold_f1
            best_model_state = best_fold_state

    # Overall CV Evaluation
    cv_accuracy = accuracy_score(oof_labels, oof_preds)
    cv_precision = precision_score(oof_labels, oof_preds, zero_division=0)
    cv_recall = recall_score(oof_labels, oof_preds, zero_division=0)
    cv_f1 = f1_score(oof_labels, oof_preds, zero_division=0)
    cv_auc = roc_auc_score(oof_labels, oof_scores)
    
    LOGGER.info("====================================")
    LOGGER.info("OVERALL 5-FOLD CV METRICS:")
    LOGGER.info("Accuracy : %.4f", cv_accuracy)
    LOGGER.info("Precision: %.4f", cv_precision)
    LOGGER.info("Recall   : %.4f", cv_recall)
    LOGGER.info("F1 Score : %.4f", cv_f1)
    LOGGER.info("ROC AUC  : %.4f", cv_auc)
    LOGGER.info("====================================")

    # Train a final model on 100% of data to maximize accuracy, or export the best CV model.
    # Training a final model on all data is usually preferred. Let's do that for 12 epochs.
    LOGGER.info("Training final model on all data...")
    final_model = build_model().to(device)
    full_dataset = RecaptureDataset(image_paths, labels, transform=train_transform)
    full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(final_model.classifier.parameters(), lr=LR, weight_decay=1e-4)
    
    for epoch in range(1, 13):
        loss = train_one_epoch(final_model, full_loader, criterion, optimizer, device)
        if epoch % 4 == 0 or epoch == 12:
            LOGGER.info("Final Epoch %d/12 | Loss: %.4f", epoch, loss)

    # Export final model to ONNX
    final_model.eval()
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    
    LOGGER.info("Exporting model to ONNX: %s", MODEL_ONNX_PATH)
    torch.onnx.export(
        final_model,
        dummy_input,
        str(MODEL_ONNX_PATH),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )

    # Save metadata
    metadata = {
        "model_type": "MobileNetV2",
        "input_size": [224, 224],
        "cv_metrics": {
            "accuracy": cv_accuracy,
            "precision": cv_precision,
            "recall": cv_recall,
            "f1": cv_f1,
            "auc": cv_auc
        },
        "fold_metrics": fold_metrics
    }
    
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)
        
    LOGGER.info("Saved metadata to %s", METADATA_PATH)

if __name__ == "__main__":
    main()
