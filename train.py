
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import glob
import numpy as np
from tqdm import tqdm
import os
from model import DeepfakeDetector
from dataset import DeepfakeDataset, get_train_transforms, get_val_transforms


DATA_DIR = 'S:\Deepfake project stuff\processed_data\pde_features'
NUM_FRAMES = 20 
IMAGE_SIZE = 224
BATCH_SIZE = 8
ACCUMULATION_STEPS = 4 # accumulate gradients over 4 batches
EFFECTIVE_BATCH_SIZE = BATCH_SIZE * ACCUMULATION_STEPS
EPOCHS = 10
LEARNING_RATE = 1e-4 
NUM_WORKERS = 4 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    train_real_files = glob.glob(os.path.join(DATA_DIR, 'train\\real\\*.npz'))
    train_fake_files = glob.glob(os.path.join(DATA_DIR, 'train\\fake\\*.npz'))

    all_train_files = train_real_files + train_fake_files
    all_train_labels = [0] * len(train_real_files) + [1] * len(train_fake_files)
    
    train_files, val_files, train_labels, val_labels = train_test_split(
        all_train_files, all_train_labels, test_size=0.2, random_state=42, stratify=all_train_labels
    )
    
    test_real_files = glob.glob(os.path.join(DATA_DIR, 'test/real/*.npz'))
    test_fake_files = glob.glob(os.path.join(DATA_DIR, 'test/fake/*.npz'))
    test_files = test_real_files + test_fake_files
    test_labels = [0] * len(test_real_files) + [1] * len(test_fake_files)
    
    train_dataset = DeepfakeDataset(train_files, train_labels, transform=get_train_transforms(IMAGE_SIZE))
    val_dataset = DeepfakeDataset(val_files, val_labels, transform=get_val_transforms(IMAGE_SIZE))
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True if NUM_WORKERS > 0 else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    
    model = DeepfakeDetector().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    scaler = torch.amp.GradScaler()

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE,
        steps_per_epoch=len(train_loader) // ACCUMULATION_STEPS,
        epochs=EPOCHS
    )

    print(f"Starting training on {DEVICE}")
    print(f"Physical Batch Size: {BATCH_SIZE}, Accumulation Steps: {ACCUMULATION_STEPS}, Effective Batch Size: {EFFECTIVE_BATCH_SIZE}")

    best_val_auc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad() 

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{EPOCHS}")
        for i, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                loss = loss / ACCUMULATION_STEPS

            scaler.scale(loss).backward()
            
            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            train_loss += loss.item() * ACCUMULATION_STEPS 
            progress_bar.set_postfix(loss=f"{train_loss/(i+1):.4f}")

        val_loss, all_preds, all_labels_val = evaluate(model, val_loader, criterion, DEVICE)
        
        val_accuracy = accuracy_score(all_labels_val, all_preds > 0.5)
        val_auc = roc_auc_score(all_labels_val, all_preds)
        val_precision = precision_score(all_labels_val, all_preds > 0.5, zero_division=0)
        val_recall = recall_score(all_labels_val, all_preds > 0.5, zero_division=0)
        val_f1 = f1_score(all_labels_val, all_preds > 0.5, zero_division=0)

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f} | Val AUC: {val_auc:.4f}")
        print(f"  Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1-Score: {val_f1:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  -> New best model saved with AUC: {best_val_auc:.4f}")

    print("\n--- Training Finished ---")
    print(f"Best validation AUC: {best_val_auc:.4f}")

    print("\nLoading best model for final evaluation on the test set...")
    model.load_state_dict(torch.load('best_model.pth'))

    test_dataset = DeepfakeDataset(test_files, test_labels, transform=get_val_transforms(IMAGE_SIZE))
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    print("Running evaluation on the test set...")
    test_loss, all_preds_test, all_labels_test = evaluate(model, test_loader, criterion, DEVICE)

    test_accuracy = accuracy_score(all_labels_test, all_preds_test > 0.5)
    test_auc = roc_auc_score(all_labels_test, all_preds_test)
    test_precision = precision_score(all_labels_test, all_preds_test > 0.5, zero_division=0)
    test_recall = recall_score(all_labels_test, all_preds_test > 0.5, zero_division=0)
    test_f1 = f1_score(all_labels_test, all_preds_test > 0.5, zero_division=0)

    print(f"\n--- Test Set Results (Final Evaluation) ---")
    print(f"  Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy:.4f} | Test AUC: {test_auc:.4f}")
    print(f"  Precision: {test_precision:.4f} | Recall: {test_recall:.4f} | F1-Score: {test_f1:.4f}")


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds =[]
    all_labels =[]
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return running_loss / len(loader), np.array(all_preds), np.array(all_labels)

if __name__ == '__main__':
    main()