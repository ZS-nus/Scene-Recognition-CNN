import sys
import os
import time  # Add this import at the top with your other imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from model import ImprovedCNN, get_device


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    Stops training when validation loss doesn't improve for a specified patience.
    
    Args:
        patience (int): How many epochs to wait after last improvement.
        min_delta (float): Minimum change to qualify as improvement.
        verbose (bool): If True, prints message when early stopping occurs.
    """
    def __init__(self, patience=5, min_delta=0.0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            # Improvement found
            self.best_loss = val_loss
            self.counter = 0
        else:
            # No improvement
            self.counter += 1
            if self.verbose:
                print(f"\n EarlyStopping: No improvement for {self.counter} epochs.")
            
            if self.counter >= self.patience:
                if self.verbose:
                    print(f"\n EarlyStopping: Stopping training after {self.patience} epochs without improvement.")
                self.early_stop = True
                
        return self.early_stop


def print_progress_bar(
    iteration,
    total,
    prefix='',
    suffix='',
    decimals=1,
    length=50,
    fill='█'
):
    """
    Prints a custom progress bar to the console, overwriting the same line.

    Args:
        iteration (int): current iteration (e.g. current batch index + 1)
        total (int): total iterations (e.g. total number of batches)
        prefix (str): text to put before the bar
        suffix (str): text to put after the bar
        decimals (int): how many decimals for the percentage
        length (int): the length of the progress bar in characters
        fill (str): the character to fill in the bar, e.g. '█' or '#'
    """
    percent = 100 * (iteration / float(total))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    # \r goes back to the beginning of the line so we can overwrite
    sys.stdout.write(f'\r{prefix} |{bar}| {percent:.{decimals}f}% {suffix}')
    sys.stdout.flush()
    # Once we reach total, print a newline
    if iteration == total:
        sys.stdout.write('\n')
        


def train(train_data_dir="./train", **kwargs):
    """
    Train a CNN model for scene classification with a custom progress bar.

    Args:
        train_data_dir (str): Path to images (in subfolders by class).
            Default = "./train"
        **kwargs options:
            batch_size (int): Batch size. Default=16
            lr (float): Learning rate. Default=0.001
            epochs (int): Number of epochs. Default=5
            val_split (float): Fraction for validation. Default=0.2
            model_save_path (str): Where to save the model. Default='trained_cnn.pth'
            early_stopping (bool): Whether to use early stopping. Default=False
            patience (int): Patience for early stopping. Default=5
    """
    # Get the best available device
    device = get_device()
    
    # 1) Hyperparameters & defaults
    batch_size = kwargs.get('batch_size', 16)
    lr = kwargs.get('lr', 0.001)
    epochs = kwargs.get('epochs', 5)
    val_split = kwargs.get('val_split', 0.2)  # e.g., 20% of images => val set
    model_save_path = kwargs.get('model_save_path', 'trained_cnn.pth')
    
    # Early stopping parameters
    use_early_stopping = kwargs.get('early_stopping', False)
    patience = kwargs.get('patience', 5)
    
    # Initialize early stopping if enabled
    early_stopper = None
    if use_early_stopping:
        early_stopper = EarlyStopping(patience=patience)
        print(f"Early stopping enabled with patience {patience} \n")
    
    # 2) Define transforms with augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize larger than needed for random crop
        transforms.RandomCrop(224),     # Random crop to 224x224
        transforms.RandomHorizontalFlip(p=0.5),  # 50% chance of horizontal flip
        transforms.RandomRotation(10),  # Small random rotations (+/- 10 degrees)
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),  # Color variations
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Define separate transforms for validation (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Define non-augmented transform for comparison
    basic_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Define the mapping (subtract 1 for 0-based indexing)
    class_to_official = {
        'bedroom': 1,
        'Coast': 2,
        'Forest': 3,
        'Highway': 4,
        'industrial': 5,
        'Insidecity': 6,
        'kitchen': 7,
        'livingroom': 8,
        'Mountain': 9,
        'Office': 10,
        'OpenCountry': 11,
        'store': 12,
        'Street': 13,
        'Suburb': 14,
        'TallBuilding': 15
    }
    
    # Load the dataset with the custom class mapping
    full_dataset = datasets.ImageFolder(
        root=train_data_dir, 
        transform=train_transform,  # Use augmentation transform here
        target_transform=lambda x: list(class_to_official.values()).index(x) if x in class_to_official.values() else x
    )

    # Update the class_to_idx mapping to use our custom indices
    full_dataset.class_to_idx = class_to_official
    
    dataset_size = len(full_dataset)

    # 4) Split into training & validation sets
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # reproducible
    )

    # After creating train_dataset and val_dataset, we need to update their transforms
    train_dataset.dataset.transform = train_transform  # Apply augmentation to training
    val_dataset.dataset.transform = val_transform      # No augmentation for validation

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Calculate effective augmented dataset size by running a few batches through both transforms
    def estimate_augmented_size(original_dataset, augmented_transform, basic_transform, samples=100):
        """Estimate the effective size increase from data augmentation"""
        import numpy as np
        
        if samples > len(original_dataset):
            samples = len(original_dataset)
            
        # Select random samples from dataset
        indices = np.random.choice(len(original_dataset), samples, replace=False)
        
        total_pixel_diff = 0
        for idx in indices:
            img, _ = original_dataset[idx]
            
            # Save current transform
            current_transform = original_dataset.transform
            
            # Apply basic transform
            original_dataset.transform = basic_transform
            basic_img, _ = original_dataset[idx]
            
            # Apply augmented transform
            original_dataset.transform = augmented_transform
            aug_img, _ = original_dataset[idx]
            
            # Restore original transform
            original_dataset.transform = current_transform
            
            # Calculate difference between augmented and basic images
            diff = torch.abs(aug_img - basic_img).sum().item()
            total_pixel_diff += diff
        
        # Calculate average difference
        avg_diff = total_pixel_diff / samples
        
        # Normalize to a percentage (higher diff = more augmentation)
        # This is a rough estimate of "how different" the augmented images are
        aug_effect = min(100, avg_diff * 10)  # Cap at 100% increase
        
        return aug_effect
    
    # Estimate augmentation effect
    aug_effect = estimate_augmented_size(full_dataset, train_transform, basic_transform)
    
    # Calculate effective training examples
    effective_train_size = int(train_size * (1 + aug_effect/100))
    
    print(f"\n Total images: {dataset_size} | Original training: {train_size}, Validation: {val_size}")
    print(f"\n Data augmentation effectiveness: {aug_effect:.1f}% increase")
    print(f"\n Effective training samples: ~{effective_train_size} (including augmentations) \n")
    
    print(full_dataset.class_to_idx)

    # 5) Create model, define loss & optimizer
    model = ImprovedCNN(num_classes=15).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Add to your train function before the training loop
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # 6) Training loop
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        epoch_start_time = time.time()  # Start timing the epoch
        model.train()
        running_loss = 0.0

        total_batches = len(train_loader)
        print(f"\nEpoch [{epoch+1}/{epochs}]")

        # For each batch, show a progress bar.
        for batch_idx, (images, labels) in enumerate(train_loader):
            # Move tensors to device
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()

            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Update our text-based progress bar
            # batch_idx+1 => so it starts counting from 1
            print_progress_bar(
                iteration=batch_idx + 1,
                total=total_batches,
                prefix="Training",
                suffix=f"Loss: {loss.item():.4f}",
                length=30  # length of the progress bar in characters
            )

        avg_train_loss = running_loss / total_batches

        # ========== Validation ==========
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                # Move tensors to device
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100.0 * correct / total
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        minutes = int(epoch_time // 60)
        seconds = int(epoch_time % 60)

        # Print summary for the epoch with timing information
        print(f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_accuracy:.2f}% | "
              f"Time: {minutes}m {seconds}s")
        
        # Inside training loop, after validation
        scheduler.step(avg_val_loss)
        
        # Check early stopping condition
        if use_early_stopping and early_stopper(avg_val_loss):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Save the best model instead of the last one
    if best_model_state is not None:
        torch.save(best_model_state, model_save_path)
        print(f"Training complete. Best model saved to: {model_save_path}")
    else:
        # Fallback to the last model if somehow no best model was found
        torch.save(model.state_dict(), model_save_path)
        print(f"Training complete. Final model saved to: {model_save_path}")


###############################################################################
# 4) test() function: if/when you have a separate test set.
###############################################################################
def test(test_data_dir, trained_cnn_path="trained_cnn.pth", **kwargs):
    """
    Evaluate the trained CNN on a separate test dataset.

    Args:
        test_data_dir (str): Directory of test images, subfolders by class.
        trained_cnn_path (str): path to the saved model file. Default='trained_cnn.pth'
        **kwargs:
            batch_size (int): default=16
    """
    batch_size = kwargs.get('batch_size', 16)
    
    # Get the best available device
    device = get_device()

    # Use the same transform as validation (no augmentation)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Define the mapping again
    class_to_official = {
        'bedroom': 1,
        'Coast': 2,
        'Forest': 3,
        'Highway': 4,
        'industrial': 5,
        'Insidecity': 6,
        'kitchen': 7,
        'livingroom': 8,
        'Mountain': 9,
        'Office': 10,
        'OpenCountry': 11,
        'store': 12,
        'Street': 13,
        'Suburb': 14,
        'TallBuilding': 15
    }
    
    # Load the dataset with the custom class mapping
    test_dataset = datasets.ImageFolder(
        root=test_data_dir, 
        transform=transform,
        target_transform=lambda x: list(class_to_official.values()).index(x) if x in class_to_official.values() else x
    )
    
    # Update the class_to_idx mapping
    test_dataset.class_to_idx = class_to_official
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Rebuild same architecture & load weights
    model = ImprovedCNN(num_classes=15).to(device)
    model.load_state_dict(torch.load(trained_cnn_path, weights_only=True))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            # Move tensors to device
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    print(f"\n Test Accuracy: {accuracy:.2f}%")
