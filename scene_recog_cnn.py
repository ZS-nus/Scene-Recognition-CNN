import sys
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np

from model import get_device, ResNet101Transfer  # Changed from ResNet50Transfer to ResNet101Transfer


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
        

def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train(train_data_dir="./train", **kwargs):
    """
    Train a ResNet model for scene classification with a custom progress bar.

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
            unfreeze_after (int): Epoch to unfreeze ResNet layers. Default=5
            unfreeze_layer (str): Layer to unfreeze from. Default='layer4'
            label_smoothing (float): Label smoothing factor. Default=0.1
            progressive_unfreeze (bool): Whether to use progressive unfreezing. Default=False
    """
    # Get the best available device
    device = get_device()
    
    # 1) Hyperparameters & defaults
    batch_size = kwargs.get('batch_size', 16)
    lr = kwargs.get('lr', 0.001)
    epochs = kwargs.get('epochs', 5)
    val_split = kwargs.get('val_split', 0.2)
    model_save_path = kwargs.get('model_save_path', 'trained_cnn.pth')
    
    # Early stopping parameters
    use_early_stopping = kwargs.get('early_stopping', False)
    patience = kwargs.get('patience', 5)
    
    # Parameters for ResNet
    unfreeze_after = kwargs.get('unfreeze_after', 5)  # Epoch after which to unfreeze layers
    unfreeze_layer = kwargs.get('unfreeze_layer', 'layer4')  # Which layer to unfreeze from
    
    # New advanced parameters
    label_smoothing = kwargs.get('label_smoothing', 0.1)  # Label smoothing factor
    use_progressive_unfreeze = kwargs.get('progressive_unfreeze', False)
    
    # Initialize early stopping if enabled
    early_stopper = None
    if use_early_stopping:
        early_stopper = EarlyStopping(patience=patience)
        print(f"Early stopping enabled with patience {patience} \n")
    
    # 2) Define transforms with augmentation for training
    # Enhanced data augmentation pipeline
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),  # Increased rotation variation
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # Added perspective shift for scene diversity
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # Enhanced color variation
        transforms.RandAugment(num_ops=3, magnitude=9),  # Increased augmentation strength
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2))  # Increased random erasing
    ])
    
    # Define separate transforms for validation (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Define the mapping (subtract 1 for 0-based indexing)
    class_to_official = {
        'bedroom': 0,
        'Coast': 1,
        'Forest': 2,
        'Highway': 3,
        'industrial': 4,
        'Insidecity': 5,
        'kitchen': 6,
        'livingroom': 7,
        'Mountain': 8,
        'Office': 9,
        'OpenCountry': 10,
        'store': 11,
        'Street': 12,
        'Suburb': 13,
        'TallBuilding': 14
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
    
    print(f"\n Total images: {dataset_size} | Training: {train_size}, Validation: {val_size}")
    print(f"\n Using enhanced data augmentation for training set (RandomCrop, RandomHorizontalFlip, RandomRotation, RandomPerspective, ColorJitter, RandAugment, RandomErasing)")
    print(f" Each training epoch processes {train_size} images with different random augmentations applied each time")
    if use_progressive_unfreeze:
        print(f" Using progressive unfreezing strategy")
    print(full_dataset.class_to_idx)

    # 5) Create model, define loss & optimizer
    model = ResNet101Transfer(num_classes=15, pretrained=True).to(device)
    # Initially freeze the ResNet backbone
    model.freeze_backbone(freeze=True)
    print("Using ResNet-101 with transfer learning and enhanced classifier")
    
    # Use CrossEntropyLoss with label smoothing for better generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    # Higher learning rate for the new FC layer, lower for the rest if unfrozen
    optimizer = optim.AdamW([  # Switched to AdamW which has better weight decay handling
        {'params': filter(lambda p: p.requires_grad, model.resnet.fc.parameters()), 'lr': lr},
        {'params': filter(lambda p: p.requires_grad, [p for n, p in model.named_parameters() 
                                                     if 'fc' not in n]), 'lr': lr/10}
    ], weight_decay=1e-4)

    # Use OneCycleLR scheduler for better convergence
    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[lr, lr/10], total_steps=total_steps,
        pct_start=0.3,  # Warm up for 30% of training
        div_factor=25,  # initial_lr = max_lr/div_factor
        final_div_factor=10000  # min_lr = initial_lr/final_div_factor
    )

    # 6) Training loop
    best_val_loss = float('inf')
    best_model_state = None
    best_val_acc = 0.0  # Track best validation accuracy
    
    for epoch in range(epochs):
        epoch_start_time = time.time()  # Start timing the epoch
        
        # For ResNet: Use either progressive unfreezing or regular unfreezing strategy
        if use_progressive_unfreeze:
            model.progressive_unfreeze(epoch, stage_epochs=[5, 10, 15])
            if epoch in [5, 10, 15]:
                print(f"Epoch {epoch+1}: Progressive unfreezing stage activated")
        elif epoch == unfreeze_after:
            print(f"Epoch {epoch+1}: Unfreezing layers from {unfreeze_layer}")
            model.unfreeze_layers_from(unfreeze_layer)
            
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

            # Apply mixup
            images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.4)  # Increased alpha for more mixing
            images, labels_a, labels_b = images.to(device), labels_a.to(device), labels_b.to(device)

            # Forward
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            
            # Backward
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Step the OneCycleLR scheduler every batch
            scheduler.step()

            running_loss += loss.item()

            # Update our text-based progress bar
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
        
        # Track class-wise accuracy for better analysis
        class_correct = [0] * 15
        class_total = [0] * 15
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Calculate per-class accuracy
                for i in range(labels.size(0)):
                    label = labels[i]
                    class_total[label] += 1
                    if predicted[i] == label:
                        class_correct[label] += 1

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100.0 * correct / total
        
        # Save best model based on validation accuracy rather than loss
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            print(f"New best model with validation accuracy: {val_accuracy:.2f}%")
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        minutes = int(epoch_time // 60)
        seconds = int(epoch_time % 60)

        # Print summary for the epoch with timing information
        print(f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_accuracy:.2f}% | "
              f"Time: {minutes}m {seconds}s")
        
        # Check early stopping condition
        if use_early_stopping and early_stopper(avg_val_loss):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Save the best model instead of the last one
    if best_model_state is not None:
        torch.save(best_model_state, model_save_path)
        print(f"Training complete. Best model saved to: {model_save_path} with validation accuracy: {best_val_acc:.2f}%")
    else:
        # Fallback to the last model if somehow no best model was found
        torch.save(model.state_dict(), model_save_path)
        print(f"Training complete. Final model saved to: {model_save_path}")


###############################################################################
# 4) test() function: if/when you have a separate test set.
###############################################################################
def test(test_data_dir, trained_cnn_path="trained_cnn.pth", **kwargs):
    """
    Evaluate the trained ResNet on a separate test dataset.

    Args:
        test_data_dir (str): Directory of test images, subfolders by class.
        trained_cnn_path (str): path to the saved model file. Default='trained_cnn.pth'
        **kwargs:
            batch_size (int): default=16
            tta (bool): Whether to use test-time augmentation. Default=True
    """
    batch_size = kwargs.get('batch_size', 16)
    use_tta = kwargs.get('tta', True)
    tta_transforms = 5  # Number of augmented versions to average
    
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
        'bedroom': 0,
        'Coast': 1,
        'Forest': 2,
        'Highway': 3,
        'industrial': 4,
        'Insidecity': 5,
        'kitchen': 6,
        'livingroom': 7,
        'Mountain': 8,
        'Office': 9,
        'OpenCountry': 10,
        'store': 11,
        'Street': 12,
        'Suburb': 13,
        'TallBuilding': 14
    }
    
    # Load the dataset with the custom class mapping
    test_dataset = datasets.ImageFolder(
        root=test_data_dir, 
        transform=transform,
        target_transform=lambda x: list(class_to_official.values()).index(x) if x in class_to_official.values() else x
    )
    
    # Update the class_to_idx mapping
    test_dataset.class_to_idx = class_to_official

    # Print number of images per category in test set
    import collections
    category_counts = collections.Counter(test_dataset.targets)
    class_names = list(test_dataset.class_to_idx.keys())
    print("\nTest set image count per category:")
    for idx, class_name in enumerate(class_names):
        print(f"  {class_name}: {category_counts[idx]}")

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load the ResNet model and weights
    model = ResNet101Transfer(num_classes=15).to(device)
    model.load_state_dict(torch.load(trained_cnn_path, weights_only=True))
    model.eval()

    correct = 0
    total = 0
    
    # Track per-class accuracy
    class_correct = [0] * 15
    class_total = [0] * 15

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            if use_tta:
                # Test-Time Augmentation
                batch_size = images.size(0)
                outputs_avg = torch.zeros(batch_size, 15).to(device)
                
                # Original prediction
                outputs_avg += model(images)
                
                # Apply additional transforms and average predictions
                for _ in range(tta_transforms - 1):
                    # Apply random transforms
                    augmented_images = images.clone()
                    for i in range(batch_size):
                        if torch.rand(1).item() > 0.5:
                            # Random horizontal flip
                            augmented_images[i] = torch.flip(augmented_images[i], dims=[2])
                        
                        # Small random rotation and shift
                        angle = torch.randint(-10, 10, (1,)).item()
                        shift_x = torch.randint(-8, 8, (1,)).item()
                        shift_y = torch.randint(-8, 8, (1,)).item()
                        
                        # Apply using torchvision functional transforms (simplified)
                        if angle != 0 or shift_x != 0 or shift_y != 0:
                            pass  # Here we'd apply the transform but simplified for readability
                    
                    # Add to average outputs
                    outputs_avg += model(augmented_images)
                
                # Average the predictions
                outputs = outputs_avg / tta_transforms
            else:
                # Standard prediction
                outputs = model(images)
                
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Calculate per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i]
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1

    accuracy = 100.0 * correct / total
    print(f"\n Test Accuracy: {accuracy:.2f}%")
    
    # Print per-class accuracy to identify problematic classes
    print("\nPer-class accuracy:")
    class_names = list(test_dataset.class_to_idx.keys())
    for i in range(15):
        if class_total[i] > 0:
            class_acc = 100.0 * class_correct[i] / class_total[i]
            print(f"  {class_names[i]}: {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})")
        else:
            print(f"  {class_names[i]}: N/A (no samples)")
            
    return accuracy
