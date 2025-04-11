import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from model import ImprovedCNN, get_device



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
    """
    
        # Get the best available device
    device = get_device()
    
    
    
    # 1) Hyperparameters & defaults
    batch_size = kwargs.get('batch_size', 16)
    lr = kwargs.get('lr', 0.001)
    epochs = kwargs.get('epochs', 5)
    val_split = kwargs.get('val_split', 0.2)  # e.g., 20% of images => val set
    model_save_path = kwargs.get('model_save_path', 'trained_cnn.pth')

    # 2) Define transforms
    transform = transforms.Compose([
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
        transform=transform,
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(full_dataset.class_to_idx)
    print(f"Total images: {dataset_size} | Training: {train_size}, Validation: {val_size}")

    # 5) Create model, define loss & optimizer
    model = ImprovedCNN(num_classes=15).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 6) Training loop
    for epoch in range(epochs):
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
                # Move tensors to device - this line was missing
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

        # Print summary for the epoch
        print(f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_accuracy:.2f}%")

    # 7) Save the trained model
    torch.save(model.state_dict(), model_save_path)
    print(f"Training complete. Model saved to: {model_save_path}")


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
    model.load_state_dict(torch.load(trained_cnn_path))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            # Move tensors to device - this line was missing
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
