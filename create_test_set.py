import os
import shutil
from pathlib import Path
import random

def create_test_set(scene15_dir, train_dir, test_dir, test_ratio=0.3):
    """
    Create a test set with images from Scene-15 that are not in the train folder.
    
    Args:
        scene15_dir: Path to the Scene-15 directory
        train_dir: Path to the training directory
        test_dir: Path to create the test directory
        test_ratio: Ratio of non-train images to use for testing (default 0.3)
    """
    # Ensure the test directory exists
    os.makedirs(test_dir, exist_ok=True)
    
    # Dictionary to store statistics
    stats = {
        "total_scene15_images": 0,
        "total_train_images": 0,
        "total_test_images": 0,
        "categories": {}
    }
    
    # Map between Scene-15 folder names and train folder names
    folder_mapping = {
        "bedroom": "bedroom",
        "CALsuburb": "Suburb", 
        "industrial": "industrial",
        "kitchen": "kitchen",
        "livingroom": "livingroom",
        "MITcoast": "Coast",
        "MITforest": "Forest",
        "MIThighway": "Highway",
        "MITinsidecity": "Insidecity",
        "MITmountain": "Mountain",
        "MITopencountry": "OpenCountry",
        "MITstreet": "Street",
        "MITtallbuilding": "TallBuilding",
        "PARoffice": "Office",
        "store": "store"
    }
    
    print(f"Creating test set from {scene15_dir}")
    
    # Process each category folder
    for scene15_folder, train_folder in folder_mapping.items():
        scene15_path = os.path.join(scene15_dir, scene15_folder)
        train_path = os.path.join(train_dir, train_folder)
        test_path = os.path.join(test_dir, train_folder)  # Use train folder naming
        
        # Skip if the Scene-15 folder doesn't exist
        if not os.path.exists(scene15_path):
            print(f"Warning: {scene15_path} not found, skipping")
            continue
            
        # Create test category folder
        os.makedirs(test_path, exist_ok=True)
        
        # Get all image filenames from Scene-15 folder
        scene15_images = set(os.listdir(scene15_path))
        
        # Get all image filenames from train folder (if it exists)
        train_images = set()
        if os.path.exists(train_path):
            train_images = set(os.listdir(train_path))
        
        # Find images in Scene-15 that are not in the train folder
        unique_images = list(scene15_images - train_images)
        
        # Randomly select a subset based on test_ratio
        num_test_images = int(len(unique_images) * test_ratio)
        if num_test_images > 0:
            selected_images = random.sample(unique_images, num_test_images)
        else:
            selected_images = unique_images
            
        # Copy selected images to test folder
        for image in selected_images:
            src = os.path.join(scene15_path, image)
            dst = os.path.join(test_path, image)
            shutil.copy2(src, dst)
        
        # Update statistics
        stats["total_scene15_images"] += len(scene15_images)
        stats["total_train_images"] += len(train_images)
        stats["total_test_images"] += len(selected_images)
        stats["categories"][train_folder] = {
            "scene15_count": len(scene15_images),
            "train_count": len(train_images),
            "test_count": len(selected_images)
        }
        
        print(f"  {train_folder}: {len(selected_images)} test images created")
    
    # Print summary statistics
    print("\nTest Set Creation Summary:")
    print(f"Total Scene-15 images: {stats['total_scene15_images']}")
    print(f"Total training images: {stats['total_train_images']}")
    print(f"Total test images: {stats['total_test_images']}")
    print("\nCategory breakdown:")
    
    for category, counts in stats["categories"].items():
        print(f"  {category}: {counts['test_count']} test images from "
              f"{counts['scene15_count']} Scene-15 images "
              f"({counts['train_count']} were in training)")

if __name__ == "__main__":
    # Set paths
    scene15_dir = "Scene-15"
    train_dir = "train"
    test_dir = "test"
    
    # Get user input for test_ratio
    try:
        test_ratio = float(input("Enter ratio of non-training images to use for test set (0.0-1.0, default 0.3): ") or 0.3)
        test_ratio = max(0.0, min(1.0, test_ratio))  # Clamp between 0 and 1
    except ValueError:
        print("Invalid input, using default ratio of 0.3")
        test_ratio = 0.3
    
    # Create the test set
    create_test_set(scene15_dir, train_dir, test_dir, test_ratio)
    
    print(f"\nTest set created successfully at {test_dir}")