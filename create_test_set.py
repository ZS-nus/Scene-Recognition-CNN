import os
import shutil
from pathlib import Path
import random
import re

def create_test_set(scene15_dir, train_dir, test_dir):
    """
    Create a test set with ALL images from Scene-15 that are not in the train folder.
    Files are compared by their numeric identifiers in the filename.
    
    Args:
        scene15_dir: Path to the Scene-15 directory
        train_dir: Path to the training directory
        test_dir: Path to create the test directory
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
    
    def extract_image_id(filename):
        """Extract numeric identifier from filename."""
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else None
    
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
        
        # Get all images from Scene-15 folder with their numeric IDs
        scene15_files = os.listdir(scene15_path)
        scene15_images = {f: extract_image_id(f) for f in scene15_files}
        
        # Get all images from train folder with their numeric IDs
        train_images = {}
        if os.path.exists(train_path):
            train_files = os.listdir(train_path)
            train_images = {f: extract_image_id(f) for f in train_files}
        
        # Find ALL images in Scene-15 that have IDs not in the train folder
        train_ids = set(id for id in train_images.values() if id is not None)
        test_images = [f for f, id in scene15_images.items() 
                      if id is not None and id not in train_ids]
            
        # Copy all non-training images to test folder
        for image in test_images:
            src = os.path.join(scene15_path, image)
            dst = os.path.join(test_path, image)
            shutil.copy2(src, dst)
        
        # Update statistics
        stats["total_scene15_images"] += len(scene15_images)
        stats["total_train_images"] += len(train_images)
        stats["total_test_images"] += len(test_images)
        stats["categories"][train_folder] = {
            "scene15_count": len(scene15_images),
            "train_count": len(train_images),
            "test_count": len(test_images)
        }
        
        print(f"  {train_folder}: {len(test_images)} test images created")
    
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
    
    # Create the test set with all non-training images
    create_test_set(scene15_dir, train_dir, test_dir)
    
    print(f"\nTest set created successfully at {test_dir}")