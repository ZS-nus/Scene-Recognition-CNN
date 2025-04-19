import os
import sys

def count_images_per_category(test_dir='./test'):
    """
    Counts the number of files (assumed to be images) in each subdirectory 
    (category) within the specified test directory.

    Args:
        test_dir (str): Path to the test directory containing category subfolders.
    """
    if not os.path.isdir(test_dir):
        print(f"Error: Directory not found at '{test_dir}'")
        sys.exit(1)

    category_counts = {}
    
    print(f"Counting images in subdirectories of: {test_dir}")

    try:
        # List all entries in the test directory
        for category_name in os.listdir(test_dir):
            category_path = os.path.join(test_dir, category_name)
            
            # Check if it's a directory
            if os.path.isdir(category_path):
                # List all files within the category directory
                files = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]
                # Store the count
                category_counts[category_name] = len(files)
            # Optional: Handle cases where entries are not directories (e.g., files directly in ./test)
            # else:
            #     print(f"  Skipping non-directory entry: {category_name}")

    except OSError as e:
        print(f"Error accessing directory or files: {e}")
        sys.exit(1)

    # Print the results
    print("\nImage count per category:")
    if not category_counts:
        print("  No category subdirectories found.")
    else:
        # Sort categories alphabetically for consistent output
        for category_name in sorted(category_counts.keys()):
            print(f"  {category_name}: {category_counts[category_name]}")

if __name__ == "__main__":
    # You can change the directory path here if needed
    test_directory = './test' 
    count_images_per_category(test_directory)