from scene_recog_cnn import train, test
import sys

def main():
    print("\nScene Recognition CNN")
    print("---------------------")
    print("1. Train CNN model")
    print("2. Test model")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ")
    
    if choice == '1':
        print("\nStarting ResNet-101 model training...")
        train(
            train_data_dir="./train",
            batch_size=32,  # Increased from 6 to 32 for better gradient estimates
            lr=0.002,  # Slightly increased learning rate
            epochs=50,
            val_split=0.2,
            model_save_path="trained_cnn.pth",  # Save as trained_cnn.pth
            early_stopping=True,
            patience=5,  # Increased patience for early stopping
            progressive_unfreeze=True,  # Use progressive unfreezing strategy
            label_smoothing=0.1  # Add label smoothing
        )
        # Exit after training
        sys.exit(0)
    elif choice == '2':
        test_dir = input("\nEnter path to test directory (default: ./test): ") or "./test"
        model_path = input("\nEnter path to model file (default: trained_cnn.pth): ") or "trained_cnn.pth"
        
        print(f"\nTesting model from {model_path} on images in {test_dir}...")
        test(
            test_data_dir=test_dir,
            trained_cnn_path=model_path,
            batch_size=64,
            tta=True  # Enable test-time augmentation
        )
        # Exit after testing
        sys.exit(0)
    elif choice == '3':
        print("\nExiting program.")
        sys.exit(0)
    else:
        print("\nInvalid choice. Please enter 1, 2, or 3.")
        sys.exit(1)

if __name__ == "__main__":
    main()
