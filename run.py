from scene_recog_cnn import train, test
import sys

def main():
    print("\nScene Recognition CNN")
    print("---------------------")
    print("1. Train model")
    print("2. Test model")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ")
    
    if choice == '1':
        print("\nStarting model training...")
        train(
            train_data_dir="./train",
            batch_size=64,
            lr=0.001,
            epochs=100,
            val_split=0.2,
            model_save_path="trained_cnn.pth",
            early_stopping=False,
            patience=5
        )
        # Exit after training
        sys.exit(0)
    elif choice == '2':
        test_dir = input("\nEnter path to test directory (default: ./train): ") or "./train"
        print(f"\nTesting model from trained_cnn.pth on images in {test_dir}...")
        test(
            test_data_dir=test_dir,
            trained_cnn_path="trained_cnn.pth",
            batch_size=64
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
