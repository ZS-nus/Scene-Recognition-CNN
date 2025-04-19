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
        print("\nStarting model training...")
        train(train_data_dir="./train",
              epochs=5, bs=32,
              lr_head=1e-3, lr_backbone=1e-4,
              save_path="trained_cnn.pth",
              weights_path="resnet50_places365.pth.tar")
        sys.exit(0)  # Consider removing this if you want the menu to reappear
    elif choice == '2':
        test_dir = input("\nEnter path to test directory (default: ./test): ") or "./test"
        model_path = input("\nEnter path to model file (default: trained_cnn.pth): ") or "trained_cnn.pth"

        print(f"\nTesting model from {model_path} on images in {test_dir}...")
        test(
            test_data_dir=test_dir,
            model_path=model_path,
            bs=64,  # Changed from batch_size
            tta=True  # Enable test-time augmentation
        )
        sys.exit(0)  # Consider removing this if you want the menu to reappear
    elif choice == '3':
        print("\nExiting program.")
        sys.exit(0)
    else:
        print("\nInvalid choice. Please enter 1, 2, or 3.")
        main()  # Example: loop back to the menu

if __name__ == "__main__":
    main()
