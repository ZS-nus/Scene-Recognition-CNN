from scene_recog_cnn import train, test
import sys

def main():
    print("\nScene Recognition CNN")
    print("---------------------")
    print("1. Train custom CNN model")
    print("2. Train ResNet-50 transfer learning model")
    print("3. Test model")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ")
    
    if choice == '1':
        print("\nStarting custom CNN model training...")
        train(
            train_data_dir="./train",
            batch_size=32,
            lr=0.001,
            epochs=100,
            val_split=0.2,
            model_save_path="trained_cnn.pth",
            early_stopping=False,
            patience=5,
            model_type="cnn"
        )
        # Exit after training
        sys.exit(0)
    elif choice == '2':
        print("\nStarting ResNet-50 transfer learning model training...")
        train(
            train_data_dir="./train",
            batch_size=32,
            lr=0.001,
            epochs=50,  # Less epochs needed for transfer learning
            val_split=0.2,
            model_save_path="trained_resnet.pth",
            early_stopping=True,
            patience=5,
            model_type="resnet",
            unfreeze_after=5,
            unfreeze_layer="layer4"
        )
        # Exit after training
        sys.exit(0)
    elif choice == '3':
        test_dir = input("\nEnter path to test directory (default: ./test): ") or "./test"
        model_path = input("\nEnter path to model file (default: trained_cnn.pth): ") or "trained_cnn.pth"
        model_type = "resnet" if "resnet" in model_path.lower() else "cnn"
        
        print(f"\nTesting model from {model_path} on images in {test_dir}...")
        test(
            test_data_dir=test_dir,
            trained_cnn_path=model_path,
            batch_size=64,
            model_type=model_type
        )
        # Exit after testing
        sys.exit(0)
    elif choice == '4':
        print("\nExiting program.")
        sys.exit(0)
    else:
        print("\nInvalid choice. Please enter 1, 2, 3, or 4.")
        sys.exit(1)

if __name__ == "__main__":
    main()
