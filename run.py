from scene_recog_cnn import train

if __name__ == "__main__":
    # Suppose your training data is in "./train" 
    # which you want to split for train vs val
    train(
        train_data_dir="./train",
        batch_size=32,  # Changed to power of 2 for better GPU utilization
        lr=0.001,
        epochs=50,
        val_split=0.2,  # 80% train, 20% validate
        model_save_path="trained_cnn.pth",
        early_stopping=True,  # Enable early stopping
        patience=5  # Stop if validation loss doesn't improve for 5 epochs
    )
