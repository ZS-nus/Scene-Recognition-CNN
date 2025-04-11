from scene_recog_cnn import train

if __name__ == "__main__":
    # Suppose your training data is in "./train" 
    # which you want to split for train vs val
    train(
        train_data_dir="./train",
        batch_size=16,
        lr=0.001,
        epochs=5,
        val_split=0.2,  # 80% train, 20% validate
        model_save_path="trained_cnn.pth"
    )
