from ultralytics import YOLO
import shutil
import os

def main():
    print("Loading base model...")
    # Load a model
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    print("Starting training...")
    # Train the model
    results = model.train(
        data="Datasets/dummy_ppe_data/data.yaml",
        epochs=5,
        imgsz=640,
        project="runs",
        name="ppe_training"
    )

    print("Training finished. Moving best model to models/ppe_yolo.pt")
    
    # Path to the newly trained model
    # Note: If 'ppe_training' already exists, it might create ppe_training2, ppe_training3, etc.
    # We will just find the most recently modified runs/ppe_training*/weights/best.pt
    
    runs_dir = "runs"
    subdirs = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d)) and d.startswith("ppe_training")]
    if subdirs:
        latest_subdir = max(subdirs, key=os.path.getmtime)
        best_pt_path = os.path.join(latest_subdir, "weights", "best.pt")
        
        if os.path.exists(best_pt_path):
            os.makedirs("models", exist_ok=True)
            shutil.copy(best_pt_path, "models/ppe_yolo.pt")
            print(f"Successfully copied {best_pt_path} to models/ppe_yolo.pt")
        else:
            print(f"Error: {best_pt_path} not found.")
    else:
        print("Error: Could not find training runs directory.")

if __name__ == "__main__":
    main()
