import os
import cv2
import numpy as np
import yaml
import random

def create_dirs(base_dir):
    os.makedirs(os.path.join(base_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'labels', 'val'), exist_ok=True)
    os.makedirs("data/input", exist_ok=True)

def generate_image_and_label(image_path, label_path, width=640, height=480):
    img = np.ones((height, width, 3), dtype=np.uint8) * 200 # Light gray background
    
    labels = []
    
    # Randomly generate 1 to 3 "workers"
    num_workers = random.randint(1, 3)
    for _ in range(num_workers):
        # Person bounding box
        pw = random.randint(50, 100)
        ph = random.randint(150, 250)
        px = random.randint(0, width - pw)
        py = random.randint(0, height - ph)
        
        # Draw Person (class 2) - Blue rectangle
        cv2.rectangle(img, (px, py), (px+pw, py+ph), (255, 0, 0), -1)
        
        # Center coords in YOLO format (normalized)
        cx = (px + pw / 2) / width
        cy = (py + ph / 2) / height
        nw = pw / width
        nh = ph / height
        labels.append(f"2 {cx:.4f} {cy:.4f} {nw:.4f} {nh:.4f}")
        
        # Hardhat (class 0) - Yellow rectangle on top of person
        if random.random() > 0.3: # 70% chance of hardhat
            hx = px + int(pw * 0.2)
            hy = py
            hw = int(pw * 0.6)
            hh = int(ph * 0.15)
            cv2.rectangle(img, (hx, hy), (hx+hw, hy+hh), (0, 255, 255), -1)
            hcx = (hx + hw / 2) / width
            hcy = (hy + hh / 2) / height
            hnw = hw / width
            hnh = hh / height
            labels.append(f"0 {hcx:.4f} {hcy:.4f} {hnw:.4f} {hnh:.4f}")
            
        # Vest (class 1) - Orange rectangle in middle of person
        if random.random() > 0.4: # 60% chance of vest
            vx = px + int(pw * 0.1)
            vy = py + int(ph * 0.2)
            vw = int(pw * 0.8)
            vh = int(ph * 0.4)
            cv2.rectangle(img, (vx, vy), (vx+vw, vy+vh), (0, 165, 255), -1)
            vcx = (vx + vw / 2) / width
            vcy = (vy + vh / 2) / height
            vnw = vw / width
            vnh = vh / height
            labels.append(f"1 {vcx:.4f} {vcy:.4f} {vnw:.4f} {vnh:.4f}")

    cv2.imwrite(image_path, img)
    with open(label_path, 'w') as f:
        f.write('\n'.join(labels))

def generate_dataset(base_dir, num_train=100, num_val=20):
    for i in range(num_train):
        generate_image_and_label(
            os.path.join(base_dir, 'images', 'train', f'img_{i:03d}.jpg'),
            os.path.join(base_dir, 'labels', 'train', f'img_{i:03d}.txt')
        )
    for i in range(num_val):
        generate_image_and_label(
            os.path.join(base_dir, 'images', 'val', f'img_{i:03d}.jpg'),
            os.path.join(base_dir, 'labels', 'val', f'img_{i:03d}.txt')
        )

def generate_yaml(base_dir):
    data = {
        'path': os.path.abspath(base_dir),
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'hardhat',
            1: 'vest',
            2: 'person'
        }
    }
    with open(os.path.join(base_dir, 'data.yaml'), 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
        
def generate_dummy_video(output_path, num_frames=100, width=640, height=480):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 10.0, (width, height))
    
    # Simulate a person moving across the screen
    px = 50
    py = 200
    for i in range(num_frames):
        img = np.ones((height, width, 3), dtype=np.uint8) * 200
        
        pw, ph = 60, 180
        
        # Person
        cv2.rectangle(img, (px, py), (px+pw, py+ph), (255, 0, 0), -1)
        # Hardhat
        cv2.rectangle(img, (px+10, py), (px+50, py+30), (0, 255, 255), -1)
        # Vest
        cv2.rectangle(img, (px+5, py+40), (px+55, py+100), (0, 165, 255), -1)
        
        out.write(img)
        px += 5 # Move right
        if px > width - pw:
            px = 50 # wrap around
            
    out.release()

if __name__ == "__main__":
    base_dir = "Datasets/dummy_ppe_data"
    print("Creating directories...")
    create_dirs(base_dir)
    print("Generating dataset...")
    generate_dataset(base_dir)
    print("Generating YAML...")
    generate_yaml(base_dir)
    print("Generating dummy video...")
    generate_dummy_video("data/input/dummy_video.mp4")
    print("Done!")
