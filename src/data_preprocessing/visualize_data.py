import cv2
import os

# Root dataset folder
dataset_path = 'C:/Users/mkr19/Documents/safetyai/data'


splits = ['train', 'val', 'test']

for split in splits:
    images_folder = os.path.join(dataset_path, split, 'images')
    labels_folder = os.path.join(dataset_path, split, 'labels')
    
    print(f"\nProcessing {split} set...")
    
    image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.jpg', '.png'))]
    total_images = 0
    missing_labels = 0

    for img_file in image_files:
        img_path = os.path.join(images_folder, img_file)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(labels_folder, label_file)
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}, skipping.")
            continue
        
        h, w, _ = img.shape
        if not os.path.exists(label_path):
            print(f"Warning: Label file missing for {img_file}")
            missing_labels += 1
        else:
            with open(label_path) as f:
                for line in f.readlines():
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    x1 = int((x_center - width/2) * w)
                    y1 = int((y_center - height/2) * h)
                    x2 = int((x_center + width/2) * w)
                    y2 = int((y_center + height/2) * h)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        
        # Show the first few images only (optional)
        if total_images < 5:  # change this number if you want more previews
            cv2.imshow(f'{split} Image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        total_images += 1

    print(f"{split}: {total_images} images processed, {missing_labels} missing labels")
