import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

IMAGE_FOLDER = 'images_data'
OUTPUT_CSV = 'image_features.csv'

VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png')

def augment_image(img):
    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    flipped = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  
    return [rotated, flipped, gray_rgb]

def extract_histogram_features(img):
    chans = cv2.split(img)
    hist = np.concatenate([
        cv2.calcHist([c], [0], None, [256], [0, 256]).flatten()
        for c in chans
    ])
    return hist

def display_images(images, titles):
    plt.figure(figsize=(15, 4))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    records = []

    if not os.path.exists(IMAGE_FOLDER):
        print(f"Folder not found: {IMAGE_FOLDER}")
        return

    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(VALID_EXTENSIONS)]

    if not image_files:
        print("No images found.")
        return

    print(f"Found {len(image_files)} image(s). Processing...")

    for file in image_files:
        img_path = os.path.join(IMAGE_FOLDER, file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Could not load image: {file}")
            continue

        person = file.split("-")[0]
        expression = file.split("-")[1].split(".")[0]

        variations = [("original", img)] + list(zip(["rotated", "flipped", "grayscale"], augment_image(img)))

        for label, variant_img in variations:
            features = extract_histogram_features(variant_img)
            record = {
                'person': person,
                'expression': expression,
                'variation': label,
                **{f'feat_{i}': val for i, val in enumerate(features)}
            }
            records.append(record)

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
