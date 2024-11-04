import os
import glob
import cv2
import numpy as np

# Step 1: Image Acquisition
def load_images_from_directory(directory_path):
    image_paths = glob.glob(os.path.join(directory_path, '*.jpg'))  # Adjust the extension as needed
    images = [cv2.imread(image_path) for image_path in image_paths]
    return images

# Step 2: Noise Reduction
def reduce_noise(image):
    denoised_image = cv2.GaussianBlur(image, (5, 5), 0)
    return denoised_image

# Step 3: Histogram Equalization
def enhance_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced_image = cv2.equalizeHist(gray_image)
    return enhanced_image

# Step 4: Resizing and Normalization
def preprocess_image(image, target_size=(256, 256)):
    resized_image = cv2.resize(image, target_size)
    normalized_image = resized_image / 255.0
    return normalized_image

# Step 5: Sharpening
def sharpen_image(image):
    kernel = np.array([[0, -1, 0], 
                       [-1, 5,-1], 
                       [0, -1, 0]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image

# Step 6: Contrast Adjustment
def adjust_contrast(image, alpha=1.5, beta=0):
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image

# Main function to process all images
def process_images(directory_path):
    images = load_images_from_directory(directory_path)
    print(f"Loaded {len(images)} images.")
    processed_images = []
    
    for i, image in enumerate(images):
        print(f"Processing image {i}...")
        
        denoised_image = reduce_noise(image)
        cv2.imshow(f'Denoised Image {i}', denoised_image)
        cv2.waitKey(0)
        
        enhanced_image = enhance_image(denoised_image)
        cv2.imshow(f'Enhanced Image {i}', enhanced_image)
        cv2.waitKey(0)
        
        preprocessed_image = preprocess_image(enhanced_image)
        cv2.imshow(f'Preprocessed Image {i}', preprocessed_image)
        cv2.waitKey(0)
        
        sharpened_image = sharpen_image(preprocessed_image)
        cv2.imshow(f'Sharpened Image {i}', sharpened_image)
        cv2.waitKey(0)
        
        final_image = adjust_contrast(sharpened_image)
        cv2.imshow(f'Final Image {i}', final_image)
        cv2.waitKey(0)
        
        processed_images.append(final_image)
    
    return processed_images

# Example usage
directory_path = r'C:\Users\cragu\minnor_project\moon_imgs'  # Update this path to your folder
processed_images = process_images(directory_path)

# Display the first processed image for verification
cv2.imshow('Processed Image', processed_images[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
