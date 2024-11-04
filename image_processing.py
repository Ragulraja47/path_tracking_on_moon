import os
import glob
import cv2
import numpy as np

# Step 1: Image Acquisition
def load_images_from_directory(directory_path):
    image_paths = glob.glob(os.path.join(directory_path, '*.jpg'))  # Adjust the extension as needed
    images = [cv2.imread(image_path) for image_path in image_paths]
    return images, image_paths

# Step 2: Noise Reduction
def reduce_noise(image):
    denoised_image = cv2.GaussianBlur(image, (5, 5), 0)
    return denoised_image

# Step 3: Histogram Equalization
def enhance_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced_image = cv2.equalizeHist(gray_image)
    return enhanced_image

# Step 4: Resizing without Normalization
def preprocess_image(image, target_size=(256, 256)):
    resized_image = cv2.resize(image, target_size)
    return resized_image  # No normalization here

# Step 5: Sharpening
def sharpen_image(image):
    kernel = np.array([[0, -1, 0], 
                       [-1, 5,-1], 
                       [0, -1, 0]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image

# Step 6: Saving the sharpened images with higher quality
def save_sharpened_image(image, save_directory, original_image_name):
    # Ensure the directory exists
    os.makedirs(save_directory, exist_ok=True)

    # Extract the filename and save the sharpened image
    filename = os.path.basename(original_image_name)
    save_path = os.path.join(save_directory, filename)

    # If the file format is JPG, set the quality parameter
    if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
        # Save JPEG with 95% quality
        cv2.imwrite(save_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"Saved JPEG image with 95% quality to {save_path}")
    else:
        # For PNG, ensure lossless quality by saving with no compression
        png_save_path = save_path.replace('.jpg', '.png').replace('.jpeg', '.png')  # Convert to PNG if needed
        cv2.imwrite(png_save_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])  # No compression for PNG
        print(f"Saved PNG image (lossless) to {png_save_path}")

# Main function to process all images
def process_images(directory_path, save_directory):
    images, image_paths = load_images_from_directory(directory_path)
    print(f"Loaded {len(images)} images.")
    
    for i, (image, image_path) in enumerate(zip(images, image_paths)):
        print(f"Processing image {i}...")

        # Step 1: Noise Reduction
        denoised_image = reduce_noise(image)
        cv2.imshow(f'Denoised Image {i}', denoised_image)
        cv2.waitKey(0)

        # Step 2: Histogram Equalization
        enhanced_image = enhance_image(denoised_image)
        cv2.imshow(f'Enhanced Image {i}', enhanced_image)
        cv2.waitKey(0)

        # Step 3: Resizing without Normalization
        preprocessed_image = preprocess_image(enhanced_image)
        cv2.imshow(f'Preprocessed Image {i}', preprocessed_image)
        cv2.waitKey(0)

        # Step 4: Sharpening (final output)
        sharpened_image = sharpen_image(preprocessed_image)
        cv2.imshow(f'Sharpened Image {i}', sharpened_image)
        cv2.waitKey(0)

        # Save the sharpened image with high quality
        save_sharpened_image(sharpened_image, save_directory, image_path)

# Example usage
directory_path = r'C:\Users\cragu\minnor_project\moon_imgs'  # Input folder
save_directory = r'C:\Users\cragu\minnor_project\processed_moon_images'  # Output folder
process_images(directory_path, save_directory)

cv2.destroyAllWindows()
