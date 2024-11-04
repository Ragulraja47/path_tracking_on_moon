import cv2
import numpy as np
import os
import heapq

# Create output directory if it doesn't exist
output_dir = "D:\\Saran\\Minor project\\python\\Scripts\\processed_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define path to the folder containing moon images
input_dir = "D:\\Saran\\Minor project\\python\\Scripts\\moon_imgs"  # Replace with your input directory path

# A* pathfinding algorithm
def a_star_pathfinding(img, start, end, mask):
    rows, cols = img.shape
    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {start: None}
    g_score = {start: 0}
    f_score = {start: np.linalg.norm(np.array(start) - np.array(end))}
    
    while open_list:
        _, current = heapq.heappop(open_list)

        if current == end:
            path = []
            while current:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and mask[neighbor] == 0:
                tentative_g_score = g_score[current] + np.linalg.norm(np.array(current) - np.array(neighbor))

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + np.linalg.norm(np.array(neighbor) - np.array(end))
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))

    return None  # Return None if no path is found

# Function to process each image
def process_image(image_path, output_dir):
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Step 1: Enhance Image for Clarity
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
    alpha, beta = 1.5, 20  # Contrast and brightness control
    enhanced_img = cv2.convertScaleAbs(blurred_img, alpha=alpha, beta=beta)
    
    # Step 2: Detect Objects (Using Canny Edge Detection and Contours)
    edges = cv2.Canny(enhanced_img, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_img, contours, -1, (0, 0, 255), 1)
    
    # Create mask for obstacles with a thickened contour
    mask = np.zeros_like(img)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    mask = cv2.dilate(mask, np.ones((5,5), np.uint8), iterations=1)  # Expand mask to ensure clearance around obstacles
    
    # Step 3: Path Planning - Creating a Safe Route
    start_point = (50, 50)  # Example start point
    end_point = (img.shape[1] - 50, img.shape[0] - 50)  # Example end point
    
    # Pathfinding with A* Algorithm
    path = a_star_pathfinding(img, start_point, end_point, mask)
    
    if path:
        for point in path:
            cv2.circle(contour_img, point, 1, (0, 255, 0), -1)  # Mark the path in green
        print(f"Path found and marked on the image.")
    else:
        cv2.putText(contour_img, "No Path Found", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print("No path found due to obstacles.")
    
    # Save the processed image in the output directory
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, contour_img)
    print(f"Processed and saved: {output_path}")

# Process all images in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_dir, filename)
        process_image(image_path, output_dir)
