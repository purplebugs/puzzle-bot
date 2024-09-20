# Import necessary libraries
import cv2
import numpy as np
import svgwrite
import matplotlib.pyplot as plt

# Load the image
print("Loading the image...")
image = cv2.imread('data/0_photos/Leica_black_bg_back_raw_2024-09-17_20.00.17.jpeg')
#image = cv2.imread('data/0_photos/Leica_black_bg_front_raw_2024-09-17_20.00.07.jpeg')
#image = cv2.imread('data/0_photos/Leica_white_bg_front_raw_2024-09-17_19.44.59.jpeg')



# Check if the image was loaded successfully
if image is None:
    print("Error: Could not load image.")
    exit()

# Display the original image using matplotlib
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')
plt.show()

# Convert the image to grayscale
print("Converting to grayscale...")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display the grayscale image
plt.imshow(gray, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')
plt.show()

# Apply Gaussian Blur to reduce noise and smooth the image
print("Applying Gaussian Blur...")
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Display the blurred image
plt.imshow(blurred, cmap='gray')
plt.title('Blurred Image')
plt.axis('off')
plt.show()

# Perform Canny edge detection
print("Performing Canny edge detection...")
edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

# Display the edge-detected image
plt.imshow(edges, cmap='gray')
plt.title('Edge Detected Image')
plt.axis('off')
plt.show()

# Find contours from the detected edges
print("Finding contours...")
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a copy of the original image to draw contours
image_with_contours = image.copy()

# Draw contours on the image
print("Drawing contours on the image...")
cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)

# Display the image with contours
plt.imshow(cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB))
plt.title('Contours')
plt.axis('off')
plt.show()

# Approximate contours to reduce the number of points
print("Approximating contours...")
approx_contours = []
for cnt in contours:
    # Calculate the perimeter of the contour
    perimeter = cv2.arcLength(cnt, True)
    # Apply contour approximation
    epsilon = 0.01 * perimeter
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    approx_contours.append(approx)

# Create an SVG drawing
print("Creating SVG file...")
dwg = svgwrite.Drawing('puzzle_vector.svg', profile='tiny')

# Add each contour to the SVG drawing
print("Adding contours to SVG...")
for cnt in approx_contours:
    # Reshape the contour array and convert it to a list of tuples
    points = cnt.reshape(-1, 2).tolist()
    # Add a polygon element to the SVG drawing
    dwg.add(dwg.polygon(points, stroke='black', fill='none'))

# Save the SVG file
dwg.save()
print("SVG file 'puzzle_vector.svg' has been saved successfully.")