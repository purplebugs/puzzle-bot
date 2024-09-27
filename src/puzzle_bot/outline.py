# Import necessary libraries
import cv2
import numpy as np
import svgwrite
import matplotlib.pyplot as plt
import sys

# Set the minimum size (in pixels) for connected components to keep
min_size = 500  # Adjust this value based on your image

# Load the image
print("Loading the image...")
if len(sys.argv)!=2:
    print("Usage: %s <image file>" % sys.argv[0])
    sys.exit(-1)

image = cv2.imread(sys.argv[1])

# Check if the image was loaded successfully
if image is None:
    print("Error: Could not load image.")
    exit()

# Display the original image using matplotlib
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')
plt.show()

# Convert the image to HSV color space
print("Converting image to HSV color space...")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the range for green color in HSV
# These values might need adjustment based on your specific shade of green
# lower_green = np.array([25, 40, 40])   # Expanded lower bound
# upper_green = np.array([95, 255, 255])  # Expanded upper bound
lower_green = np.array([20, 10, 10])    # Very broad lower bound
upper_green = np.array([100, 255, 255])  # Very broad upper bound

# Create a mask for green color
print("Creating mask for green color...")
mask_green = cv2.inRange(hsv, lower_green, upper_green)

# Invert the mask to get the non-green areas
mask_non_green = cv2.bitwise_not(mask_green)

# Apply the mask to the original image
print("Applying mask to remove green background...")
image_no_green = cv2.bitwise_and(image, image, mask=mask_non_green)

# Display the image after removing green background
plt.imshow(cv2.cvtColor(image_no_green, cv2.COLOR_BGR2RGB))
plt.title('Image after Removing Green Background')
plt.axis('off')
plt.show()

# Convert the masked image to grayscale
print("Converting to grayscale...")
gray = cv2.cvtColor(image_no_green, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise and smooth the image
print("Applying Gaussian Blur...")
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Perform Canny edge detection
print("Performing Canny edge detection...")
edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

# Dilate the edges to close gaps (optional)
print("Dilating edges to close gaps...")
kernel = np.ones((3, 3), np.uint8)
edges_dilated = cv2.dilate(edges, kernel, iterations=1)

# Display the edge-detected image
plt.imshow(edges_dilated, cmap='gray')
plt.title('Edge Detected Image')
plt.axis('off')
plt.show()

# Perform connected component analysis to remove small components
print("Removing small connected components...")
# Invert the edges image for connectedComponentsWithStats (background should be zero)
edges_inv = cv2.bitwise_not(edges_dilated)

# Perform connected components analysis
num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(edges_inv, connectivity=8)

# Create an empty image to store the filtered components
filtered_edges_inv = np.zeros_like(edges_inv)

# Iterate through each component and keep those larger than min_size
for i in range(1, num_labels):  # Start from 1 to skip the background
    area = stats[i, cv2.CC_STAT_AREA]
    if area >= min_size:
        # Keep the component
        filtered_edges_inv[labels_im == i] = 255

# Invert the image back to get the filtered edges
filtered_edges = cv2.bitwise_not(filtered_edges_inv)

# Display the filtered edge image
plt.imshow(filtered_edges, cmap='gray')
plt.title('Filtered Edge Image')
plt.axis('off')
plt.show()

# Find contours from the filtered edges
print("Finding contours...")
contours, hierarchy = cv2.findContours(filtered_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Check if any contours were found
if not contours:
    print("No contours found. Adjust the parameters and try again.")
    exit()

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
    epsilon = 0.0005 * perimeter  # Adjusted epsilon for better approximation
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    approx_contours.append(approx)

# Create an SVG drawing
print("Creating SVG file...")
height, width = image.shape[:2]
dwg = svgwrite.Drawing('puzzle_vector.svg', size=(width, height), profile='tiny')

# Add each contour to the SVG drawing
print("Adding contours to SVG...")
for cnt in approx_contours:
    # Reshape the contour array and convert it to a list of tuples
    points = cnt.reshape(-1, 2)
    # Convert points to a list of (x, y) tuples
    point_list = [(int(point[0]), int(point[1])) for point in points]
    # Add a polygon element to the SVG drawing
    dwg.add(dwg.polygon(point_list, stroke='black', fill='none'))

# Save the SVG file
dwg.save()
print("SVG file 'puzzle_vector.svg' has been saved successfully.")