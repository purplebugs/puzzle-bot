# Import necessary libraries
import cv2
import numpy as np
import svgwrite
import matplotlib.pyplot as plt
import sys
from scipy.interpolate import splprep, splev


def find_closes_corner(corner, component, search_radius):
    TAB_PCS = 0.3
    candidate = None
    box = component['box']
    height = component['height']
    width = component['width']
    x,y = corner
    min_x = sys.maxsize
    min_y = sys.maxsize
    closest_box_point = None
    for box_point in box:
        dx = abs(x - box_point[0])
        dy = abs(y - box_point[1])
        updateX = False
        updateY = False
        if dx < min_x and dx < search_radius * width:
            updateX = True
        if dy < min_y and dy < search_radius * height:
            updateY = True
        if dx < dy and updateX:
            if abs(y - box_point[1]) < TAB_PCS * height:
                closest_box_point = box_point
                min_x = dx
        if dy < dx and updateY:
            if abs(x - box_point[0]) < TAB_PCS * width:
                closest_box_point = box_point
                min_y = dy
    return closest_box_point
    
def find_corners_by_approxPoly(component, image_with_contours, search_radius=0.05, eps=0.01):
    # Approximate contours and find corners
    print("Approximating contours and finding corners...")
    corners_list = []
    contour = component['contour']
    box = component['box']
    width = component['width']
    height = component['height']
        # Calculate the perimeter of the contour
    perimeter = cv2.arcLength(contour, True)
    # Apply contour approximation
    epsilon = eps * perimeter  # Adjusted epsilon for better corner detection
    approx = cv2.approxPolyDP(contour, epsilon, True)
    # Extract corners from the approximated contour
    corners = approx.reshape(-1, 2)
    for corner in corners:
       matching_box_point = find_closes_corner(corner, component, search_radius)
       if matching_box_point is not None:
           corners_list.append(corner)

    # Optionally, draw the corners on the image
    for corner in corners_list:
        x, y = corner
        cv2.circle(image_with_contours, (x, y), 5, (0, 0, 255), -1)
    print(f"Number of corners: {len(corners_list)}")

    # Display the image with contours and corners
    plt.imshow(cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB))
    plt.title('Contours with Corners')
    plt.axis('off')
    plt.show()

    return np.array(corners_list)

def find_corners_by_angle_change(cnt, angle_threshold=80, window_size=5):
    # Ensure the contour is a 2D array of points
    cnt = cnt.reshape(-1, 2)
    num_points = len(cnt)
    corners = []

    # Convert angle threshold to radians
    angle_threshold_rad = np.deg2rad(angle_threshold)

    angles=[]
    # Iterate over the contour points with a larger sliding window
    for i in range(num_points):
        prev_index = (i - window_size) % num_points
        next_index = (i + window_size) % num_points
        prev_point = cnt[prev_index]
        curr_point = cnt[i]
        next_point = cnt[next_index]

        # Vectors from current point to previous and next points
        vec1 = prev_point - curr_point
        vec2 = next_point - curr_point

        # Normalize the vectors
        vec1_norm = vec1 / np.linalg.norm(vec1) if np.linalg.norm(vec1) != 0 else vec1
        vec2_norm = vec2 / np.linalg.norm(vec2) if np.linalg.norm(vec2) != 0 else vec2

        # Compute the angle between the vectors
        dot_product = np.dot(vec1_norm, vec2_norm)
        # Ensure dot_product is within valid range for arccos
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle = np.arccos(dot_product)
        angles.append(angle)


        # If the angle is less than the threshold, consider it a corner
        if angle <= angle_threshold_rad:
            corners.append(curr_point)

    # Remove duplicate points
    corners = np.unique(corners, axis=0)

    angles_array = np.array(angles)
    print('Angles array:', angles_array)
    sorted_indices = np.argsort(angles_array)
    print('Sorted indices:', sorted_indices)
    smallest_angles = sorted_indices[:4]
    print('Smallest angles:', smallest_angles)
    smallest_numbers = angles_array[smallest_angles]
    print('Smallest numbers:', smallest_numbers)
    
    
    corners = cnt[smallest_angles]
    print('Corners:', corners)

    plt.plot(angles)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Simple Array Plot')
    plt.show()

    

    return corners

# Set the minimum size (in pixels) for connected components to keep
min_size = 5000  # Adjust this value based on your image

# Load the image
print("Loading the image...")
if len(sys.argv) != 2:
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
#lower_green = np.array([20, 10, 10])    # Very broad lower bound
#upper_green = np.array([100, 255, 255])  # Very broad upper bound
lower_green = np.array([30, 50, 50])   # Narrower lower bound
upper_green = np.array([85, 255, 255])  # Narrower upper bound

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

kernel = np.ones((5, 5), np.uint8)  # Adjust kernel size as needed
mask_cleaned = cv2.morphologyEx(mask_non_green, cv2.MORPH_CLOSE, kernel)
mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)

image_cleaned = cv2.bitwise_and(image, image, mask=mask_cleaned)

plt.imshow(cv2.cvtColor(image_cleaned, cv2.COLOR_BGR2RGB))
plt.title('Image after cleaning')
plt.axis('off')
plt.show()

# Convert the masked image to grayscale
print("Converting to grayscale...")
gray = cv2.cvtColor(image_cleaned, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise and smooth the image
print("Applying Gaussian Blur...")
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Perform Canny edge detection
print("Performing Canny edge detection...")
edges = cv2.Canny(blurred, threshold1=100, threshold2=200)

# Dilate the edges to close gaps (optional)
print("Dilating edges to close gaps...")
kernel = np.ones((3, 3), np.uint8)
edges_dilated = cv2.dilate(edges, kernel, iterations=1)

# Display the edge-detected image
plt.imshow(edges_dilated, cmap='gray')
plt.title('Edge Detected Image')
plt.axis('off')
plt.show()

# Remove small connected components
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

print("Drawing contours...")
# Create a black image to draw contours on
contours_image = np.zeros_like(filtered_edges)

# Draw the contours on the black image
cv2.drawContours(contours_image, contours, -1, (255, 255, 255), thickness=2)

# Display the contours alone
plt.imshow(contours_image, cmap='gray')
plt.title('Contours Alone')
plt.axis('off')
plt.show()

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

print("Filled contours...")
bitmap_image = np.zeros_like(filtered_edges)
cv2.drawContours(bitmap_image, contours, -1, 255, thickness=-1)  # Fill the contour
# Display the image with filled contours
plt.imshow(bitmap_image, cmap='gray')
plt.title('Filled Contours')
plt.axis('off')
plt.show()

# Connected component labeling after finding contours
print("Processing contours to extract component properties...")

components = []

# Create an empty image to store the labeled components
label_image = np.zeros_like(filtered_edges, dtype=np.int32)

for idx, cnt in enumerate(contours):
    # Create a mask for the contour
    mask = np.zeros_like(filtered_edges)
    cv2.drawContours(mask, [cnt], -1, 255, thickness=-1)  # Fill the contour

    # Get the pixels belonging to the contour
    pixels = np.column_stack(np.where(mask > 0))

    num_pixels = len(pixels)
    x_coords = pixels[:, 1]
    y_coords = pixels[:, 0]
    min_x, max_x = x_coords.min(), x_coords.max()
    min_y, max_y = y_coords.min(), y_coords.max()
    width_comp = max_x - min_x + 1
    height_comp = max_y - min_y + 1

    # Create bitmap of the outline
    outline_bitmap = np.zeros((height_comp, width_comp), dtype=np.uint8)
    outline_bitmap[y_coords - min_y, x_coords - min_x] = 255

    component = {
        'label': idx + 1,  # Labels starting from 1
        'num_pixels': num_pixels,
        'width': width_comp,
        'height': height_comp,
        'outline_bitmap': outline_bitmap,
        'min_x': min_x,
        'min_y': min_y,
        'contour': cnt,
        'rect': cv2.boundingRect(cnt),
        'box': cv2.boxPoints(cv2.minAreaRect(cnt))
    }
    if component['num_pixels'] > 5000:
        # Find corners
        corners = find_corners_by_approxPoly(component, image_with_contours, search_radius=0.05, eps=0.04)
        # corners = find_corners_by_angle_change(component['contour'], angle_threshold=90)
        component['corners'] = corners
        components.append(component)
    else:
        print(f"Component {idx + 1} has less than 5000 pixels. It will be removed.")

    # Update the label image
    label_image[mask > 0] = idx + 1

print("Number of components:", len(components))


# Display the outline bitmap
for idx, component in enumerate(components):
    print(f"Component {idx + 1}:")
    print(f"  Label: {component['label']}")
    print(f"  Number of Pixels: {component['num_pixels']}")
    print(f"  Width: {component['width']}")
    print(f"  Height: {component['height']}")
    print(f"  Number of pixels in contour: {len(component['contour'])}")
    print(f"  Corners: {component['corners']}")
    
    # Adjust corners to the outline_bitmap coordinate system
    adjusted_corners = component['corners'] - [component['min_x'], component['min_y']]
    
    # Display the outline bitmap with corners overlaid
    plt.imshow(component['outline_bitmap'], cmap='gray')
    plt.title(f'Outline Bitmap of Component {idx + 1}')
    plt.axis('off')
    
    # Plot the corners
    for corner in adjusted_corners:
        plt.plot(corner[0], corner[1], 'ro')  # Red dots for corners

    # Optionally, connect the corners to visualize the shape
    x = np.append(adjusted_corners[:, 0], adjusted_corners[0, 0])  # Close the shape
    y = np.append(adjusted_corners[:, 1], adjusted_corners[0, 1])
    plt.plot(x, y, 'r-')  # Red lines connecting the corners
    
    plt.show()


# # Countours still contains the contour points

# # Function to fit spline to a set of points
# def fit_spline(points):
#     points = np.array(points)
#     tck, u = splprep([points[:, 0], points[:, 1]], s=0)  # s is the smoothing factor
#     spline_points = splev(u, tck)
#     return np.array(spline_points).T


# # Function to check if a point is near the top side of the bounding box
# def is_near_top(point, box, threshold=60):
#     return point[1] <= box[:, 1].min() + threshold


# # Function to check if a point is near the bottom side of the bounding box
# def is_near_bottom(point, box, threshold=60):
#     return point[1] >= box[:, 1].max() - threshold


# # Function to check if a point is near the left side of the bounding box
# def is_near_left(point, box, threshold=60):
#     return point[0] <= box[:, 0].min() + threshold


# # Function to check if a point is near the right side of the bounding box
# def is_near_right(point, box, threshold=60):
#     return point[0] >= box[:, 0].max() - threshold



# for idx, cnt in enumerate(contours):
#     # Only process large enough contours to filter out noise
#     if cv2.contourArea(cnt) > 5000:
#         # Step 3: Detect the bounding box and four sides
#         rect = cv2.minAreaRect(cnt)
#         box = cv2.boxPoints(rect)
#         box = np.int32(box)
#         print(f"Contour {idx + 1}: Bounding box: {box}")

#         # Step 4: Split the contour into four sides
#         top_side = []
#         bottom_side = []
#         left_side = []
#         right_side = []

#         for point in cnt:
#             point = point[0]  # Flatten the point
#             if is_near_top(point, box):
#                 top_side.append(point)
#             elif is_near_bottom(point, box):
#                 bottom_side.append(point)
#             elif is_near_left(point, box):
#                 left_side.append(point)
#             elif is_near_right(point, box):
#                 right_side.append(point)

#         # Step 5: Fit splines for each side
#         top_spline = fit_spline(top_side)
#         bottom_spline = fit_spline(bottom_side)
#         left_spline = fit_spline(left_side)
#         right_spline = fit_spline(right_side)

#         # Combine splines into a complete piece outline
#         full_spline = np.vstack([top_spline, right_spline, bottom_spline[::-1], left_spline[::-1]])

#         # Step 6: Plot the original contour and the fitted spline
#         plt.figure()
#         plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         plt.plot(full_spline[:, 0], full_spline[:, 1], 'r-', label='Fitted Spline')
#         plt.plot(cnt[:, 0, 0], cnt[:, 0, 1], 'b--', label='Original Contour')
#         plt.title(f'Puzzle Piece {idx + 1}')
#         plt.legend()
#         plt.axis('off')
#         plt.show()

#         print(f'Puzzle Piece {idx + 1} detected and spline fitted.')