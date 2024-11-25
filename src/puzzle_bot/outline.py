# Import necessary libraries
import cv2
import numpy as np
import svgwrite
import matplotlib.pyplot as plt
import sys
import os

from corners import *
from splines import get_splines


def process_image(image):

    # Set the minimum size (in pixels) for connected components to keep
    min_size = 5000  # Adjust this value based on your image

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
    print(f"Number of connected components: {num_labels}")

    # Create an empty image to store the filtered components
    filtered_edges_inv = np.zeros_like(edges_inv)

    # Iterate through each component and keep those larger than min_size
    for i in range(1, num_labels):  # Start from 1 to skip the background
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_size:
            print(f"Component {i} area: {area}") 
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

    # Filter out small contours
    min_contour_area = min_size  # Use the same minimum size as before
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_contour_area]

    print(f"Number of contours after filtering: {len(contours)}")
    
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
            corners = find_corners_by_approxPoly(component, image_with_contours, eps=0.01)
            if corners is None:
                print(f"No corners found for component {idx + 1}. It will be removed.")
                continue
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

    # Extract 4 splines from the contours
    for idx, component in enumerate(components):
        contour = component['contour']
        corners = component['corners']
        print(f"Component {idx + 1}:")
        
        splines = get_splines(corners, contour, idx, component)
        print(f"Number of splines for component {idx + 1}: {len(splines)}")
        print(f"Splines for component {idx + 1}: {splines}")
        
    
   

if __name__ == "__main__":

    print("Loading the images...")
    if len(sys.argv) < 2:
        print("Usage: %s <image files>" % sys.argv[0])
        sys.exit(-1)

    for f in sys.argv[1:]:
        print(f"Processing image: {f}")
        image = cv2.imread(f)
        if image is None:
            print("Error: Could not load image.")
            exit()
        else:
            process_image(image)
