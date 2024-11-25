import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import splprep, splev

def fit_spline(points, axis=None):
    points = np.array(points)
    if axis is not None:
        points = points.reshape(-1, 2)
    
    # If we have too few points, use a lower degree spline
    k = min(3, len(points) - 1)  # k must be less than number of points
    if k < 1:
        return points  # Return original points if we can't fit a spline
        
    tck, u = splprep([points[:, 0], points[:, 1]], s=0, k=k)  # s=0 for no smoothing, k=degree
    u_new = np.linspace(0, 1, 100)  # Increase number of points for smoother curve
    spline_points = splev(u_new, tck)
    return np.array(spline_points).T

def get_splines(corners, contour, idx, component):
            # Find closest contour points to corners
        corner_indices = []
        contour_points = contour[:, 0, :]  # Extract x,y coordinates from contour
        for corner in corners:
            # Calculate distances to all contour points at once
            distances = np.sqrt(np.sum((contour_points - corner) ** 2, axis=1))
            min_dist_index = np.argmin(distances)
            corner_indices.append(min_dist_index)
            print(f"Distances to corner {corner}: {distances[min_dist_index]}")
            
        corner_indices = np.array(corner_indices)
        
        # Sort indices to ensure proper ordering
        corner_indices.sort()
        
        # Extract contour segments and fit splines
        splines = []
        for i in range(len(corner_indices)):
            start_idx = corner_indices[i]
            end_idx = corner_indices[(i + 1) % len(corner_indices)]
            
            if end_idx < start_idx:
                segment = np.vstack((contour[start_idx:], contour[:end_idx+1]))
            else:
                segment = contour[start_idx:end_idx+1]
                
            if len(segment) > 1:
                spline = fit_spline(segment, axis=0)
                splines.append(spline / component['width'])
        
        # Plot the splines
        plt.figure(figsize=(10, 10))
        for i, spline in enumerate(splines):
            plt.plot(spline[:, 0], spline[:, 1], label=f'Spline {i+1}')
        
        # Plot corners
        corner_points = corners / component['width']
        plt.scatter(corner_points[:, 0], corner_points[:, 1], color='red', s=50, label='Corners')
        
        plt.legend()
        plt.title(f'Splines and Corners of Component {idx + 1}')
        plt.axis('equal')
        plt.grid(True)
        plt.show()

        # Rotate splines so that first and last point are on a vertical line
        new_splines = []
        for spline in splines:
            # Calculate angle between first and last point
            angle = np.arctan2(spline[-1, 1] - spline[0, 1], spline[-1, 0] - spline[0, 0])
            # Create rotation matrix
            rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            # Rotate the spline
            new_spline = np.dot(spline, rotation_matrix)
            new_splines.append(new_spline)
        
        # Plot new splines
        fig, ax = plt.subplots(ncols=len(splines), figsize=(10, 10))
        for i, (ax_, spline) in enumerate(zip(ax, new_splines)):
            ax_.plot(spline[:, 0], spline[:, 1], label=f'Spline {i+1}')
            ax_.scatter(spline[[0, -1], 0], spline[[0, -1], 1], color='red', s=50, label='Corners')
            ax_.legend()
            ax_.set_title(f'Spline {i+1}')
            ax_.axis('equal')
            ax_.grid(True)
        plt.show()
   
        # Plot all splines on top of each other
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Center and align each spline
        aligned_splines = []
        for spline in new_splines:
            # Center the spline at origin
            centered = spline - spline[0]
            # Scale to normalize length
            length = np.sqrt(np.sum((centered[-1] - centered[0])**2))
            normalized = centered / length if length > 0 else centered
            aligned_splines.append(normalized)
        
        # Plot aligned splines
        for i, spline in enumerate(aligned_splines):
            ax.plot(spline[:, 0], spline[:, 1], label=f'Spline {i+1}', alpha=0.7)
            print(f"Length of spline {i+1}: {np.sum(np.sqrt(np.sum((spline[1:] - spline[:-1])**2, axis=1)))}")
            # print(spline)
        
        ax.legend()
        ax.set_title('Aligned Splines Comparison')
        ax.axis('equal')
        ax.grid(True)
        plt.show()

        return splines