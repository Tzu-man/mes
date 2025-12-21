import numpy as np
from skimage.segmentation import active_contour
from skimage.measure import EllipseModel
from skimage.filters import gaussian

def refine_ellipse_active_contour(z_score_map, initial_yx, init_radius=15):
    """
    Refines detection using Active Contours (Snakes).
    1. Initializes a circle around the user click.
    2. 'Shrinks' the circle until it hits the edges of the Z-Score anomaly.
    3. Fits a perfect geometric ellipse to the final organic shape.
    
    Parameters:
    -----------
    z_score_map : np.ndarray
        The (Patch - Mean) / Std map.
    initial_yx : tuple
        User click (y, x).
    init_radius : int
        Radius of the starting search circle.

    Returns:
    --------
    tuple : (cy, cx, major, minor, angle_deg)
    """
    
    # 1. Pre-process Z-Map
    # Snakes need positive energy/gradients. We take Abs() so dark and bright defects both work.
    # We apply a slight Gaussian blur to help the snake slide smoothly.
    image_smooth = gaussian(np.abs(z_score_map), sigma=1)
    
    # 2. Initialize the Snake (Points along a circle)
    cy, cx = initial_yx
    s = np.linspace(0, 2*np.pi, 100)
    # Create (N, 2) array of coordinates: y = cy + r*sin, x = cx + r*cos
    init_snake = np.array([cy + init_radius*np.sin(s), cx + init_radius*np.cos(s)]).T
    
    # 3. Run Active Contour
    # alpha: Snake length shape (higher = smoother)
    # beta: Snake smoothness (higher = resists bending)
    # w_edge: Attraction to edges (higher = snaps harder)
    snake_points = active_contour(image_smooth, init_snake, 
                                  alpha=0.015, beta=10, w_line=0, w_edge=1)
    
    # 4. Fit an Ellipse to the resulting Snake points
    ell = EllipseModel()
    success = ell.estimate(snake_points)
    
    if not success:
        print("Warning: Active Contour collapsed. Returning default.")
        return (*initial_yx, 10, 10, 0)
        
    # EllipseModel returns: (cy, cx, a, b, theta)
    # Note: scikit-image 'theta' is radians, and axes are semi-axes (radius)
    y_fit, x_fit, a_semi, b_semi, theta_rad = ell.params
    
    # Convert to our standard format (Diameter and Degrees)
    major_axis = max(a_semi, b_semi) * 2
    minor_axis = min(a_semi, b_semi) * 2
    angle_deg = -np.degrees(theta_rad) # Negative to match Matplotlib rotation
    
    return (y_fit, x_fit, major_axis, minor_axis, angle_deg)
