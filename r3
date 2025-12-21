import cv2
import numpy as np

def refine_ellipse_grabcut(z_score_map, initial_yx, box_size=20):
    """
    Refines detection using GrabCut (Graph Cut Segmentation).
    1. Converts Z-Score map to an image format OpenCV understands.
    2. Runs GrabCut inside a box around the user click.
    3. Fits an ellipse to the segmented foreground mask.
    
    Parameters:
    -----------
    z_score_map : np.ndarray
        The Z-Score map.
    initial_yx : tuple
        User click (y, x).
    box_size : int
        Half-width of the search box (total box = 2*box_size).
        
    Returns:
    --------
    tuple : (cy, cx, major, minor, angle_deg)
    """
    
    # 1. Convert Z-Score to 8-bit Image (Required by OpenCV)
    # We map Z-scores 0 to 6 sigma -> 0 to 255 intensity
    abs_z = np.abs(z_score_map)
    norm_img = np.clip(abs_z / 6.0 * 255, 0, 255).astype(np.uint8)
    # GrabCut requires 3-channel input
    img_bgr = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2BGR)
    
    # 2. Define the Region of Interest (ROI) Rect
    h, w = norm_img.shape
    cy, cx = int(initial_yx[0]), int(initial_yx[1])
    
    x1 = max(0, cx - box_size)
    y1 = max(0, cy - box_size)
    x2 = min(w, cx + box_size)
    y2 = min(h, cy + box_size)
    
    # Rect format: (x, y, w, h)
    rect = (x1, y1, x2-x1, y2-y1)
    
    # 3. Run GrabCut
    mask = np.zeros(norm_img.shape[:2], np.uint8)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    
    try:
        # 5 iterations of graph cut
        cv2.grabCut(img_bgr, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    except Exception as e:
        print(f"GrabCut failed: {e}")
        return (*initial_yx, 10, 10, 0)
    
    # 4. Extract Foreground
    # Mask values: 0=BG, 1=FG, 2=Prob_BG, 3=Prob_FG
    # We treat 1 and 3 as the defect
    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    
    if np.sum(mask2) == 0:
        print("GrabCut found no foreground. Returning default.")
        return (*initial_yx, 10, 10, 0)

    # 5. Fit Ellipse to the Mask
    contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return (*initial_yx, 10, 10, 0)
        
    # Get the largest contour (main defect)
    largest_cnt = max(contours, key=cv2.contourArea)
    
    if len(largest_cnt) < 5:
        # Too small to fit ellipse, return bounding circle
        (x,y), radius = cv2.minEnclosingCircle(largest_cnt)
        return (y, x, radius*2, radius*2, 0)
    
    # fitEllipse returns ((x,y), (major, minor), angle)
    (x_fit, y_fit), (MA, ma), angle = cv2.fitEllipse(largest_cnt)
    
    # Note: OpenCV fitEllipse angle is clockwise. 
    # Also, it sometimes swaps major/minor depending on orientation.
    # We normalize to ensure Major is the long one.
    major_axis = max(MA, ma)
    minor_axis = min(MA, ma)
    
    return (y_fit, x_fit, major_axis, minor_axis, angle)
