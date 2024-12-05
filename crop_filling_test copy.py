import cv2
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from mmdet.apis import init_detector, inference_detector
import mmcv

# Helper function for corner detection
def detect_best_corners(gray_image, quality_levels, min_distances):
    """Detect corners and return the best set based on least x-difference variance."""
    best_corners = None
    least_variance = float('inf')

    def compute_x_diff_variance(corners):
        if len(corners) < 2:
            return float('inf')
        x_diffs = np.diff(corners[:, 0])
        return np.var(x_diffs)

    for quality_level in quality_levels:
        for min_distance in min_distances:
            corners = cv2.goodFeaturesToTrack(
                gray_image,
                maxCorners=30,
                qualityLevel=quality_level,
                minDistance=min_distance
            )
            if corners is not None:
                corners = np.intp(corners).reshape(-1, 2)
                corners = corners[corners[:, 0].argsort()]  # Sort by x-coordinate
                variance = compute_x_diff_variance(corners)
                if variance < least_variance:
                    least_variance = variance
                    best_corners = corners

    return best_corners

# Main function
def process_image(input_path, model):
    img = mmcv.imread(input_path)
    result = inference_detector(model, img)

    target_class = 'plot_area'
    if target_class in model.CLASSES:
        class_index = model.CLASSES.index(target_class)
        plot_area_bboxes = result[class_index]

        for i, bbox in enumerate(plot_area_bboxes):
            x1, y1, x2, y2, score = bbox
            y1, x1, x2 = y1 - 10, x1 - 4, x2 + 2
            if score >= 0.3:
                cropped_img = img[int(y1):int(y2), int(x1):int(x2)]
                image = cropped_img
                image_height = image.shape[0]
                image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert to HSV
                pixels = image_hsv.reshape(-1, 3)  # Reshape to a list of HSV pixels

                # Perform K-means clustering to group similar colors in HSV space
                num_colors = 15  # Adjust to the desired number of dominant colors
                kmeans = KMeans(n_clusters=num_colors, random_state=42)
                kmeans.fit(pixels)
                dominant_colors = np.round(kmeans.cluster_centers_).astype(int)

                # Convert dominant colors to list of tuples for easier comparison
                dominant_colors = [tuple(color) for color in dominant_colors]

                # Determine background color (usually the most common color)
                labels, counts = np.unique(kmeans.labels_, return_counts=True)
                background_color = tuple(dominant_colors[labels[np.argmax(counts)]])

                # Define tolerances for ignoring background and colors close to black and white in HSV
                hue_tolerance = 2
                saturation_tolerance = 50
                value_black_tolerance = 50
                value_white_tolerance = 300
                hue_difference_tolerance = 2  # New tolerance for similar hues

                # Filter out similar hues, keeping only the one with higher saturation
                filtered_colors = []
                for color in dominant_colors:
                    keep = True
                    for filtered_color in filtered_colors:
                        if abs(color[0] - filtered_color[0]) < hue_difference_tolerance:
                            # If hues are similar, keep only the color with higher saturation
                            if color[1] > filtered_color[1]:
                                filtered_colors.remove(filtered_color)
                                filtered_colors.append(color)
                            keep = False
                            break
                    if keep:
                        filtered_colors.append(color)

                # Convert filtered colors back to numpy array
                dominant_colors = np.array(filtered_colors)
                filtered_dominant_colors = []

                # Initialize the composite image
                composite_image = np.ones_like(cropped_img) * 255  # White background
                filled_img = img.copy()

                # Filter colors and calculate the number of valid lines
                valid_colors = []
                for color in dominant_colors:
                    if (
                        not np.all(np.isclose(color, background_color, atol=[hue_tolerance, saturation_tolerance, 50]))
                        and color[1] > saturation_tolerance
                        and (value_black_tolerance < color[2] < value_white_tolerance)
                    ):
                        valid_colors.append(color)

                num_lines = len(valid_colors)  # Number of lines
                bar_height_factor = 1 / num_lines if num_lines > 0 else 1  # Avoid division by zero

                # Initialize a dictionary to track cumulative heights for grouped x-locations
                cumulative_heights = {}

                # Threshold for grouping nearby x-locations
                x_group_threshold = 10

                def find_nearest_x(x, x_groups):
                    """Find the nearest x in the existing groups within the threshold."""
                    for group_x in x_groups:
                        if abs(group_x - x) <= x_group_threshold:
                            return group_x
                    return x

                for line_idx, color in enumerate(valid_colors):
                    mask = np.all(np.isclose(image_hsv, color, atol=[2, 50, 50]), axis=-1)
                    isolated_image = np.ones_like(image_hsv) * 255
                    isolated_image[mask] = color
                    isolated_image_rgb = cv2.cvtColor(isolated_image, cv2.COLOR_HSV2RGB)
                    gray = cv2.cvtColor(isolated_image_rgb, cv2.COLOR_RGB2GRAY)

                    # Detect best corners
                    quality_levels = np.arange(0.1, 0.6, 0.1)
                    min_distances = np.arange(20, 70, 10)
                    best_corners = detect_best_corners(gray, quality_levels, min_distances)

                    if best_corners is not None:
                        bar_heights = [int(height * bar_height_factor) for height in [cropped_img.shape[0] - y for _, y in best_corners]]
                        bar_positions = [x for x, _ in best_corners]
                        bar_width = 20
                        line_color_rgb = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_HSV2RGB)[0][0]

                    for x, height in zip(bar_positions, bar_heights):
                        # Group x-locations based on the threshold
                        grouped_x = find_nearest_x(x, cumulative_heights.keys())
                        if grouped_x not in cumulative_heights:
                            cumulative_heights[grouped_x] = 0

                        # Determine the top position for this bar based on cumulative height
                        top_y = cropped_img.shape[0] - cumulative_heights[grouped_x] - height
                        bottom_y = cropped_img.shape[0] - cumulative_heights[grouped_x]

                        # Update the cumulative height for this x-position
                        cumulative_heights[grouped_x] += height

                        # Draw the bar with the calculated fixed width
                        top_left = (grouped_x - bar_width // 2, top_y)
                        bottom_right = (grouped_x + bar_width // 2, bottom_y)
                        cv2.rectangle(composite_image, top_left, bottom_right, tuple(map(int, line_color_rgb[::-1])), -1)




                # Fill the composite image back into the blank region
                filled_img[int(y1):int(y2), int(x1):int(x2)] = composite_image

                # Plot the results
                plt.figure(figsize=(15, 5))
                plt.subplot(1, 3, 1)
                plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
                plt.title("Cropped Plot Area")
                plt.axis("off")

                plt.subplot(1, 3, 2)
                plt.imshow(cv2.cvtColor(composite_image, cv2.COLOR_BGR2RGB))
                plt.title("Composite Bar Chart")
                plt.axis("off")
                cv2.imwrite('composite_bar_chart.png', composite_image)

                plt.subplot(1, 3, 3)
                plt.imshow(cv2.cvtColor(filled_img, cv2.COLOR_BGR2RGB))
                plt.title("Final Image with Filled Plot")
                cv2.imwrite('final_bar_chart.png', filled_img)
                plt.axis("off")

                plt.tight_layout()
                plt.show()
    else:
        print(f"Class '{target_class}' not found in the model classes.")

# Initialize the model
config_file = './work_dirs/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss.py'
checkpoint_file = './work_dirs/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss/checkpoint.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# Input image path
input_image_path = 'line_chart_eg7.png'

# Process and plot results
process_image(input_image_path, model)