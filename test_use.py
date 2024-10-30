import os
import cv2
from mmdet.apis import init_detector, inference_detector
import mmcv
from pathlib import Path

# Specify the path to model config and checkpoint file
config_file = './work_dirs/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss.py'
checkpoint_file = './work_dirs/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss/checkpoint.pth'

# Initialize the model
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# Set the target class name and directories
target_class = 'plot_area'
input_folder = './test_image'
output_folder = './test_output'
os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists

# Loop through each .png file in the input folder
for img_file in Path(input_folder).glob('*.png'):
    # Load the image
    img = mmcv.imread(str(img_file))
    
    # Run inference
    result = inference_detector(model, img)

    # Get the class index for 'plot_area'
    if target_class in model.CLASSES:
        class_index = model.CLASSES.index(target_class)
        plot_area_bboxes = result[class_index]
        
        # Loop through each bounding box for 'plot_area' and crop the area
        for i, bbox in enumerate(plot_area_bboxes):
            x1, y1, x2, y2, score = bbox
            if score >= 0.3:  # Filter out low-confidence detections
                # Crop the bounding box area from the image
                cropped_img = img[int(y1):int(y2), int(x1):int(x2)]
                
                # Define the output file path
                output_path = os.path.join(output_folder, f"{img_file.stem}_plot_area_{i}.jpg")
                
                # Save the cropped image
                mmcv.imwrite(cropped_img, output_path)
                print(f"Cropped image saved at {output_path}")
    else:
        print(f"Class '{target_class}' not found in the model classes.")

