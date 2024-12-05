import os
import cv2
from mmdet.apis import init_detector, inference_detector
import mmcv

def crop_plot_area(input_file_path):

    # Specify the path to model config and checkpoint file
    config_file = './work_dirs/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss.py'
    checkpoint_file = './work_dirs/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss/checkpoint.pth'

    # Initialize the model
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    # Specify the input image file and target class
    input_file = input_file_path  # Replace with your input file path
    target_class = 'plot_area'

    # Load the image
    img = mmcv.imread(input_file)

    # Run inference
    result = inference_detector(model, img)

    # Get the class index for 'plot_area'
    if target_class in model.CLASSES:
        class_index = model.CLASSES.index(target_class)
        plot_area_bboxes = result[class_index]
        assert len(plot_area_bboxes) == 1
        
        # Loop through each bounding box for 'plot_area' and crop the area
        for i, bbox in enumerate(plot_area_bboxes):
            x1, y1, x2, y2, score = bbox
            if score >= 0.3:  # Filter out low-confidence detections
                # Crop the bounding box area from the image
                cropped_img = img[int(y1):int(y2), int(x1):int(x2)]
                
                # Save the cropped image
                mmcv.imwrite(cropped_img, "plot_area.png")
    else:
        print(f"Class '{target_class}' not found in the model classes.")


crop_plot_area('line_chart_eg9.png')