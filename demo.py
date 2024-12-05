from mmdet.apis import init_detector, inference_detector
import mmcv

# Specify the path to model config and checkpoint file
config_file = './work_dirs/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss.py'
checkpoint_file = './work_dirs/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss/checkpoint.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = 'line_chart_eg9.png'  # or img = mmcv.imread(img), which will only load it once
target_class = 'plot_area'
result = inference_detector(model, img)
# visualize the results in a new window
# model.show_result(img, result)
# or save the visualization results to image files

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
