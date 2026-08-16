import time
import torch
import cv2
from PIL import Image, ImageDraw
from transformers import Sam3Model, Sam3Processor
import numpy as np
import matplotlib

image_path = "Gate.png"
prompts_to_try = ["vertical black pole"] # "horizontal black pole"
confidence_threshold = 0.5

def overlay_masks(image, masks):
    image = image.convert("RGBA")
    masks = 255 * masks.cpu().numpy().astype(np.uint8)

    n_masks = masks.shape[0]
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(n_masks)]

    for mask, color in zip(masks, colors):
        mask = Image.fromarray(mask)
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha = mask.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)
    return image

def draw_boxes(image, boxes):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(box.tolist(), outline="red", width=3)
    return image

def draw_global_corners(image, combined_boxes):
    draw = ImageDraw.Draw(image)

    # Separate Left and Right Posts Using x_min (Arbitrary)
    sorted_ind = torch.argsort(combined_boxes[:, 0])

    # sort by ascending x_min
    sorted_boxes = combined_boxes[sorted_ind]

    left_post = sorted_boxes[0]
    right_post = sorted_boxes[-1]

    lx_min, ly_min, lx_max, ly_max = left_post.tolist()
    rx_min, ry_min, rx_max, ry_max = right_post.tolist()
    
    # Define the four absolute corners
    corners = [
        ((lx_max + lx_min)/2, ly_max),  # Top-Left
        ((rx_max + rx_min)/2, ry_max),  # Top-Right
        ((lx_max + lx_min)/2, ly_min),  # Bottom-Left
        ((rx_max + rx_min)/2, ry_min),  # Bottom-Right
    ]
    
    # Draw cyan dots at each corner
    r = 16  # Radius of the dot
    for cx, cy in corners:
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill="cyan", outline="white", width=2)
        
    return image

def keep_largest_component(mask_tensor, original_box):
    """
    Helper function for segmentations that identify discontinuous bodies as one object.
    """
    # Convert tensor into a cv2 compatible mask
    mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np)

    # Ignore if there is no separation
    if num_labels <= 1:
        return mask_np, original_box

    # Isolate pixle counts for each detected label
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + np.argmax(areas)

    # Only keep the largest label
    clean_mask_np = (labels == largest_label).astype(np.uint8)
    clean_mask_tensor = torch.from_numpy(clean_mask_np).to(mask_tensor.device)

    x_min = stats[largest_label, cv2.CC_STAT_LEFT]
    y_min = stats[largest_label, cv2.CC_STAT_TOP]
    width = stats[largest_label, cv2.CC_STAT_WIDTH]
    height = stats[largest_label, cv2.CC_STAT_HEIGHT]

    # Draw bounding box
    clean_box = torch.tensor([x_min, y_min, x_min + width, y_min + height], device=mask_tensor.device)

    return clean_mask_tensor, clean_box
    
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")
 
print("Loading model weights into memory...")
load_start = time.time()
 
model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")
 
load_end = time.time()
print(f"Model loaded in: {load_end - load_start:.2f} seconds\n")
 
image = Image.open(image_path).convert("RGB")
 
print(f"Running sequential inference for: {prompts_to_try}...")
inference_start = time.time()

# Setup empty lists to accumulate results across all prompts
all_masks = []
all_boxes = []
all_scores = []

# Loop through each prompt
for prompt_text in prompts_to_try:
    print(f"  -> Processing: '{prompt_text}'")
    inputs = processor(images=image, text=prompt_text, return_tensors="pt").to(device)
     
    with torch.no_grad():
        outputs = model(**inputs)
     
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=confidence_threshold,
        mask_threshold=confidence_threshold,
        target_sizes=[image.size[::-1]], 
    )[0]
    
    # Only append if the model actually found a matching mask for this specific prompt
    if len(results["masks"]) > 0:
        masks = results["masks"]
        boxes = results["boxes"]
        scores = results["scores"]
        
        areas = []
        # Convert tensor masks to numpy arrays for OpenCV
        masks_np = masks.cpu().numpy().astype(np.uint8)
        
        for m in masks_np:
            # Scale 0/1 binary mask to 0/255 for cv2
            m_255 = m * 255
            contours, _ = cv2.findContours(m_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest contour area within this specific mask
                max_area = max([cv2.contourArea(c) for c in contours])
                areas.append(max_area)
            else:
                areas.append(0.0)
                
        # Sort indices by area in descending order and slice the top 2
        top_2_indices = np.argsort(areas)[::-1][:2].tolist()
        
        # Filter the tensors using the top 2 indices
        filtered_masks = masks[top_2_indices]
        filtered_boxes = boxes[top_2_indices]
        filtered_scores = scores[top_2_indices]

        cleaned_masks = []
        cleaned_boxes = []
        
        for mask, box in zip(filtered_masks, filtered_boxes):
            c_mask, c_box = keep_largest_component(mask, box)
            cleaned_masks.append(c_mask)
            cleaned_boxes.append(c_box)
            
        filtered_masks = torch.stack(cleaned_masks)
        filtered_boxes = torch.stack(cleaned_boxes)
        
        all_masks.append(filtered_masks)
        all_boxes.append(filtered_boxes)
        all_scores.append(filtered_scores)
 
inference_end = time.time()
elapsed_time = inference_end - inference_start
 
print("\nInference completed successfully!")
print(f"Total sequential execution time: {elapsed_time:.4f} seconds")

# 4. Combine all collected data and draw
if all_masks:
    combined_masks = torch.cat(all_masks, dim=0)
    combined_boxes = torch.cat(all_boxes, dim=0)
    combined_scores = torch.cat(all_scores, dim=0)
    
    print(f"Found {len(combined_masks)} instance(s) combined (Top 2 per prompt)")
    print(f"Scores: {combined_scores.tolist()}")

    result = overlay_masks(image, combined_masks)
    result = draw_boxes(result, combined_boxes)
    result = draw_global_corners(result, combined_boxes)

    result.save("output.png")
    print("Saved combined visualization to 'output.png'")
else:
    print("No instances found for any of the provided prompts.")