import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

image = cv2.imread('/home/bibahaduri/segment-anything/assets/crop_1_0.jpeg')##('/home/bibahaduri/dota_dataset/coco/test2017/P0086.1.0.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam_checkpoint = "weights/sam_vit_b_01ec64.pth"##"sam_vit_h_4b8939.pth"
model_type = "vit_b"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator_2 = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=256,
    pred_iou_thresh=0.86,  #86     0.60*
    stability_score_thresh=0.92,   #0.92               0.85**
    crop_n_layers=0,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=400,  # Requires open-cv to run post-processing
)


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)



area_threshold = 250
masks2 = mask_generator_2.generate(image)
masks2 = [mask for mask in masks2 if mask['area'] >= area_threshold]

masks2 = [mask for mask in masks2 if mask['bbox'][2] <= 509 and mask['bbox'][3] <= 509]

def show_boxes(anns):
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for ann in anns:
        # rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
        box = ann['bbox']
        x0, y0 = box[0], box[1]
        w, h = box[2], box[3] #  - box[0], - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h,
                                fill=False, color='blue', linewidth=3))


plt.figure(figsize=(10,10))
plt.imshow(image)
#show_anns(masks2)
show_boxes(masks2)
plt.axis('off')
plt.savefig('assets/200plotnibox8692.png')
plt.close()