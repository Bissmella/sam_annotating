from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import json
from PIL import Image


ann_file = '/home/bibahaduri/dota_dataset/coco/annotations/jointinstances_train2017_0_cmprs.json'
annFile = '/home/bibahaduri/dota_dataset/coco/annotations/instances_train2017.json'

with open(ann_file, 'r') as file:
    annot = json.load(file)


coco = COCO(annFile)

image_id = annot['annotations'][0]['image_id']
segmentation = annot['annotations'][0]['segmentation']

img_file = coco.loadImgs(image_id)[0]['file_name']
print(img_file)

# rle = maskUtils.frPyObjects(segmentation, coco.loadImgs(image_id)[0]['height'], coco.loadImgs(image_id)[0]['width'])
# rle = maskUtils.merge(rle)
rle = segmentation
mask = maskUtils.decode(rle)

mask_img = Image.fromarray((mask * 255).astype('uint8'))

# Save the mask as a PNG file
mask_img.save('assets/mask_cmprs.png')
breakpoint()


