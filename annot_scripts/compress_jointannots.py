import json
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from tqdm import tqdm



ann_file = '/home/bibahaduri/dota_dataset/coco/annotations/jointinstances_train2017_3.json'
annFile = '/home/bibahaduri/dota_dataset/coco/annotations/instances_train2017.json'
with open(ann_file, 'r') as file:
    annot = json.load(file)


coco = COCO(annFile)

for ann in tqdm(annot['annotations']):
    # image_id = ann['image_id']
    # segmentation = ann['segmentation']

    # height = coco.loadImgs(image_id)[0]['height']
    # width = coco.loadImgs(image_id)[0]['width']
    #segm = coco.annToRLE(segmentation)
    segm = coco.annToRLE(ann)
    mask = maskUtils.decode(segm)
    segment = maskUtils.encode(mask)
    segment['counts'] = segment['counts'].decode('ascii')

    ann['segmentation'] = segment

#breakpoint()
with open("/home/bibahaduri/dota_dataset/coco/annotations/jointinstances_train2017_3_cmprs.json", 'w') as f:
        json.dump(annot, f) 