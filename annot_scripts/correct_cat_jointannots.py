import json
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from tqdm import tqdm



ann_file = '/home/bibahaduri/dota_dataset/coco/annotations/samAnnotated_test2017.json'
annFile = '/home/bibahaduri/dota_dataset/coco/annotations/instances_train2017.json'
with open(ann_file, 'r') as file:
    dataset = json.load(file)
    save_info = dataset['info']
    save_licenses = dataset['licenses']
    save_images = dataset['images']
    save_categories = dataset['categories']
    save_annotations = dataset['annotations']


coco = COCO(annFile)

for ann in tqdm(dataset['annotations']):
    # image_id = ann['image_id']
    # segmentation = ann['segmentation']

    # height = coco.loadImgs(image_id)[0]['height']
    # width = coco.loadImgs(image_id)[0]['width']
    #segm = coco.annToRLE(segmentation)
    # segm = coco.annToRLE(ann)
    # mask = maskUtils.decode(segm)
    # segment = maskUtils.encode(mask)
    # segment['counts'] = segment['counts'].decode('ascii')

    ann['category_id'] = 1     # 'category_id': -1,



# joint_dataset = {
#         'info': save_info,
#         'licenses': save_licenses,
#         'images': save_images,
#         'annotations': joint_annots,
#         'categories': save_categories}
#breakpoint()
with open("/home/bibahaduri/dota_dataset/coco/annotations/samAnnotated_test2017_crct.json", 'w') as f:
        json.dump(dataset, f) 