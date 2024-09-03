import json
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from tqdm import tqdm



ann_file0 = '/home/bibahaduri/dota_dataset/coco/annotations/jointinstances_test2017_0.json'
ann_file1 = '/home/bibahaduri/dota_dataset/coco/annotations/jointinstances_test2017_1.json'
ann_file2 = '/home/bibahaduri/dota_dataset/coco/annotations/jointinstances_test2017_2.json'
ann_file3 = '/home/bibahaduri/dota_dataset/coco/annotations/jointinstances_test2017_3.json'
annFile = '/home/bibahaduri/dota_dataset/coco/annotations/instances_test2017.json'


with open(ann_file0, 'r') as file:
    annot0 = json.load(file)

with open(ann_file1, 'r') as file:
    annot1 = json.load(file)

with open(ann_file2, 'r') as file:
    annot2 = json.load(file)

with open(ann_file3, 'r') as file:
    annot3 = json.load(file)

with open(annFile, 'r') as file:
    annot = json.load(file)


joint_annots = []

for anns in [annot0, annot1, annot2, annot3]:
    joint_annots.extend(anns['annotations'])


with open(annFile,'r') as load_f:
    dataset = json.load(load_f)
    print(dataset.keys())
    save_info = dataset['info']
    save_licenses = dataset['licenses']
    save_images = dataset['images']
    save_categories = dataset['categories']
    save_annotations = dataset['annotations']


#coco = COCO(annFile)
annId = 0##max(coco.getAnnIds()) + 1


for ann in tqdm(joint_annots):
    ann['id'] = annId
    annId = annId + 1

joint_dataset = {
        'info': save_info,
        'licenses': save_licenses,
        'images': save_images,
        'annotations': joint_annots,
        'categories': save_categories}


jointAnnFile = '/home/bibahaduri/dota_dataset/coco/annotations/samAnnotated_test2017.json'
with open(jointAnnFile, 'w') as f:
        json.dump(joint_dataset, f) 

print("Total number of boxes is ", len(joint_annots))
