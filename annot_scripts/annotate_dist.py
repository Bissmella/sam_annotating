

from pycocotools.coco import COCO
from os.path import join
import json
import cv2
from torchvision.ops import box_iou, box_convert
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from pycocotools import mask as maskUtils
from skimage import measure
import numpy as np
from tqdm import tqdm
import argparse



from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets


import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import models
import torch.distributed as dist




LOCAL_RANK = int(os.environ['LOCAL_RANK'])
WORLD_SIZE = int(os.environ['WORLD_SIZE'])
WORLD_RANK = int(os.environ['RANK'])


class CustomCocoDataset(Dataset):
    def __init__(self, root, annFile, transform=None):
        """
        Args:
            root (string): Directory with all the images.
            annFile (string): Path to the json annotation file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        img = self.coco.loadImgs(img_id)[0]
        image_path = join(self.root, img['file_name'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        data={'image':image,
                'anns': anns,
                'img_id': img_id}
        return data


def get_dataloader(batch_size, rank, world_size, root, annFile):

    dataset = CustomCocoDataset(root=root, annFile=annFile)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=1, collate_fn= lambda x: x)
    return dataloader





def setup(rank, world_size):
    # os.environ['MASTER_ADDR'] = '127.0.0.1'
    # os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("nccl", rank=WORLD_RANK, world_size=WORLD_SIZE)
    #dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()

def union(old_boxes, new_boxes):
    iou_threshold = 0.55
    boxes1 = torch.tensor([ann['bbox'] for ann in old_boxes])
    boxes2 = torch.tensor([ann['bbox'] for ann in new_boxes])
    boxes1 = box_convert(boxes1, "xywh", "xyxy")
    boxes2 = box_convert(boxes2, "xywh", "xyxy")
    iou = box_iou(boxes2, boxes1)
    max_iou, _ = iou.max(dim=1)
    keep = max_iou < iou_threshold
    return keep.tolist()##boxes2[keep]

def inference(rank, world_size, dataloader):
    setup(rank, world_size)

    area_threshold = 250
    
    # Move model to the corresponding GPU and wrap with DDP
    sam_checkpoint = "weights/sam_vit_b_01ec64.pth"##"sam_vit_h_4b8939.pth"
    model_type = "vit_b"

    device = torch.device(f"cuda:{LOCAL_RANK}")

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    model = sam
    model = model.to(device)
    ddp_model = DDP(model, device_ids=[rank])
    mask_generator = SamAutomaticMaskGenerator(
        model=model,
        points_per_side=64,
        points_per_batch=256,
        pred_iou_thresh=0.60,  #86
        stability_score_thresh=0.85,   #0.92
        crop_n_layers=0,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=400,  # Requires open-cv to run post-processing
    )
    result = []
    
    with torch.no_grad():
        print(f"len of loader {rank}: ", len(dataloader))
        for batch_idx, data in enumerate(dataloader):
            data= data[0]
            image = data['image']
            anns = data['anns']
            iid = data['img_id']
            masks = mask_generator.generate(image)
            masks = [mask for mask in masks if mask['area'] >= area_threshold]

            masks = [mask for mask in masks if mask['bbox'][2] <= 509 and mask['bbox'][3] <= 509]
            #keeps = union(anns, masks)
            #masks = masks[keep]
            #masks = [item for item, keep in zip(masks, keeps) if keep]
            for mask in masks:
            # Remove unnecessary keys
                for key in ['crop_box', 'stability_score', 'point_coords', 'predicted_iou']:
                    mask.pop(key, None)
                segment = maskUtils.encode(mask['segmentation'])
                segment['counts'] = segment['counts'].decode('ascii')
                ##contours = measure.find_contours(mask['segmentation'], 0.5)

                # Add new keys
                
                mask.update({
                    'image_id': iid,
                    'category_id': -1,
                    'iscrowd': 0,
                    'segmentation': [],
                    'id': -1
                })
            
                # Process contours
                
                mask['segmentation'] = segment##[np.flip(contour, axis=1).ravel().tolist() for contour in contours]
            #print(f'{rank}: one iteration done')
            #result.extend(anns)
            result.extend(masks)
    joint_dataset = {
        'annotations':result
    }
    with open(f"/home/bibahaduri/dota_dataset/coco/annotations/jointinstances_test2017_{WORLD_RANK}.json", 'w') as f:
        json.dump(joint_dataset, f) 
    

    cleanup()
    #return result

            



if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--local-rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument("--backend", type=str, default="nccl", choices=['nccl', 'gloo'])
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    annFile = "/home/bibahaduri/dota_dataset/coco/annotations/instances_test2017.json"
    dataDir = "/home/bibahaduri/dota_dataset/coco/test2017"
    jointAnnFile = "/home/bibahaduri/dota_dataset/coco/annotations/jointinstances_test2017.json"
    #coco = COCO(annFile)




    # sam_checkpoint = "weights/sam_vit_b_01ec64.pth"##"sam_vit_h_4b8939.pth"
    # model_type = "vit_b"

    # device = "cuda"

    # sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    

    area_threshold = 250  # area threshold for removing small bounding boxes from SAM outputs
    iou_threshold = 0.55  # iou threshold for keeping the SAM generated bbox after comparing with existing annotations


    # manager = mp.Manager()
    # results = manager.list()
    #event = mp.Event()


    results = []

    # Spawn processes for each GPU
    # processes = []
    # for rank in range(world_size):
    dataloader = get_dataloader(batch_size=1, rank=WORLD_RANK, world_size=WORLD_SIZE, root = dataDir, annFile=annFile)
    inference(WORLD_RANK, WORLD_SIZE, dataloader)
    #     p = mp.Process(target=inference, args=(rank, world_size, dataloader))
    #     p.start()
    #     processes.append(p)

    # for p in processes:
    #     p.join()
        
   

    

    # # Collect results from all processes
    # print("Inference completed on all GPUs")
    # with open(annFile,'r') as load_f:
    #     dataset = json.load(load_f)
    # print(dataset.keys())
    # save_info = dataset['info']
    # save_licenses = dataset['licenses']
    # save_images = dataset['images']
    # save_categories = dataset['categories']
    # save_annotations = dataset['annotations']
    # # Save or process results as needed
    # joint_annotations = results
    # joint_dataset = {
    #     'info': save_info,
    #     'licenses': save_licenses,
    #     'images': save_images,
    #     'annotations': joint_annotations,
    #     'categories': save_categories}
    # with open(jointAnnFile, 'w') as f:
    #     json.dump(joint_dataset, f) 

    # print("Total number of boxes is ", len(joint_annotations))


