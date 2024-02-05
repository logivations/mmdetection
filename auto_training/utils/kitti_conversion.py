import os
from dataclasses import dataclass
from typing import List

import cv2

@dataclass
class KittiAnnotation:
    image_name: str
    image_path: str
    annot_name: str
    annot_path: str

@dataclass
class COCOAnnotatation:
    image_id: int
    category_id: int
    bbox: List[float]
    area: float
    id: int
    category_id: int
    iscrowd: int = 0

@dataclass
class ImageAnnot:
    id: int
    file_name: str
    height: int
    width: int

@dataclass
class CategoryAnnot:
    id: int
    name: str




def read_files(root):
    for path, subdirs, files in os.walk(root):
        for name in files:
            yield os.path.join(path, name)

def decode_file_content(file_content):
    for annot in file_content:
        contents = annot.split(" ")
        bb = [float(a) for a in contents[4:8]]
        label = contents[0]
        yield bb, label

def read_txt(path):
    with open(path, "r") as p:
        return p.read().splitlines()

def is_image(path: str):
    normal_path = path.lower()
    return normal_path.endswith("jpg") or normal_path.endswith("jpeg") or normal_path.endswith("png")

def replace_extention(name, extention):
    return f'{".".join(name.split(".")[:-1])}.{extention}'

def xyxy2xywh(bbox):
    return [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]

def apply_target_class_map(cat: str, target_class_map: dict):
    if cat in target_class_map:
        return target_class_map[cat]
    else:
        return cat

def read_files_make_dict(path):
    files = [file for file in read_files(path)]
    path_dict = {os.path.basename(file): file for file in files}
    return files, path_dict

def convert_kitti_files(train_path, val_path, test_path, target_class_map={}):
    image_ids = 0
    category_ids = 0
    annot_ids = 0
    categories = {}
    cocos = []
    for path in train_path, val_path, test_path:
        image_dict = {}
        annot_dict = {}
        files, path_dict = read_files_make_dict(path)
        for file in files:
            imname = os.path.basename(file)
            txt_name = replace_extention(imname, "txt")
            if is_image(file) and txt_name in path_dict:
                image = cv2.imread(file)
                if not imname in image_dict:
                    image_annot = ImageAnnot(id=image_ids, file_name=imname, height=image.shape[0], width=image.shape[1])
                    image_dict[imname] = image_annot
                    image_ids += 1
                annots = read_kitti_annot(path_dict[txt_name])
                for bb, cat in annots:
                    cat = apply_target_class_map(cat, target_class_map)
                    if cat is None:
                        continue
                    if not cat in categories:
                        category = CategoryAnnot(id=category_ids, name=cat)
                        categories[cat] = category
                        category_ids += 1
                    bbox = xyxy2xywh(bb)
                    area = bbox[2] * bbox[3]
                    annot = COCOAnnotatation(bbox=bbox, area=area, image_id=image_dict[imname].id, id=annot_ids, category_id=categories[cat].id)
                    annot_ids += 1
                    annot_dict[annot.id] = annot
        coco = {}
        coco["images"] = [image.__dict__ for image in image_dict.values()]
        coco["annotations"] = [annot.__dict__ for annot in annot_dict.values()]
        coco["categories"] = [cat.__dict__ for cat in categories.values()]
        cocos.append(coco)
        print(f"{len(coco['annotations'])} annotations\n{len(coco['images'])} images")
    print(f"classes found {coco['categories']}\n" )
    return cocos


def read_kitti_annot(kitti_path):
    file_content = read_txt(kitti_path)
    return [an for an in decode_file_content(file_content)]

