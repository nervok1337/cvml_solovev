import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.measure import regionprops, label
from skimage.morphology import dilation, footprint_rectangle
from collections import Counter
from pathlib import Path

def extractor_from_props(prop_group, binary):
    prop = max(prop_group, key=lambda p: p.area)

    x0 = min(p.bbox[0] for p in prop_group)
    y0 = min(p.bbox[1] for p in prop_group)
    x1 = max(p.bbox[2] for p in prop_group)
    y1 = max(p.bbox[3] for p in prop_group)

    crop = binary[x0:x1, y0:y1].astype(np.uint8)

    m = cv2.moments(crop, binaryImage=True)
    hu = cv2.HuMoments(m).reshape(-1).astype(np.float32) 

    eccentricity = float(prop.eccentricity)
    count_props = float(len(prop_group))

    binary_after_dilate = dilation(crop.astype(bool), footprint_rectangle((21, 3)))
    count_props_after_dilate = float(len(regionprops(label(binary_after_dilate))))
    cy = prop.centroid[0]
    height = x1 - x0
    norm_cy = (cy - x0) / height

    features = [
        *hu.tolist(),
        eccentricity,
        count_props,
        count_props_after_dilate,
        norm_cy
    ]
    return np.array(features, dtype="f4")

def extractor(image, binary):
    lb = label(binary)
    props = regionprops(lb)

    prop = max(props, key=lambda r: r.area)

    x0, y0, x1, y1 = prop.bbox
    crop = binary[x0:x1, y0:y1].astype(np.uint8)

    m = cv2.moments(crop, binaryImage=True)
    hu = cv2.HuMoments(m).reshape(-1).astype(np.float32)  

    eccentricity = float(prop.eccentricity)
    count_props = float(len(props))

    binary_after_dilate = dilation(crop.astype(bool), footprint_rectangle((21, 3)))
    count_props_after_dilate = float(len(regionprops(label(binary_after_dilate))))
    cy = prop.centroid[0]
    height = x1 - x0
    norm_cy = (cy - x0) / height

    features = [
        *hu.tolist(),
        eccentricity,
        count_props,
        count_props_after_dilate,
        norm_cy
    ]
    return np.array(features, dtype="f4")

def merge_i_props(props):
    merged = []
    index_repeat = None

    for i, p in enumerate(props):
        #print(p.area, p.eccentricity)
        if index_repeat == i:
            continue
        if p.area < 240:
            if props[i-1].eccentricity >= 0.975:
                merged[-1] = [props[i-1], p]
                continue

            if props[i+1].eccentricity >= 0.975:
                merged.append([p, props[i+1]])
                index_repeat = i+1
                continue
        merged.append([p])

    return merged

def detect_spaces(merged_groups):
    boxes = []
    widths = []
    for g in merged_groups:
        x0 = min(p.bbox[0] for p in g)
        y0 = min(p.bbox[1] for p in g)
        x1 = max(p.bbox[2] for p in g)
        y1 = max(p.bbox[3] for p in g)
        boxes.append((x0, y0, x1, y1))
        widths.append(y1 - y0)

    thr = np.mean(widths) / 2

    spaces = []
    prev_max_col = None

    for idx, (_, y0, _, y1) in enumerate(boxes):
        if prev_max_col is not None:
            gap = y0 - prev_max_col
            if gap > thr:
                spaces.append(idx)
        prev_max_col = y1

    return spaces

def edit_tags(tags):
    for tag in tags:
        if len(tag) == 2:
            tags[tags.index(tag)] = tag[1]
    return tags

def make_train(path):
    train = []
    responses = []
    tags = []
    ncls = 0
    for cls in sorted(path.glob("*")):
        ncls+=1
        #print(cls.name, ncls)
        tags.append(cls.name)   
        for p in cls.glob("*.png"):
            #print(p)

            image = imread(p)
            if image.ndim == 2:
                binary = image.astype(bool)
            else:
                gray = np.mean(image, 2).astype("u1")
                binary = gray > 0
            train.append(extractor(image, binary))

            responses.append(ncls)

    train = np.array(train, dtype="f4").reshape(-1,11)
    responses = np.array(responses, dtype="f4").reshape(-1,1)
    return train, responses, tags

def make_predict(image, binary):
    lb = label(binary)
    props = regionprops(lb)
    props.sort(key=lambda p: p.centroid[1])
    
    merged = merge_i_props(props)

    features = [extractor_from_props(g, binary) for g in merged]
    features = np.array(features, dtype="f4").reshape(-1,11)
    spaces = detect_spaces(merged)

    return features, spaces

data = Path("/home/user/Documents/CV/3/task/")

train, responses, tags = make_train(data / "train")
edited_tags = edit_tags(tags)
knn = cv2.ml.KNearest.create()
knn.train(train,cv2.ml.ROW_SAMPLE,responses)

for i in range(0,7):
    image = imread(data / f"{i}.png")

    if image.ndim == 2:
        binary = image.astype(bool)
    else:
        gray = np.mean(image, 2).astype("u1")
        binary = gray > 0

    features, spaces = make_predict(image, binary)

    ret, result, neighbours, dist = knn.findNearest(features, 5)
   
    result_str = ""
    for idx, r in enumerate(result[:, 0]):
        if idx in spaces:
            result_str += " "
        result_str += edited_tags[int(r) - 1]

    print(i, result_str)
