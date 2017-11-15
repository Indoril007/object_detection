import os
import random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np

# This parses the text file containing all class names and stores in classes
with open('./clsid_to_name.txt', 'r') as f:
    classes = list(map(lambda l: l.split('\'')[1].split(',')[0].strip(' ,\''), f.readlines()))

def random_image_set(seed, directory, num):
    # Get image filenames from val.txt image set
    with open(os.path.join(directory, 'ImageSets/DET/val.txt')) as f:
        dataset = list(map(lambda f: f.strip().split()[0], f.readlines()))
    
    # Sample random images from dataset
    random.seed(seed)
    sample = random.sample(range(len(dataset)), num)
    image_paths = [os.path.join(directory, 'Data/DET/val', dataset[i] + '.JPEG') for i in sample]
    return image_paths

def get_images(image_paths, display=True):
    # Load and Display images
    imgs = list()
    for img_file in image_paths:
        img = mpimg.imread(img_file)
        area = img.shape[0]*img.shape[1]
        imgs.append({"original": img,
                     "cropped_regions": [],
                     "bboxes": [],
                     "classes": [],
                     "confidences": [],
                     "area": area})
        if display:
            plt.figure(figsize=(9,9)) 
            plt.imshow(img)
    return imgs

def perform_selectivesearch(ss_func, imgs, params, display=True):

    SCALE = params["SCALE"]
    SIGMA = params["SIGMA"]
    MIN_SIZE = params["MIN_SIZE"]
    MIN_REGION = params["MIN_REGION"]
    MAX_RATIO = params["MAX_RATIO"]

    # Perform selective search on each image
    for img in imgs:
        img_lbl, regions = ss_func(img["original"], scale=SCALE, sigma=SIGMA, min_size=MIN_SIZE)
        
        if display: 
            img_copy = img["original"].copy() # image copy for drawing regions and displaying

        for region in regions:
            x, y, w, h = region['rect']
            if (region['size'] < MIN_REGION) or (w / h > MAX_RATIO) or (h / w > MAX_RATIO) or (((w*h)/img["area"]) > 0.75):
                continue
    
            img["bboxes"].append(np.array([x,x+w,y,y+h]))
            
            # axis order and data type changed in preparation for input into CNN model
            img["cropped_regions"].append(np.moveaxis(img["original"][y:y+h,x:x+w,:],2,0).astype(np.float32))

            if display: 
                # for displaying regions (copies are made so as to not leave red borders in original image)
                cv2.rectangle(img_copy, (x, y), (x+w, y+h), 0xFF3333, thickness=2)

        if display:
            plt.figure(figsize=(10,10))
            plt.imshow(img_copy)

def perform_classification(model, imgs):
    for img in imgs:
        img["predictions"] = model.predict(img["cropped_regions"])
        for pred in img["predictions"]:
            img["classes"].append(np.argmax(pred))
            img["confidences"].append(np.max(pred))

def display_detections(imgs, conf_threshold):
    for img in imgs:
        img_copy = img["original"].copy()
        
        for sr in img["selected_regions"]:
            if img["confidences"][sr] < conf_threshold:
               continue
            x1, x2, y1, y2 = img["bboxes"][sr]
            cls = img["classes"][sr]
            rand_color = (random.randint(0,200), random.randint(0,200), random.randint(0,200))
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), rand_color, thickness=2)
            cv2.putText(img_copy, classes[cls], (x1,y1), cv2.FONT_HERSHEY_TRIPLEX, 0.6, rand_color)
        plt.figure(figsize=(10,10))
        plt.imshow(img_copy)

