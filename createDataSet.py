'''
@author: Zhihua Cai
'''

import matplotlib
# search the text label

import numpy as np
import os
import pdb
from PIL import Image
import cv2

from matplotlib.pyplot import imsave

workFolder = os.path.abspath(".")
workFolder = os.path.join(workFolder, "datasets", "birds")
new_size = (32, 32)


def get_img(image_file_path):
    im = cv2.imread(image_file_path)
    res = cv2.resize(im, dsize=new_size)
    
    # (32,32,3)->(3,32,32)
    res = res.transpose(2, 0, 1)
    return res
    

def get_match_img(text_file_path):
    image_file_path = text_file_path.replace("text", os.path.join("CUB_200_2011", "images")).replace(".txt", ".jpg")
    image_file_path = os.path.join(workFolder, image_file_path)
    
    img = get_img(image_file_path)
    return img


def get_caption(text_file_path):
    try:
        f = None
        filepath = os.path.join(workFolder, text_file_path) 
        f = open(filepath, "r")
        t = f.readline()
        return t
        
    finally:
        if f is not None:
            f.close()


def get_label(text_file_path):
    filename = os.path.join(workFolder, text_file_path)
    parentFolder = os.path.split(os.path.split(filename)[0])[1]
    label = int(parentFolder[:3])
    return label


if __name__ == "__main__":
    labels = []
    captions = []
    imgs = []
    
    all_data = []
    for pathi in os.listdir(os.path.join(workFolder, "text")):
        
        for filename in os.listdir(os.path.join(workFolder, "text", pathi)):
            label = int(pathi[:3])
            sep = os.sep
            text_file_path = sep.join(["text", pathi, filename])
            all_data.append(text_file_path)
    
    images = len(all_data)
    image_ids = np.arange(images)        
    np.random.shuffle(image_ids)
    
    block_id = 0
    block_size = 1000
    image_cnt = 0
    
    for image_id in image_ids:        
        text_file_path = all_data[image_id]
            
        caption = get_caption(text_file_path)
        image_file_path = get_match_img(text_file_path)
        labels.append(label)
        captions.append(caption)
        
        imgs.append(image_file_path)
        
        if (image_cnt + 1) % block_size == 0:
            # save data
            np.savez("train_bird_blk_%d.npz" % block_id,
                     imgs=imgs, labels=labels, captions=captions)
            labels = []
            imgs = []
            captions = []
            block_id += 1
            print("created %s" % ("train_bird_blk_%d.npz" % block_id))
                
        image_cnt += 1
    
    # rest of the data used for validation    
    np.savez("val_bird.npz" , imgs=imgs, labels=labels, captions=captions)
    print("created val_bird.npz")
                        
    print("blocks: {}".format(block_id))
    print("images: {}".format(image_cnt))
    print("Done")        
    
    # save resized images to calculate fid
    for i, img in enumerate(imgs):
        img = img.transpose(1, 2, 0)
        imsave("img_out/%d.jpg" % i, img)
