from curses.panel import bottom_panel
import cv2



# watermark synthesis
import os 
import random
import shutil
from PIL import Image, ImageEnhance, ImageDraw
from matplotlib.pyplot import fill
import numpy as np
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import torch.multiprocessing as mp
import gc

cpu_number = mp.cpu_count()
executor = ThreadPoolExecutor(max_workers=cpu_number)



def trans_paste(bg_img,fg_img,mask,box=(0,0)):
    fg_img_trans = Image.new("RGBA",bg_img.size)
    fg_img_trans.paste(fg_img,box,mask=mask)
    new_img = Image.alpha_composite(bg_img,fg_img_trans)
    
    return new_img,fg_img_trans

outdir = 'dataset_un_wm_muaban'
if os.path.isdir(outdir):
    shutil.rmtree(outdir)

wm_classes_mapping = {
    "batdongsan_1": 0,
    "batdongsan_2": 1,
    "chotot": 2,
    "watermark_rever_1": 3,
    "watermark_rever_2": 4,
    "watermark_rever_3": 5,
    "watermark_rever_4": 6,
    "muaban": 7
}
os.mkdir(outdir)
BASE_IMG_DIR = '/disk_local/vypham/dataset/watermark/unwatermarked'
WATERMARK_DIR = '/disk_local/vypham/dataset/watermark/watermarks' #1080 
images = sorted([os.path.join(BASE_IMG_DIR,x) for x in os.listdir(BASE_IMG_DIR) if '.jpg' in x])
# images = images[:20000]
watermarks = sorted([os.path.join(WATERMARK_DIR,x) for x in os.listdir(WATERMARK_DIR) if '.png' in x])
print(watermarks)
# rename all the watermark from replace ' ' to '_'

# random.seed(1280)
random.shuffle(images)
random.shuffle(watermarks)

train_images = images[:int(len(images)*0.9)]
val_images = images[int(len(images)*0.9):]

train_wms = watermarks
val_wms = watermarks

# save all the settings to file
names = ['train','valid']
lists = [train_images,val_images]
dataset = dict(zip(names, lists))

# for name, content in dataset.items():
#     with open(f'{outdir}/{name}.txt','w') as f:
#         f.write("\n".join(content))

# os.mkdir(f'{outdir}/natural')
# print('SAVE ALL THE SETTING')
wm_images = [Image.open(wmg).convert("RGBA") for wmg in watermarks]

img_size_min = 8
img_size_max = 128
wm_per_image = 5

def random_crop(img, n=1):
    w, h = img.size
    img1 = ImageDraw.Draw(img) 
    for i in range(n):
        top_left = (random.randint(0, w), random.randint(0, h))
        bottom_right = (random.randint(top_left[0], w), random.randint(top_left[1], h))
        # img1.rectangle(shape, fill ="# ffff33", outline ="red")
        # print("Shape ", (top_left, bottom_right))
        img1.rectangle((top_left, bottom_right), fill="#ffffff")
    return img
for name, images in dataset.items():
    # if 'train' in name:
    #     continue
    # for each setting, synthesis the watermark
    # for each image, add X(X=6) watermark in differnet position, alpha,
    # save the synthesized image, watermark mask, reshaped mask,
    save_path = f'{outdir}/{name}/'
    os.makedirs(f'{save_path}/images')
    os.makedirs(f'{save_path}/labels')

    def run(img, watermarks, wm_images, wm_classes_mapping):
        im = Image.open(img)
        # im = im.resize((512, 512), Image.BILINEAR)
        imw, imh = im.size



        bboxes = []
        # for idx in range(len(watermarks)):
        im = im.convert('RGBA')
        # watermark = watermarks[idx]
        # wm_class_id = wm_classes_mapping[watermark.rsplit('/', 1)[-1].rsplit('.')[0]]
        # wm_image = wm_images[idx]
        # idx = random.randint(0, len(watermarks) - 1)
        
        for wmg, _wm in zip(watermarks, wm_images):
            wm_class_id = wm_classes_mapping[wmg.rsplit('/', 1)[-1].rsplit('.')[0]]
            wm_img_size = random.randint(img_size_min, img_size_max)
            if imw < wm_img_size or imh < wm_img_size:
                return
            # wm = Image.open(wmg).convert("RGBA") # RGBA
    
            wm = _wm.copy()
            if random.randint(0, 1) == 1:
                # print("crop")
                wm = random_crop(wm, n = 2)
   
            opacity = random.randint(50, 100) / 100

            alpha = wm.split()[3]
            alpha = ImageEnhance.Brightness(alpha).enhance(opacity)
            wm.putalpha(alpha)

            scaled = random.randint(10, 45) / 100
            imrw = int(imw * scaled)
            imrh = int(wm.size[1] * imrw / wm.size[0])
            # imrh = imrh + random.randint(-imrh // 5, imrh // 5)
            
            # wmsize = imrh if imrw > imrh else imrw
            wm = wm.resize((imrw, imrh), Image.BILINEAR)
            w,h = wm.size # new size 

            
            box_left = random.randint(0,imw-w)
            box_upper = random.randint(0,imh-h)
            wmm = wm.copy()
            # wm.putalpha(random.randint(int(255*0.4),int(255*0.8))) # alpha
            
            im,_ = trans_paste(im,wm,wmm,(box_left,box_upper))
            # start = (box_left, box_upper)
            # end = (box_left + w, box_upper + h)
            # draw = ImageDraw.Draw(im)
            # draw.rectangle((start, end))
            # draw.text(start, str(wm_class_id))
            # im = cv2.rectangle(im, start, end, (255, 0, 0), 2)
            
            box = f'{wm_class_id} {(box_left + w / 2) / imw} {(box_upper + h / 2) / imh} {w / imw} {h / imh}\n'
            bboxes.append(box)

        identifier = os.path.basename(img).split('.')[0] + '.' + os.path.basename(wmg).split('.')[1]
            # save 
        im = im.convert('RGB')
        im.save(f'{save_path}/images/{identifier}')
        label_name = os.path.basename(img).rsplit('.', 1)[0] + '.txt'
        with open(f'{save_path}labels/{label_name}', 'w') as f:
            for box in bboxes:
                f.write(box)

            
    
    
    imgs = []
    for img in tqdm(images):
        image_name = img.split('/')[-1]
        imgs.append(executor.submit(run, img=img,  watermarks=watermarks, wm_images=wm_images, wm_classes_mapping=wm_classes_mapping))

    for idx, img in tqdm(enumerate(concurrent.futures.as_completed(imgs))):
        img.result()
        # del imgs[idx]
        gc.collect()
 



