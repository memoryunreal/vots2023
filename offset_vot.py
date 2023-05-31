import os
import sys
sys.path.insert(0,"./")
import vot
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
votdir = "/home/lizhe/paper/vots2023/workspace"
mask_vis = "/home/lizhe/paper/vots2023/mask_vis"
seq_dir = os.path.join(votdir, "sequences")
seqlist = os.listdir(mask_vis)
seqlist.sort()
# seqlist = ["singer3"]
for seq in tqdm(seqlist):
    image = np.asarray(Image.open(os.path.join(seq_dir, seq, "color", "00000001.jpg")))
    image_shape = image.shape[:2]

    masks = os.listdir(os.path.join(mask_vis, seq))
    id = 1
    mask_0 = np.asarray(Image.open(os.path.join(mask_vis, seq, masks[0]))) 
    if mask_0.shape == image_shape:
        continue
    else:
        print(seq)
    for mask in masks:
        mask = np.asarray(Image.open(os.path.join(mask_vis, seq, mask)))
        mask_shape = mask.shape
        pad_x = image_shape[1] - mask_shape[1]
        pad_y = image_shape[0] - mask_shape[0]

        mask_padded = np.pad(mask, ( (0, pad_y), (0, pad_x)), 'constant', constant_values=0)
        Image.fromarray(mask_padded).save(os.path.join(mask_vis, seq, masks[id-1]))
        # mask_padded = Image.fromarray(mask_padded)

        # mask_rgba = Image.new("RGBA", mask_padded.size)
        # for x in range(mask_padded.width):
        #     for y in range(mask_padded.height):
        #         value = mask_padded.getpixel((x,y))
        #         if value == 1:
        #             mask_rgba.putpixel((x,y), (255, 0, 0, 127)) # 红色，半透明
        #         else:
        #             mask_rgba.putpixel((x,y), (0, 0, 0, 0)) # 完全透明

        # # 然后，将 RGB 图像转换为 RGBA
        # image_rgba = Image.fromarray(image).convert("RGBA")

        # # 最后，将两个 RGBA 图像混合在一起
        # blend = Image.alpha_composite(image_rgba, mask_rgba)

        # # 保存混合后的图像
        # blend.save(os.path.join("/home/lizhe/paper/vots2023/mask_064","{}_{}.png".format(seq, id)))
        # mask_padded.save()
        id += 1