import PIL
from PIL import Image
import numpy as np
import glob
import os

def factorize(num):
    return [n for n in range(1, num + 1) if num % n == 0]
 
def get_grid_layout(num_samples):
    factors = factorize(num_samples)
    median_factor = factors[len(factors)//2]
    columns = median_factor
    rows = num_samples // columns
    return (rows, columns)
 
def get_image_grid(imgs, layout, mode=0): # make an image grid out of a set of images
    assert len(imgs) == layout[0]*layout[1]
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(layout[1]*w, layout[0]*h))
    grid_w, grid_h = grid.size
    for i, img in enumerate(imgs):
        if mode: grid.paste(img, box=(i//layout[0]*w, i%layout[0]*h))
        else: grid.paste(img, box=(i%layout[1]*w, i//layout[1]*h))
    return grid

file_list = sorted(glob.glob("./*.png"), key=os.path.getmtime)
print(file_list)
img_list = []
for file in file_list:
    img_list.append(Image.open(file))
num_columns = 2
list_size = int(len(file_list)/num_columns+.5)
my_layout = (list_size,num_columns)

#img_list[0].show()

image_grid = get_image_grid(img_list,(my_layout), 1)

image_grid.show()
image_grid.save("tifa/tifa.png")

#h_stack_list = []

#for i in range(len(file_list)-list_size):
#    img1 = Image.open(file_list[i])
#    img2 = Image.open(file_list[i+list_size])
#    h_stack_list.append(np.hstack([img1,img2]))

#Image.open(file_list[0]).show()