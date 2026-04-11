import math
import os
import sys
from pathlib import Path
import einops
import numpy as np
import torch as t
from torch import Tensor
from PIL import Image

## i run shift + enter as  an interactive window
arr = np.load(Path("prereqs").joinpath("numbers.npy"))
print(arr[0].shape)
img_arr = Image.fromarray(arr[0].transpose(1, 2, 0))
img_arr.show()

## the monochrome image
img_mono=Image.fromarray(arr[0,0])
img_mono.show()

## let's define a helper function, to make our lives easier
def display_array_as_img(array):
    '''
    Input: numpy array
    Output: displays the image
    Only accepts arrays of ndim 2 or 3
    '''
    if array.ndim == 3:
        img_arr = Image.fromarray(array.transpose(1, 2, 0))
    else:
        img_arr = Image.fromarray(array)
    img_arr.show()


## let's see what the array looks like
print(arr.shape)

### 1st excercise -- column stacking
arr_column_stacked = einops.rearrange(arr, "b c h w -> c (b h) w")
print(arr_column_stacked.shape)
display_array_as_img(arr_column_stacked)

### 2nd excercise -- column stacking and copying
arr_2 = einops.repeat(arr[0], "c h w -> c (repeat h) w", repeat=2)
print(arr_2.shape)
display_array_as_img(arr_2)

### 3rd excercise -- row stacking and double copying
arr_3 = einops.repeat(arr[0:2], "b c h w -> c (b h) (repeat w)", repeat=2)
print(arr_3.shape)
display_array_as_img(arr_3)

### 4th excercise -- stretching
arr_4 = einops.repeat(arr[0], "c h w -> c (h repeat) w", repeat=2)
print(arr_4.shape)
display_array_as_img(arr_4)

### 5th excercise -- split channels
arr_5 = einops.rearrange(arr[0], "c h w -> h (c w)")
print(arr_5.shape)
display_array_as_img(arr_5)

### 6th excercise -- split channels, let it (b2 b1) fill rows first
arr_6 = einops.rearrange(arr, "(b2 b1) c h w -> c (b2 h) (b1 w)", b2=2)
print(arr_6.shape)
display_array_as_img(arr_6)

### 7th excercise -- transpose
arr_7 = einops.rearrange(arr[1], "c h w -> c w h")
print(arr_7.shape)
display_array_as_img(arr_7)

### 8th excercise -- shrinking
arr_8 = einops.reduce(arr, "(b1 b2) c (h h2) (w w2) -> c (b1 h) (b2 w)", "max", h2=2, w2=2,b1=2)
print(arr_8.shape)
display_array_as_img(arr_8)
