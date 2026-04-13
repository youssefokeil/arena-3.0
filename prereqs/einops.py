import math
import os
import sys
from pathlib import Path
import einops
import numpy as np
import torch as t
from torch import Tensor
from PIL import Image
## shift to prereqs, adds to the directories that python can look in
sys.path.append('./prereqs')
from tests_einops import assert_all_equal,assert_all_close


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

### 6th excercise -- split channels, let it b2 (internal) next to w fill rows first
arr_6 = einops.rearrange(arr, "(b1 b2) c h w -> c (b1 h) (b2 w)", b2=3)
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

## A1-rearrange
def rearrange_1() -> Tensor:
    """Return the following tensor using only t.arange and einops.rearrange:

    [[3, 4],
     [5, 6],
     [7, 8]]
    """
    arr_1 = t.arange(3,9)
    rearranged = einops.rearrange(arr_1, "(b1 b2) -> b1 b2", b2=2 )
    return rearranged

expected = t.tensor([[3, 4], [5, 6], [7, 8]])
assert_all_equal(rearrange_1(), expected)

## A2-rearrange
def rearrange_2() -> Tensor:
    """Return the following tensor using only t.arange and einops.rearrange:

    [[1, 2, 3],
     [4, 5, 6]]
    """
    arr = t.arange(1,7)
    arr = einops.rearrange(arr, "(b1 b2) -> b1 b2", b2=3)
    return arr
    
assert_all_equal(rearrange_2(), t.tensor([[1, 2, 3], [4, 5, 6]]))

## b1 - temperature average
def temperatures_average(temps: Tensor) -> Tensor:
    """Return the average temperature for each week.

    temps: a 1D temperature containing temperatures for each day.
    Length will be a multiple of 7 and the first 7 days are for the first week, second 7 days for the second week, etc.

    You can do this with a single call to reduce.
    """
    assert len(temps) % 7 == 0
    arr = einops.reduce(temps, "(b 7) -> b", "mean")
    return arr

temps = t.tensor([71, 72, 70, 75, 71, 72, 70, 75, 80, 85, 80, 78, 72, 83]).float()
expected = [71.571, 79.0]
assert_all_close(temperatures_average(temps), t.tensor(expected))

## b2 - temperatures difference
def temperatures_differences(temps: Tensor) -> Tensor:
    """For each day, subtract the average for the week the day belongs to.

    temps: as above
    """
    assert len(temps) % 7 == 0
    ## make use of broadcasting by first rearranging temperatures
    temperatures_rearranged = einops.rearrange(temps, "(h w) -> h w", w=7)
    temperatures_average = einops.reduce(temps, "(h 7) -> h", "mean").unsqueeze(1)
    temperatures_differences = temperatures_rearranged - temperatures_average
    temperatures_differences = einops.rearrange(temperatures_differences, "h w -> (h w)", w=7)
    return temperatures_differences


expected = [-0.571, 0.429, -1.571, 3.429, -0.571, 0.429, -1.571, -4.0, 1.0, 6.0, 1.0, -1.0, -7.0, 4.0]
actual = temperatures_differences(temps)
assert_all_close(actual, t.tensor(expected))

## b3 - temperature normalized
def temperatures_normalized(temps: Tensor) -> Tensor:
    """For each day, subtract the weekly average and divide by the weekly standard deviation.

    temps: as above

    Pass t.std to reduce.
    """
    avg = einops.reduce(temps,"(h 7) -> h","mean")
    std = einops.reduce(temps,"(h 7) -> h",t.std)
    temps_normalized = (temps - einops.repeat(avg, "h -> (h 7)"))/einops.repeat(std, "w -> (w 7)")
    return temps_normalized

expected = [-0.333, 0.249, -0.915, 1.995, -0.333, 0.249, -0.915, -0.894, 0.224, 1.342, 0.224, -0.224, -1.565, 0.894]
actual = temperatures_normalized(temps)
assert_all_close(actual, t.tensor(expected))

## C1 - Normalize matrix
def normalize_rows(matrix: Tensor) -> Tensor:
    """Normalize each row of the given 2D matrix.

    matrix: a 2D tensor of shape (m, n).

    Returns: a tensor of the same shape where each row is divided by its l2 norm.
    """
    l2_norm = t.norm(matrix, dim =1 ,keepdim=True)
    return matrix/l2_norm



matrix = t.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).float()
expected = t.tensor([[0.267, 0.535, 0.802], [0.456, 0.570, 0.684], [0.503, 0.574, 0.646]])
assert_all_close(normalize_rows(matrix), expected)

## C2 - cosine similarity
def cos_sim_matrix(matrix: Tensor) -> Tensor:
    """Return the cosine similarity matrix for each pair of rows of the given matrix.

    matrix: shape (m, n)
    """
    normalized = matrix/t.norm(matrix, dim=1, keepdim=True)
    return normalized @ normalized.T  


matrix = t.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).float()
expected = t.tensor([[1.0, 0.975, 0.959], [0.975, 1.0, 0.998], [0.959, 0.998, 1.0]])
assert_all_close(cos_sim_matrix(matrix), expected)

## D - sample distribution