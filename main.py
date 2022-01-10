

# submission
## name:          Freiherr von Arnim, Gustav 
## mail:          gfreiher@uni-muenster.de
## matr.no.:      505 350
## github:        https://github.com/OTI2020/seam_carving_cv_2022
## delivery date: 2022-01-10


# references
## image import:  https://www.delftstack.com/de/howto/python/python-display-image/ 
## energy map:    https://karthikkaranth.me/blog/implementing-seam-carving-with-python/#deleting-the-pixels-from-the-seam-with-the-least-energy
## theorie:       https://trekhleb.dev/blog/2021/content-aware-image-resizing-in-javascript/
## theorie:       https://sso.uni-muenster.de/LearnWeb/learnweb2/pluginfile.php/3451305/mod_assign/introattachment/0/a10-avidan.pdf?forcedownload=1 
## process time:  https://gertingold.github.io/pythonnawi/profiling.html
 

# hint
## please read README.md first


# imports
import time
import cv2
import numpy as np
# from numpy.core.multiarray import array
# from scipy._lib.doccer import doc_replace
from scipy.ndimage.filters import convolve


# start "counting" time 
t_start = time.time()


# helping function to get minimum of tree numbers
def min_of_3(in_1, in_2, in_3):
    min_num = in_1
    if in_2 < min_num and in_2 < in_3:
        min_num = in_2
    elif min_num > in_3:
            min_num = in_3
    return min_num


# helping function to get the index minimum of tree numbers in array
def find_index_of_value(in_value, in_array): #, in_last_index):
    index = 0
    #if in_last_index > 0:
    #    a = in_last_index - 1
    #else:
    #    a = in_last_index
    #if in_last_index < len(in_array):
    #    b = in_last_index + 1
    #else:
    #    b = in_last_index
    try:
        for i in range(0, len(in_array)-1):   #range(a, b):
            if index != 0:
                break
            elif in_value == in_array[i]:
                index = i
        return index
    except:
        print("something went wrong while trying to find index of value")



# user interaction
num_of_seams = input("Geben Sie etwas ein\n")
num_of_seams = int(num_of_seams)
# print(type(num_of_seams))


# read and show original image
img = cv2.imread("test_image_3.jpg", cv2.IMREAD_COLOR)
cv2.imshow('Original-Image', img)
cv2.waitKey(10) # otherwise the image fades out too quickly


# energie function
## 1. find the partial derivative in the x axis
## 2. find the partial derivative in the y axis
## 3. sum their absolute values
def calc_derivation_with_sobel_cernel(input_img):
    # define filter y dimension from top to bottom
    dy_filter = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0]
    ])
    # This converts it from a 2D filter to a 3D filter, replicating the same
    # filter for each channel: R, G, B
    dy_filter = np.stack([dy_filter] * 3, axis=2)

    # define filter x dimension from left to right
    dx_filter = np.array([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0]
    ])
    # convertion from a 2D filter to a 3D filter, 
    # replicating the same filter for each channel: R, G, B
    dx_filter = np.stack([dx_filter] * 3, axis=2)

    input_img = input_img.astype('float32')
    convolved = np.absolute(convolve(input_img, dy_filter)) + np.absolute(convolve(input_img, dx_filter))

    # sum of energies in the red, green, and blue channels
    energy_map = convolved.sum(axis=2)

    # print("energy_map")
    # print(energy_map)
    cv2.imwrite('energie-map-image.jpg', energy_map)
    # cv2.imshow("energie-map-image", energy_map)
    # cv2.waitKey()

    return energy_map


# the minimum number in the lowest row would be the lowest possible seam energy
def calc_seams_energie(input_energy_map):
    # use .shape operator from the cv2 to get the length of y and x
    y_length, x_length = input_energy_map.shape
    min_engy = input_energy_map
    # print(x_length, y_length)
    # print(len(min_engy))
    # print(len(min_engy[0]))

    for i in range(1, y_length):
        for j in range(1, x_length-1):
            min_neighbor_from_top = min_of_3(min_engy[i-1][j-1], min_engy[i-1][j], min_engy[i-1][j+1])
            min_engy[i][j] = min_engy[i][j] + min_neighbor_from_top

    # print("min_engy")
    # print(min_engy)    
    return min_engy 


# find the minimal number on bottom for seam start 
def calc_minimal_seam_start(input_min_engy):
    bottom = len(input_min_engy)-1
    bottom_length = len(input_min_engy[0])-1
    seam_bottom_start = input_min_engy[0][bottom_length]
    
    for i in range(1, bottom_length):
        if input_min_engy[bottom][i] < input_min_engy[bottom][i-1]:
            seam_bottom_start = i
    # print("seam_bottom_start")
    # print(seam_bottom_start)
    return seam_bottom_start


# find minimal seam
## this is dynamic programming
## only three options get checkt in each step
def calc_minimal_seam(input_start_index, input_min_engy):
    seam_length = len(input_min_engy)-1 # length of energy_map is length of seam
    seam_coordinates = []

    # three temporary variables are used: temp_1, temp_2 and temp_3
    temp_1 = input_start_index
    temp_2 = seam_length
    value_of_index = min_of_3(input_min_engy[temp_2][temp_1-1], input_min_engy[temp_2][temp_1], input_min_engy[temp_2][temp_1+1])
    i_row_of_energy_map = input_min_engy[seam_length]
    temp_3 = find_index_of_value(value_of_index, i_row_of_energy_map) #, temp_1+1)

    print(temp_3)
    for i in range(1, seam_length+2):
        seam_coordinates.append([temp_2, temp_3])

        # print(i, temp_2, temp_3)
        value_of_index = min_of_3(input_min_engy[temp_2][temp_1-1], input_min_engy[temp_2][temp_1], input_min_engy[temp_2][temp_1+1])    
        i_row_of_energy_map = input_min_engy[temp_2]
        temp_3 = find_index_of_value(value_of_index, i_row_of_energy_map) #, temp_1)
        #seam_coordinates.append([temp_2, temp_3])
        temp_2 = seam_length-i
        temp_1 = temp_3
    
    # print(seam_coordinates)
    # print(input_min_engy.shape)

    return seam_coordinates


# delete minimal seam
def delete_seam(input_image, input_seam_coordinates):
    image_height, image_width, _ = input_image.shape
    operating_image = input_image
    smaller_img = np.zeros((image_height, image_width-1, 3))

    for i in range(0, len(input_seam_coordinates)-1):
        x_dim = input_seam_coordinates[i][1]
        operating_image[i][x_dim] = -1
        print(i, x_dim, operating_image[i][x_dim])

    for j in range(0, image_height -1):
        l=0
        for k in range(0, image_width -1 ): 
            if np.any(operating_image[j][l] == [255, 255, 255]):
                l+1
            else:
                smaller_img[j][k] = operating_image[j][l]
            l+1


    # print(operating_image)


    cv2.imwrite('smaller_image.jpg', smaller_img)

    return smaller_img



# call functions
def main():
    for i in range(num_of_seams-1):
        step_1 = calc_derivation_with_sobel_cernel(img)
        step_2 = calc_seams_energie(step_1)
        step_3 = calc_minimal_seam_start(step_2)
        step_4 = calc_minimal_seam(step_3, step_2)
        step_5 = delete_seam(img, step_4)
        img = cv2.imread("smaller_image.jpg", cv2.IMREAD_COLOR)
main()


# stop counting time - claculate time differnece
t_ende = time.time()
print("-> processing time:")
print('{:5.3f}s'.format(t_ende-t_start))
