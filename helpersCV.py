import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import MiniBatchKMeans
from glob import glob
import os
# Emotion dictionary
ed = {'happy' : 0, 'sad' : 1, 'fear' : 2, 'calm' : 3}
###             Loading and displaying functions

def load_images(path):
    return [cv.resize(cv.cvtColor(cv.imread(file), cv.COLOR_BGR2RGB),(200,200)) for file in glob("{0}/*.jpg".format(path))]

# return a random img from a set of images
def get_random_image(set_of_images):
    return set_of_images[np.random.randint(1, len(set_of_images))]

# Display image source, target and transfer with titles t1, t2, t3   
def display_images(source,target,transfer, t1, t2, t3):
    
    fig2, axs2 = plt.subplots(1, 3 , figsize=(12,12))
    axs2[0].imshow(source)
    axs2[0].set_title(t1)
    axs2[1].imshow(target)
    axs2[1].set_title(t2)
    axs2[2].imshow(transfer)
    axs2[2].set_title(t3)


###             Color related functions

#Extract color pallette  from an img
def extract_palette(img, K=5):
    
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    label = np.zeros_like(img)[:,0]
    
    resized_img =   cv.resize(img,(200,200))
    usable_img = np.float32(resized_img.reshape((-1,3)))
    ret,label,center = cv.kmeans(usable_img,K,None,criteria,3,cv.KMEANS_PP_CENTERS)
    palette = np.uint8(center).reshape((1,K,3))
    
    return palette

# extract mean and std of an image 
def get_mean_and_std(x):
    x_mean, x_std = cv.meanStdDev(x)
    x_mean = np.hstack(np.around(x_mean,2))
    x_std = np.hstack(np.around(x_std,2))
    return x_mean, x_std

#  transfer the color from the image image t to the image s 
def color_transfer(s,t):
    s = cv.cvtColor(s,cv.COLOR_RGB2LAB)
    t = cv.cvtColor(t,cv.COLOR_RGB2LAB)

    s_mean, s_std = get_mean_and_std(s)
    t_mean, t_std = get_mean_and_std(t)

    height, width, channel = s.shape
    for i in range(0,height):
        for j in range(0,width):
            for k in range(0,channel):
                x = s[i,j,k]
                x = ((x-s_mean[k])*(t_std[k]/s_std[k]))+t_mean[k]
                # round or +0.5
                x = round(x)
                # boundary check
                x = 0 if x<0 else x
                x = 255 if x>255 else x
                s[i,j,k] = x

    s_transfer = cv.cvtColor(s,cv.COLOR_LAB2RGB)
    return s_transfer


#Change the opacity of the img by the value (between 0 and 1 )
def change_opacity(img, value):
    return cv.addWeighted(img, value, img, 0, 0)


###              Contrast, colorfulness and brightness related functions



#Takes a RGB image and return its average contrast
def extract_contrast(img):
    lab = cv.cvtColor(img,cv.COLOR_BGR2LAB)

    # separate channels
    L,A,B = cv.split(lab)

    # compute minimum and maximum in 5x5 region using erode and dilate
    kernel = np.ones((5,5),np.uint8)
    min = cv.erode(L,kernel,iterations = 1)
    max = cv.dilate(L,kernel,iterations = 1)

    # convert min and max to floats
    min = min.astype(np.float64) 
    max = max.astype(np.float64) 

    # compute local contrast
    contrast = (max-min)/(max+min)

    # get average across whole image
    average_contrast = 100*np.mean(contrast)
    if np.isnan(average_contrast):
        average_contrast = 0.0
    
    return average_contrast

#takes a RGB image with contrast `initial_contrast` and transforms it to have a `target_contrast` value
def adjust_contrast(img, target_contrast):
    initial_contrast = extract_contrast(cv.cvtColor(img,cv.COLOR_RGB2BGR))
    compensation = target_contrast / initial_contrast
    return cv.addWeighted(img, compensation, img, 0, 0)

# Takes an RGB image an return its colofulness
def extract_colorfulness(image):
    # split the image into its respective RGB components
    (B, G, R) = cv.split(image.astype("float"))

    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))
    
    # combine the mean and standard deviations
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
    
    #The "colorfulness" metric
    return stdRoot + (0.3 * meanRoot)

#takes a RGB image and transforms it to have a `target_colofulness` value  
def adjust_colorfulness(img, target_colorfulness):
    initial_colorfulness = extract_colorfulness(img)
    compensation = target_colorfulness / initial_colorfulness
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hsv_img[:,:,1] = hsv_img[:,:,1] * compensation
    return cv.cvtColor(hsv_img, cv.COLOR_HSV2BGR)

#Takes an RGB image and return its luminance_mean
def extract_luminance(img):
    luminance_array = cv.cvtColor(img,cv.COLOR_RGB2HSV)[:,:,2]
    luminance_mean = luminance_array.mean()
    return luminance_mean

#Takes an RGB image and return a transformed image with target luminance by a rate between 0 and 1 
def adjust_luminance(img,luminance_target, rate):
    img_luminance = extract_luminance(img)
    img = cv.cvtColor(img,cv.COLOR_RGB2HSV)
    diff_lum = luminance_target - img_luminance
    # The marginal diff is the luminance added to each pixel
    marginal_diff = diff_lum * rate
    img[:,:,2] = np.clip(img[:,:,2] + marginal_diff, 0,255)
    modified_img = cv.cvtColor(img,cv.COLOR_HSV2RGB)
    return modified_img

def equalize_rgb(image):
    channels = cv.split(image)
    eq_channels = []
    for ch, color in zip(channels, ['B', 'G', 'R']):
        eq_channels.append(cv.equalizeHist(ch))

    eq_image = cv.merge(eq_channels)
    #eq_image = cv.cvtColor(eq_image, cv.COLOR_BGR2RGB)
    return eq_image

def equalize_hsv(image):
    H, S, V = cv.split(cv.cvtColor(image, cv.COLOR_BGR2HSV))
    eq_V = cv.equalizeHist(V)
    eq_image = cv.merge([H, S, eq_V])
    return cv.cvtColor(eq_image, cv.COLOR_HSV2BGR)

    
# From a list of images, extract a final palette of num_of_colors colors, a avg contrast, colorfulness and luminance
def extract_features(images, num_of_colors):
    contrast_data = []
    colorfulness_data = []
    luminance_data = []
    palette_total = np.array(extract_palette(images[0], num_of_colors)) #initial

    for i, img in enumerate(images):
        palette_total = cv.vconcat([palette_total, extract_palette(img, num_of_colors)])
        contrast_data.append(extract_contrast(img))
        colorfulness_data.append(extract_colorfulness(img))
        luminance_data.append(extract_luminance(img))

    final_palette = extract_palette(palette_total, num_of_colors*2)
    contrast_avg = np.mean(contrast_data)
    colorfulness_avg = np.mean(colorfulness_data)
    luminance_avg = np.mean(luminance_data)
    
    return final_palette, contrast_avg, colorfulness_avg, luminance_avg

def color_transfer_with_weights(source_img, average_palette, random_img, s_weight, t_weight, r_weight):

    if(s_weight + t_weight + r_weight != 1):
        print(s_weight,t_weight,r_weight)
        print("Warning! sum of weights is not 1")
    
    source_transfer_random_img = color_transfer(source_img, extract_palette(random_img, 5))
    transfer_img = color_transfer(source_img, average_palette)

    final_image = change_opacity(source_transfer_random_img, r_weight) + change_opacity(transfer_img, t_weight) + change_opacity(source_img, s_weight)
    
    return final_image

def mix_images(img1, img2, img3, w1, w2, w3):
        return change_opacity(img1, w1) + change_opacity(img2, w2) + change_opacity(img3, w3)

def pipeline(img, emotion, random_image,features, num_of_colors,  show = True): 
    rand_palette, rand_contrast, rand_colorfulness , rand_luminance = extract_features(random_image,num_of_colors)
    # First the color
    step11 = color_transfer(img, features[0][ed[emotion]])
    step12 = color_transfer(img, rand_palette)
    #step1 = mix_images(random_source_img, step11, step12, 1/3, 1/3, 1/3)

    step41 = adjust_luminance(step11, features[3][ed[emotion]], 0.5)
    step42 = adjust_luminance(step12, rand_luminance, 0.5)
    # Then the contrast
    step21 = adjust_contrast(step41, features[1][ed[emotion]])
    step22 = adjust_contrast(step42, rand_contrast)
    #step2 = mix_images(step1, step21, step22, 3/4, 1/8, 1/8)

    # Then the colorfulness
    step31 = adjust_colorfulness(step21, features[2][ed[emotion]])
    step32 = adjust_colorfulness(step22, rand_colorfulness)
    #step3 = mix_images(step2, step31, step32, 1/3, 1/3, 1/3)

    # Then the luminance
    modified_image = mix_images(img, step31, step32, 1/3, 1/3, 1/3)
    if show:
        fig2, axs2 = plt.subplots(1, 6, figsize=(20,20))
        axs2[0].imshow(img)
        axs2[0].set_title('Source image')
        axs2[1].imshow(step11)
        axs2[1].set_title("Color palette adjusted")
        axs2[2].imshow(step21)
        axs2[2].set_title('Contrast adjusted')
        axs2[3].imshow(step31)
        axs2[3].set_title('Colorfulness adjusted')
        axs2[4].imshow(step41)
        axs2[4].set_title('Luminance adjusted')
        axs2[5].imshow(modified_image)
        axs2[5].set_title('Blended with the random image')       

        fig2.show()

    return modified_image
