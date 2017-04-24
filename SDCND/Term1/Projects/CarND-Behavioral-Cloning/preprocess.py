import cv2
import tensorflow as tf

def crop_img(img) :
    return img[50:140,:,:]

def g_blur(img) :
    return cv2.GaussianBlur(img, (3,3), 0)

def resize_img_tf(img, new_size=[66,200]):
    return tf.image.resize_images(img, new_size)

def resize_img_cv(img, new_size=(200,66)):
    return cv2.resize(img, new_size, interpolation = cv2.INTER_AREA)

def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

def preprocess_img(img) :
    # remove unnecessary data
    new_img = crop_img(img)
    new_img = g_blur(new_img)
    new_img = resize_img_cv(new_img)
    return to_rgb(new_img)
