#first

# import numpy as np
# import imageio
# import scipy.ndimage
# import cv2

# img = 'lion.jpg'

# def rgb2gray(rgb):
#     return np.dot(rgb[...,:3],[0.2989,0.5870,0.1140])

# def dodge(front,back):
#     final_sketch = front+255/(255-back)
#     final_sketch[final_sketch>255] = 255
#     final_sketch[back==255] = 255
#     return final_sketch.astype('uint8')


# ss = imageio.imread(img)
# gray = rgb2gray(ss)


# i = 255-gray

# blur = scipy.ndimage.filters.gaussian_filter(i,sigma=15)

# r = dodge(blur,gray)

# cv2.imwrite('lion.png',r)




#second
# import cv2

# def convert_to_sketch(image_path):
#     # Load the image
#     image = cv2.imread(image_path)

#     # Convert the image to grayscale
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Invert the grayscale image
#     inverted_image = cv2.bitwise_not(gray_image)

#     # Apply Gaussian blur
#     blurred_image = cv2.GaussianBlur(inverted_image, (111, 111), 0)

#     # Invert the blurred image
#     inverted_blurred_image = cv2.bitwise_not(blurred_image)

#     # Create the sketch image by dividing the grayscale image by the inverted blurred image
#     sketch_image = cv2.divide(gray_image, inverted_blurred_image, scale=200.0)

#     # Show the sketch image
#     cv2.imshow("Sketch Image", sketch_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# # Call the function and provide the path to your image
# convert_to_sketch("test.jpg")



#third

# import cv2

# def convert_to_sketch(image_path):
#     # Read the image
#     image = cv2.imread(image_path)

#     # Convert the image to grayscale
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Apply Gaussian blur
#     blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

#     # Perform edge detection
#     edges = cv2.Canny(blurred_image, 30, 70)

#     # Invert the image
#     inverted_edges = cv2.bitwise_not(edges)

#     # Display the sketch image
#     cv2.imshow("Sketch", inverted_edges)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# # Path to the input image
# image_path = "mana1.jpg"

# # Convert the image to a sketch
# convert_to_sketch(image_path)



#fourth

import cv2

def convert_to_sketch(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Invert the grayscale image
    inverted_image = cv2.bitwise_not(gray_image)

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(inverted_image, (111, 111), 0)

    # Invert the blurred image
    inverted_blurred_image = cv2.bitwise_not(blurred_image)

    # Create the sketch image by dividing the grayscale image by the inverted blurred image
    sketch_image = cv2.divide(gray_image, inverted_blurred_image, scale=256.0)

    # Save the sketch image
    cv2.imwrite(output_path, sketch_image)

    print("Sketch image saved successfully.")

# Call the function and provide the path to your image and output file path
convert_to_sketch("test.jpg", "test_sketch.jpg")
