import cv2 as cv

def zoom_in(image, zoom_level):
    s = image.shape
    h = s[0]
    w = s[1]
    zoomed_h = zoom_level * h
    zoomed_w = zoom_level * w
    center_width = int(zoomed_w/2)
    center_height = int(zoomed_h/2)

    image = cv.resize(image,(0,0), fx=zoom_level, fy=zoom_level, interpolation=cv.INTER_LINEAR)

    image = image[int(round(center_height - (h/2))) : int(round(center_height + (h/2))),
          int(round(center_width - (w/2))): int(round(center_width + (w/2))),
          :]

    return image
