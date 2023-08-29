import cv2 as cv

def zoom_in(image, zoom_level):
    s = image.shape
    zoomed_h = zoom_level * s[0]
    zoomed_w = zoom_level * s[1]
    center_width = int(zoomed_w/2)
    center_height = int(zoomed_h/2)

    image = cv.resize(image,(0,0), fx=zoom_level, fy=zoom_level, interpolation=cv.INTER_LINEAR)

    image = image[int(round(center_height - zoomed_h / zoom_level * .5)): int(round(center_height + zoomed_h / zoom_level * .5)),
          int(round(center_width - zoomed_w / zoom_level * .5)): int(round(center_width + zoomed_w / zoom_level * .5)),
          :]

    return image