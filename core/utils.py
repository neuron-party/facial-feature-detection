import cv2

def get_coords(img_cv2, fc, tolerance, multi=False):
    gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
    coords = fc.detectMultiScale(gray, 1.2, tolerance, minSize=(60, 60))
    
    if len(coords) == 0:
        return None
    
    if multi:
        return coords
    else:
        if len(coords) == 1:
            return coords[0]
        else:
            max_area = 0
            index = 0
            for i in range(len(coords)):
                _, _, wi, hi = coords[i] 
                area = wi*hi
                if area > max_area:
                    max_area = area
                    index = i
            return coords[index]