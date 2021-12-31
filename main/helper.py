import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import time



####  Preprocessing Image
def preProcess(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # CONVERT IMAGE TO GRAY SCALE
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 0)  # ADD GAUSSIAN BLUR
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2)  # APPLY ADAPTIVE THRESHOLD
    return imgThreshold

def prepro(img):
    blur = cv2.GaussianBlur(img, (5, 5), 0)

    # threshold it
    # imgThreshold = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

    # invert it so the grid lines and text are white
    inverted = cv2.bitwise_not(thresh, 0)

    # get a rectangle kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # morph it to remove some noise like random dots
    morph = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)

    # dilate to increase border size
    result = cv2.dilate(morph, kernel, iterations=1)
    return result

#### Reorder points for Warp Perspective
def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] =myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew


#### FINDING THE BIGGEST COUNTOUR ASSUING THAT IS THE SUDUKO PUZZLE
def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest,max_area


#### TO SPLIT THE IMAGE INTO 81 DIFFRENT IMAGES
def splitBoxes(img):
    rows = np.vsplit(img,9)
    boxes=[]
    for r in rows:
        cols= np.hsplit(r,9)
        for box in cols:
            h = box.shape[0]
            w = box.shape[1]
            box = box[int(h * 0.05): int(h * 0.95), int(0.05 * w): int(w * 0.95)]
            boxes.append(box)
    return boxes


#### HELPER FUNCTION FOR CLEAN_SQUARES
def clean_helper(img):
    temp = img.copy()
    h = temp.shape[0]
    w = temp.shape[1]
    temp = temp[int(h * 0.15): int(h * 0.85), int(0.15 * w): int(w * 0.85)]
    temp = prepro(temp)
    
    # cv2.imshow('temp', temp)
    if np.isclose(temp, 0).sum() / (temp.shape[0] * temp.shape[1]) >= 0.95:
        return np.zeros_like(img), False

    # if there is very little white in the region around the center, this means we got an edge accidently
    height, width = temp.shape
    mid = width // 2
    if np.isclose(temp[:, int(mid - width * 0.4):int(mid + width * 0.4)], 0).sum() / (2 * width * 0.4 * height) >= 0.90:
        return np.zeros_like(img), False
    return img, True


#### CLEAN THE BOXES AND CHECK IF ANY DIGIT IS PRESENT OF NOT
def clean_squares(squares):
    cleaned_squares = []
    i = 0

    for square in squares:
        new_img, is_number = clean_helper(square)

        if is_number:
            cleaned_squares.append(new_img)
            i += 1

        else:
            cleaned_squares.append(0)

    return cleaned_squares


#### GET PREDECTIONS ON ALL IMAGES
def getPredection(boxes,model):
    result = []
    formatted_squares = []
    location_of_zeroes = set()
    count = 0

    blank_image = np.zeros_like(cv2.resize(boxes[0], (32, 32)))
    start = time.process_time()
    for i in range(len(boxes)):
        if type(boxes[i]) == int:
            count += 1
            location_of_zeroes.add(i)
            formatted_squares.append(blank_image)
        else:
            img = np.asarray(boxes[i])
            img = img[4:img.shape[0] - 4, 4:img.shape[1] -4]
            img = cv2.resize(img, (32, 32))
            img = img / 255
            formatted_squares.append(img)

    formatted_squares = np.array(formatted_squares)
    all_preds = list(map(np.argmax, model(tf.convert_to_tensor(formatted_squares))))
    for i in range(len(all_preds)):
        if i in location_of_zeroes:
            result.append(0)
        else:
            result.append(all_preds[i])
    return result


#### TO DISPLAY THE SOLUTION ON THE IMAGE
def displayNumbers(img,numbers,color = (0,255,0)):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for x in range (0,9):
        for y in range (0,9):
            if numbers[(y*9)+x] != 0 :
                 cv2.putText(img, str(numbers[(y*9)+x]),
                               (x*secW+int(secW/2)-10, int((y+0.8)*secH)), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            2, color, 2, cv2.LINE_AA)
    return img


#### DRAW GRID TO SEE THE WARP PRESPECTIVE EFFICENCY (OPTIONAL)
def drawGrid(img):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for i in range (0,9):
        pt1 = (0,secH*i)
        pt2 = (img.shape[1],secH*i)
        pt3 = (secW * i, 0)
        pt4 = (secW*i,img.shape[0])
        cv2.line(img, pt1, pt2, (255, 255, 0),2)
        cv2.line(img, pt3, pt4, (255, 255, 0),2)
    return img


#### TO STACK ALL THE IMAGES IN ONE WINDOW
def stackImages(imgArray,scale):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    return ver
