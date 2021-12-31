from django.http import HttpResponse
from django.shortcuts import render

from live_sudoku_solver.sudoku_solver.sudoku_solver.settings import BASE_DIR
from .models import *
from django.core.mail import EmailMessage
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
import cv2
import os
from tensorflow.keras.models import load_model
from .helper import *
from . import solver

import threading
# from .camera import VideoCamera

# model = load_model('C:\\Users\\Suman Saurav\\Desktop\web\\live_sudoku_solver\\sudoku_solver\\main\\ml\\trained_model.h5')
model = load_model(BASE_DIR / 'main/ml/trained_model.h5')

@gzip.gzip_page
def Home(request):
    return StreamingHttpResponse(gen(VideoCamera()), content_type="multipart/x-mixed-replace;boundary=frame")

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()

        heightImg = 450
        widthImg = 450

        frame_rate = 30
        prev = 0

        time_elapsed = time.time() - prev
        img = image


        if time_elapsed > 1. / frame_rate:
            prev = time.time()

            try:
                ### PREPARING THE IMAGE
                img = cv2.resize(img, (widthImg, heightImg))  # RESIZE IMAGE TO MAKE IT A SQUARE IMAGE
                imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  
                imgThreshold = preProcess(img)
                imgContours = img.copy() 
                imgBigContour = img.copy() 
                contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS
                cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3) # DRAW ALL DETECTED CONTOURS
                
                biggest, maxArea = biggestContour(contours) # FIND THE BIGGEST CONTOUR
                if biggest.size != 0:
                    biggest = reorder(biggest)
                    cv2.drawContours(imgBigContour, biggest, -1, (0, 0, 255), 25) # DRAW THE BIGGEST CONTOUR
                    pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
                    pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
                    matrix = cv2.getPerspectiveTransform(pts1, pts2) # GER
                    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
                    imgDetectedDigits = imgBlank.copy()
                    imgWarpColored = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)

                    ### SPLIT THE IMAGE AND FIND EACH DIGIT AVAILABLE

                    imgSolvedDigits = imgBlank.copy()
                    boxes = splitBoxes(imgWarpColored)
                    refine_boxes = clean_squares(boxes)
                    # print(len(boxes))
                    
                    ### PREDICTION
                    numbers = getPredection(refine_boxes, model)
        
                    # print(numbers)
                    imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers, color=(255, 0, 255))
                    numbers = np.asarray(numbers)
                    posArray = np.where(numbers > 0, 0, 1)
                    # print(posArray)

                    ## SOLVING THE BOARD
                    board = np.array_split(numbers,9)
                    if solver.isValid(board):
                        solver.solve(0, 0, board)
                    flatList = []
                    for sublist in board:
                        for item in sublist:
                            flatList.append(item)
                    solvedNumbers =flatList*posArray
                    imgSolvedDigits= displayNumbers(imgSolvedDigits,solvedNumbers)

                    ## OVERLAY SOLUTION
                    pts2 = np.float32(biggest) # PREPARE POINTS FOR WARP
                    pts1 =  np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
                    matrix = cv2.getPerspectiveTransform(pts1, pts2)  # GER
                    imgInvWarpColored = img.copy()
                    imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (widthImg, heightImg))
                    inv_perspective = cv2.addWeighted(imgInvWarpColored, 1, img, 0.5, 1)
                    imgDetectedDigits = drawGrid(imgDetectedDigits)
                    imgSolvedDigits = drawGrid(imgSolvedDigits)

                    imageArray = ([img, imgBigContour],
                                [imgDetectedDigits,inv_perspective])
                    stackedImage = stackImages(imageArray, 1)
                    # cv2.imshow('Stacked Images', stackedImage)
                    ret, jpeg = cv2.imencode('.jpg', inv_perspective)
                    return jpeg.tobytes()

                else:
                    ret, jpeg = cv2.imencode('.jpg', img)
                    return jpeg.tobytes()
                    print("No Sudoku Found")
            except Exception as e:
                print(str(e))

def gen(camera):
    while True:
        try:
            frame = camera.get_frame()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        except:
            pass

def index(request):
    return render(request, 'home.html')