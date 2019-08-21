#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 15:51:54 2018

@author: marina
"""
import sys
import cv2
import numpy as np
import os.path
import glob

"""
Main method, executed when initializing the program. Executes the corresponding function according to user input.
@param
@return 
"""
def main():
      global imgMod, imgOrg
      print("Author: Marina Torres \n")
      while(True):
          inp = raw_input('Enter a filename, press "q" to quit, or press "h" for help: ')
          if(inp == "q"):    
              sys.exit(0)
          elif(inp == "h"):
              help()
          else:
       
              if(inp=="a"):
                  imgOrg = featPoints()
              elif(inp=="m"):
                  imgOrg = featPointsMan()
      
          
"""
Loads image from file
@param Name of the file to load
@return Image object in color scale
"""
def getImage(file):
       img = cv2.imread("../data/"+file)
       return img              

"""
Converts image to gray scale
@param Image object in color scale
@return Image object in gray scale
"""
def gray(img):
    img = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return imgGray

"""
Displays the image received as input. Closes the image window when 'q' key is pressed.
@param Image object to display
@return 
"""
def display(img):
    cv2.imshow('Feature detection',img)
    value = cv2.waitKey(50)&0xff
    if(value == ord('q')):
      cv2.destroyAllWindows()
      return 

"""
Resizes the image received as input to half its size.
@param Image object to resize
@return Image object resized
"""       
def resize(img):
    img = img.copy()
    imgS = cv2.resize(img, (img.shape[1]/2, img.shape[0]/2)) 
    return imgS
 
 """
Automate the extraction of features from chessboard images and write into computer feature files for each of them.
@param 
@return 
"""
def featPoints():
    # TERM_CRITERIA_EPS: Specified accuracy, stop iterations when reached.
    # TERM_CRITERIA_MAX_ITER: Maximum number of iterations.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    objp = np.zeros((6*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)*30    
    
    images = glob.glob('../data/*.jpg')

    i = 0
    # For each image in a set of chessboard images, extract the inner corners and create features file
    for fname in images:
        objpoints = []
        imgpoints = []
        f1 = ""
        c1 = ""
        c2 = ""
        img = getImage(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        # Locate internal corners of the chessboard image. 
        # @params image, pattern size, corners 
        # @return retval: Boolean that will be true if pattern obtained
        #         corners: array of detected corners
        ret,corners = cv2.findChessboardCorners(gray, (8,6),None)
        if(ret == True):
            objpoints.append(objp)
            # Refine corner locations. 
            # @params image, corners, window size, zero zone, criteria
            cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners)
            
            # Create feature file with corners extracted from object 
            if(os.path.isfile('../files/features.txt')==False):
                f = open("../files/features.txt","a")
                for j in range(0, len(objpoints[0])):
                    for k in range(0, len(objpoints[0][j])):
                        opoint = objpoints[0][j][k]
                        s1 = str(opoint)
                        f1 +=s1+" "
                    f2 = f1 +"\n"
                    f1 = ""
                    f.write(f2)
                f.close()
             
            # Create feature file with corners extracted from image file.
            c = open("../files/features"+str(i)+".txt","a")
            i+=1
            for j in range(0, len(imgpoints[0])):
                for k in range(0, len(imgpoints[0][j][0])):
                    ipoint = imgpoints[0][j][0][k]
                    s2 = str(ipoint)
                    c1 +=s2+" "
                c2 = c1 +"\n"
                c1 = ""
                c.write(c2)
            c.close()
        
        # Render the detected chessboard corners and display chessboard image
        cv2.drawChessboardCorners(img, (8,6), corners,ret)
        display(img)

"""
Extract features manually selected from chessboard images and write them into a text file.
@param 
@return image
"""
def featPointsMan():
    images = glob.glob('../data/*.jpg')
    i = 0
    objp = np.zeros((6*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)*30
    objpoints = []
    
    # For each image in a set of chessboard images, extract the inner corners and create features file for each image
    for fname in images:

        img = getImage(fname)
        inp = raw_input('Please insert coordinates: ')   
        coordX = inp.split()[0]
        coordY = inp.split()[1]  
        print("coords: ", coordX, coordY)  
    
        f1 = ""
    
        if(os.path.isfile('../files/features.txt')==False):
                    f = open("../files/features.txt","a")
                    for j in range(0, len(objpoints[0])):
                        for k in range(0, len(objpoints[0][j])):
                            opoint = objpoints[0][j][k]
                            s1 = str(opoint)
                            f1 +=s1+" "
                        f2 = f1 +"\n"
                        f1 = ""
                        f.write(f2)
                    f.close()
                    
        c = open("../files/features"+str(i)+".txt","a")
        c.write("("+coordX+","+coordY+","+"1"+")" +"-"+"("+coordX+","+coordY+")"+ "\n")
        c.close()
        # Plot a circle around the selected coordinates
        cv2.circle(img, (int(coordX), int(coordY)), 5, (255,0,0), 4)
        
        display(img)
        return img


"""
Display user manual
@param
@return
"""
def help():
    print("This program implements a camera calliration algorithm")
    print("For automatic feature detection, press a")
    print("For manual feature detection, press m")
    
if __name__ == "__main__": main()
