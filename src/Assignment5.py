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
#from matplotlib import pyplot as plt




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
      
          
  
def getImage(file):
       img = cv2.imread("../data/"+file)
       return img              

def gray(img):
    img = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return imgGray

def display(img):
    cv2.imshow('Feature detection',img)
    value = cv2.waitKey(50)&0xff
    if(value == ord('q')):
      cv2.destroyAllWindows()
      return 

        
def resize(img):
    img = img.copy()
    imgS = cv2.resize(img, (img.shape[1]/2, img.shape[0]/2)) 
    return imgS
 
def featPoints():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    objp = np.zeros((6*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)*30    
    
    images = glob.glob('../data/*.jpg')

    i = 0
    for fname in images:
        objpoints = []
        imgpoints = []
        f1 = ""
        c1 = ""
        c2 = ""
        img = getImage(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        ret,corners = cv2.findChessboardCorners(gray, (8,6),None)
        if(ret == True):
            objpoints.append(objp)
            cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners)
            
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
        
        
       
        
        cv2.drawChessboardCorners(img, (8,6), corners,ret)
        display(img)
     
def onclick(event):
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))
    

def featPointsMan():
    images = glob.glob('../data/*.jpg')
    i = 0
    objp = np.zeros((6*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)*30
    objpoints = []
    
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
            
        cv2.circle(img, (int(coordX), int(coordY)), 5, (255,0,0), 4)
        
        display(img)
        return img
    #except:
    #    print("Please, insert valid coordinates") 
   # return img

def help():
    print("This program implements a camera calliration algorithm")
    print("For automatic feature detection, press a")
    print("For manual feature detection, press m")
    
if __name__ == "__main__": main()
