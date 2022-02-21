from xml.dom import minidom
import cv2
import numpy as np
import csv


class Runway:
    def __init__(self,frame=0,xmin=0,xmax=0,ymin=0,ymax=0):
        self.frame = frame
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.crop_img = None

    @classmethod
    def LoadFromXml(cls,fileName):
        cls.xmin, cls.xmax, cls.ymin, cls.ymax = cls.getBoundingBoxCoordinates(fileName+'.xml')
        cls.frame = cv2.imread(fileName+'.png')
        return cls(cls.frame,cls.xmin,cls.xmax,cls.ymin,cls.ymax)
    
    @classmethod
    def LoadImage(cls,imagePath):
        cls.frame = cv2.imread(imagePath)
        return cls(cls.frame)


    def getBoundingBoxCoordinates(xmlPath):
        doc = minidom.parse(xmlPath)
        return (
            doc.getElementsByTagName('xmin')[0].firstChild.nodeValue,
            doc.getElementsByTagName('xmax')[0].firstChild.nodeValue,
            doc.getElementsByTagName('ymin')[0].firstChild.nodeValue,
            doc.getElementsByTagName('ymax')[0].firstChild.nodeValue
        )
    def roi(self):
        self.crop_img = self.frame[int(self.ymin):int(self.ymax), int(self.xmin):int(self.xmax)]

    def canny(self,t1,t2):
        if  self.crop_img is None:
            self.roi()
        self.cannyArray = cv2.Canny(self.crop_img,t1,t2,None,3)

    def HougLineStandart(self,t1,lineNum = 0 ,r=1,angle=180):
        lines = cv2.HoughLines(self.cannyArray, r, np.pi / angle, t1)
        if lines is not None:
            leftLines,rightLines = [],[]
            lastAlpha = 0

            for i in range(0, len(lines)):
                for rho, theta in lines[i]:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                #cv2.line(self.crop_img, (x1, y1), (x2, y2), (0, 0, 255), 3,cv2.LINE_AA)
                if x2-x1 != 0:
                    alpha = (y2-y1)/(x2-x1)
                    if alpha<15 and alpha>0 and (abs(lastAlpha-alpha)<alpha*0.2):
                        leftLines.append([[x1,y1],[x2,y2]])
                    if alpha>-15 and alpha<0 and (abs(lastAlpha-alpha)<alpha*0.2):
                        rightLines.append([[x1,y1],[x2,y2]])
                    lastAlpha = alpha
            x1,y1,x2,y2 = averageLines(leftLines)
            cv2.line(self.crop_img,(x1,y1),(x2,y2),(50,50,255),3,cv2.LINE_AA)
            x1,y1,x2,y2 = averageLines(rightLines)
            cv2.line(self.crop_img,(x1,y1),(x2,y2),(25,25,255),3,cv2.LINE_AA)
                
                    
        self.frame[int(self.ymin):int(self.ymax), int(self.xmin):int(self.xmax)] = self.crop_img

    def HougLineP(self,t1,r=1,angle=180,minL=20,maxG = 10):
        #linesP = cv2.HoughLinesP(self.cannyArray, r, np.pi / angle, t1, None, minL, maxG)
        linesP = cv2.HoughLinesP(self.cannyArray, r, np.pi / angle, t1, None, int(self.cannyArray.shape[0]*0.75), maxG)
        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv2.line(self.crop_img, (l[0], l[1]), (l[2], l[3]), (255, 0, 0), 3, cv2.LINE_AA)

        self.frame[int(self.ymin):int(self.ymax), int(self.xmin):int(self.xmax)] = self.crop_img
    
    def getROICoord(self):        
        frame_height, frame_width, _ = self.frame.shape
        tfNet = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb','schema.pbtxt')
        tfNet.setInput(cv2.dnn.blobFromImage(self.frame,size=(300,300)))
        outputNet = tfNet.forward()
        for detection in outputNet[0,0,:,:]:
            if detection[2]>0.5: #Confidence
                self.xmin=detection[3]*frame_width
                self.ymin= detection[4]*frame_height
                self.xmax= detection[5]*frame_width
                self.ymax = detection[6]*frame_height
                self.roi()
        return (int(self.xmin),int(self.ymin),int(self.xmax),int(self.ymax))

def averageLines(coord):
    sx1,sy1,sx2,sy2 = 0,0,0,0
    l = len(coord)
    for c in coord:
        sx1 += c[0][0]
        sy1 += c[0][1]
        sx2 += c[1][0]
        sy2 += c[1][1]
    if l != 0:
        return (sx1//l,sy1//l,sx2//l,sy2//l)
    else:
        return (0,0,0,0)


def addNoise(frame, mean, var):
    
    row, col, ch = frame.shape
    
    sigma = var**0.5
    gaussNoise = np.random.normal(mean,sigma,(row,col))
    

    noisy_image = np.zeros(frame.shape, np.float32)

    if len(frame.shape) == 2:
        noisy_image = frame + gaussNoise
    else:
        noisy_image[:, :, 0] = frame[:, :, 0] + gaussNoise
        noisy_image[:, :, 1] = frame[:, :, 1] + gaussNoise
        noisy_image[:, :, 2] = frame[:, :, 2] + gaussNoise

    cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    noisy_image = noisy_image.astype(np.uint8)

    return(noisy_image)

def addBlur(frame, windowSizeX, windowSizeY):
    gaussBlurred = cv2.GaussianBlur(frame,(windowSizeX,windowSizeY),cv2.BORDER_DEFAULT)
    return(gaussBlurred)

class confidenceRowClass:
    def __init__(self, frameKey, originalConfidenceKey, noiseConfidenceKey, blurConfidenceKey): 
        self.frameKey = frameKey 
        self.originalConfidenceKey = originalConfidenceKey
        self.noiseConfidenceKey = noiseConfidenceKey
        self.blurConfidenceKey = blurConfidenceKey


confidencesList = []


if __name__ == '__main__':
    #rw1 =Runway.LoadFromXml('vlcsnap-error355')
    #rw2 = Runway.LoadFromXml('vlcsnap-error355')
    #rw3 = Runway.LoadImage('vlcsnap-error189.png')
    #cv2.rectangle(rw3.frame,rw3.getROICoord(), (0, 0, 255),thickness=3)

    #print(str(rw3.getROICoord()))
    #cv2.imshow('deneme',rw3.frame)
    #cv2.imshow('deme',np.array(np.array(rw3.frame)*1.2,dtype='uint8'))
    #rw1.canny(50,100)
    #rw1.HougLineStandart(50,5)
    #rw2.canny(50,100)
    #rw2.HougLineP(20)
    #cv2.imshow('RW-1 Standart',rw1.frame)
    #cv2.imshow('RW-2 Probability',rw2.frame)
    #cv2.waitKey(0)
    
    
    videoname='video\\dalaman720gray.mkv'
    cap = cv2.VideoCapture(videoname)
    tfNet = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb','schema.pbtxt')
    rw = Runway() # original video
    rw2 = Runway() # white noise video
    rw3 = Runway() # blur video

    kernelSizeH = 5 # The height size of Gaussian kernel used for blurring 
    kernelSizeW = 5 # The width size of Gaussian kernel used for blurring
    variance = 100 

    var = 'Variance of Noise is: ' + str(variance)
    kernel = 'Window Size is: ' + str(kernelSizeH) + ' by ' + str(kernelSizeW)
    

    while(cap.isOpened()):
        frameNumber = cap.get(cv2.CAP_PROP_POS_FRAMES)
        #print(frameNumber, 'frame')
        totalFrames = 'Number of total Frames: ' + str(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        ret,frame =cap.read()
        rw.frame =frame
        frame_height, frame_width, _ = frame.shape
        tfNet.setInput(cv2.dnn.blobFromImage(frame,size=(300,300)))
        outputNet = tfNet.forward()
        
        originalConfidence = outputNet[0,0,0,2]
        #print(originalConfidence, 'org')

        for detection in outputNet[0,0,:,:]:
            if detection[2]>0.8: #Confidence
                rw.xmin=detection[3]*frame_width
                rw.ymin= detection[4]*frame_height
                rw.xmax= detection[5]*frame_width
                rw.ymax = detection[6]*frame_height
                rw.roi()
        if rw.crop_img is not None:
            rw.canny(50,200)
            #rw.HougLineP(20)
            rw.HougLineStandart(50)
            cv2.imshow('Canny',rw.cannyArray)
            cv2.rectangle(rw.frame,(int(rw.xmin),int(rw.ymin)),(int(rw.xmax),int(rw.ymax)), (0, 0, 255),thickness=3)
        cv2.imshow(videoname, rw.frame)

        whiteNoiseFrame = addNoise(frame, 0, variance)
        rw2.frame = whiteNoiseFrame
        noise_frame_height, noise_frame_width, _ = whiteNoiseFrame.shape
        tfNet.setInput(cv2.dnn.blobFromImage(whiteNoiseFrame,size=(300,300)))
        outputNetNoise = tfNet.forward()

        noiseConfidence = outputNetNoise[0,0,0,2]
        #print(noiseConfidence, 'noise')

        for detection in outputNetNoise[0,0,:,:]:
            if detection[2]>0.5: #Confidence
                rw2.xmin=detection[3]*noise_frame_width
                rw2.ymin= detection[4]*noise_frame_height
                rw2.xmax= detection[5]*noise_frame_width
                rw2.ymax = detection[6]*noise_frame_height
                rw2.roi()
        if rw2.crop_img is not None:
            rw2.canny(50,200)
            #rw.HougLineP(20)
            rw2.HougLineStandart(50)
            cv2.imshow('Canny',rw2.cannyArray)
            cv2.rectangle(rw2.frame,(int(rw2.xmin),int(rw2.ymin)),(int(rw2.xmax),int(rw2.ymax)), (0, 0, 255),thickness=3)
        cv2.imshow('white noise', rw2.frame)

        whiteBlurFrame = addBlur(frame, kernelSizeH, kernelSizeW)
        rw3.frame = whiteBlurFrame
        blur_frame_height, blur_frame_width, _ = whiteBlurFrame.shape
        tfNet.setInput(cv2.dnn.blobFromImage(whiteBlurFrame,size=(300,300)))
        outputNetblur = tfNet.forward()

        blurConfidence = outputNetblur[0,0,0,2]
        #print(blurConfidence, 'blur')

        for detection in outputNetblur[0,0,:,:]:
            if detection[2]>0.5: #Confidence
                rw3.xmin=detection[3]*blur_frame_width
                rw3.ymin= detection[4]*blur_frame_height
                rw3.xmax= detection[5]*blur_frame_width
                rw3.ymax = detection[6]*blur_frame_height
                rw3.roi()
        if rw3.crop_img is not None:
            rw3.canny(50,200)
            #rw.HougLineP(20)
            rw3.HougLineStandart(50)
            cv2.imshow('Canny',rw3.cannyArray)
            cv2.rectangle(rw3.frame,(int(rw3.xmin),int(rw3.ymin)),(int(rw3.xmax),int(rw3.ymax)), (0, 0, 255),thickness=3)
        cv2.imshow('blur', rw3.frame)

        confidenceRowInstance = confidenceRowClass(frameNumber, originalConfidence, noiseConfidence, blurConfidence)
        confidencesList.append(confidenceRowInstance)

        #for obj in confidencesList:
         #   print( obj.frameKey , obj.originalConfidenceKey, obj.noiseConfidenceKey, obj.blurConfidenceKey)

        
    
        
        if cv2.waitKey(1)& 0xFF ==ord('q'):
            break
            cap.release()
            cv2.destroyAllWindows()
        
        



        

