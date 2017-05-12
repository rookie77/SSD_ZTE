#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 21:19:09 2017

@author: ra
"""
# -*- coding: utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf8')
from PyQt5 import QtWidgets, QtGui,QtCore
import sys
from ZTE_UI import Ui_Form   # 导入生成first.py里生成的类
from PyQt5.QtWidgets import QFileDialog

#import cv2
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread
import tensorflow as tf
import os

from color_kmeans import coulor_Kmeans

from ssd import SSD300
from ssd_utils import BBoxUtility
from sklearn.cluster import KMeans
import utils
from colourTransmate import rgb2hsv,hsv2colour
from w2txt import write2txt


class mywindow(QtWidgets.QWidget,Ui_Form):
    signal_end=QtCore.pyqtSignal()
    file_signal_end=QtCore.pyqtSignal()

    def __init__(self):
        super(mywindow,self).__init__()
        self.setupUi(self)
        self.signal_end.connect(self.show_result)
        self.file_signal_end.connect(self.endOfFilePredict)

        #定义槽函数
    
        
    def openimage(self):
   # 打开文件路径
   #设置文件扩展名过滤,注意用双分号间隔
        global imgName
        self.file.clear()
        self.file.repaint()

#        self.sex.clear()
#        self.hat.clear()
#        self.mask.clear()
#        self.glass.clear()
        self.file_path.clear()
        imgName,imgType= QFileDialog.getOpenFileName(self,
                                    "打开图片",
                                    "",
                                    " *.jpg;;*.png;;*.jpeg;;*.bmp;;All Files (*)")

        print('imgName=',imgName)
        #利用qlabel显示图片
        png = QtGui.QPixmap(imgName).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(png)
    
        
    def img_predict(self):
        self.file.clear()
        self.file.repaint()
        self.sex.clear()
        self.hat.clear()
        self.mask.clear()
        self.glass.clear()
        self.file_path.clear()
        global imgName,colour_label
        colour_label={'sex':'male','glasses':'none','hat':'none','mask':'none'}
        inputs = []
        images = []
       
       
        img = image.load_img(imgName, target_size=(300, 300))
        img = image.img_to_array(img)
        images.append(imread(imgName))
        inputs.append(img.copy())
        
        inputs = preprocess_input(np.array(inputs))
        preds = model.predict(inputs, batch_size=1, verbose=1)
        results = bbox_util.detection_out(preds)
        for i, img in enumerate(images):
    # Parse the outputs.
            
            det_label = results[i][:, 0]
            det_conf = results[i][:, 1]
            det_xmin = results[i][:, 2]
            det_ymin = results[i][:, 3]
            det_xmax = results[i][:, 4]
            det_ymax = results[i][:, 5]

    # Get detections with confidence higher than 0.6.
            top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]

            top_conf = det_conf[top_indices]
            top_label_indices = det_label[top_indices].tolist()
            top_xmin = det_xmin[top_indices]
            top_ymin = det_ymin[top_indices]
            top_xmax = det_xmax[top_indices]
            top_ymax = det_ymax[top_indices]

            colors = plt.cm.hsv(np.linspace(0, 1, 7)).tolist()
            if (len(img.shape)==2):
                plt.imshow(img,plt.cm.gray)
            else:
                plt.imshow(img / 255.)
            currentAxis = plt.gca()
            
            
    
        for i in range(top_conf.shape[0]):
    
            xmin = int(round(top_xmin[i] * img.shape[1]))
            ymin = int(round(top_ymin[i] * img.shape[0]))
            xmax = int(round(top_xmax[i] * img.shape[1]))
            ymax = int(round(top_ymax[i] * img.shape[0]))


            score = top_conf[i]
            label = int(top_label_indices[i])
            label_name = zhongxing_class[label - 1]
#            print (label_name,top_xmax[i],xmax)
#            print( img.shape[1], img.shape[0])
        ###decide colour 
            if label_name=='male' or label_name=='female':
                colour_label['sex']=label_name
            if label_name!='male' and label_name!='female':
                
#                if label_name=='glasses':
#                    xmax=xmax-(xmax-xmin)/2
                w=xmax-xmin
                h=ymax-ymin
                cut_rate=0.2
                if (len(img.shape)==3):
                    image_new=img[int(ymin+h*cut_rate):int(ymax-h*cut_rate),int(xmin+cut_rate*w):int(xmax-w*cut_rate),:]
                    image_new = image_new.reshape((image_new.shape[0] * image_new.shape[1], 3))

                else:
                    image_new=img[int(ymin+h*cut_rate):int(ymax-h*cut_rate),int(xmin+cut_rate*w):int(xmax-w*cut_rate)]

# cluster the pixel intensities
                clt = KMeans(n_clusters = 5)
                clt.fit(image_new)
                hist = utils.centroid_histogram(clt)
                label_RGB=clt.cluster_centers_[hist.argmax()]
                
  ####distinguish colour
              
                label_HSV=rgb2hsv(label_RGB[0],label_RGB[1],label_RGB[2])                
                colour_label[label_name]=hsv2colour(label_HSV[0],label_HSV[1],label_HSV[2])
#                if label_name=='glasses':
#                    xmax=xmax+w
               
        
            
            display_txt = '{:0.2f}, {}'.format(score, label_name)
            coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
            color = colors[label]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})
            currentAxis.set_xticks([])
            currentAxis.set_yticks([])

            
   
        
        plt.savefig("result.jpg")
        plt.close()
       
#        if (colour_label['glasses']!='none'):
#            print (colour_label['glasses'])
        if(colour_label['glasses']!='none' and colour_label['glasses']!='黑色'):
                colour_label['glasses']='透明'
        print("Done!")
        self.signal_end.emit()
    def show_result(self):
        global colour_label
        
        png_result= QtGui.QPixmap("./result.jpg").scaled(self.fig_detected.width(), self.fig_detected.height())
        
        self.fig_detected.setPixmap(png_result)
#        self.file.clear()

        self.file.append('Done!')
#        self.sex.clear()
        self.sex.append(colour_label['sex'])
#        self.glass.clear()
        self.glass.append(colour_label['glasses'])
        
#        self.hat.clear()
        self.hat.append(colour_label['hat'])
        
#        self.mask.clear()
        self.mask.append(colour_label['mask'])
        
        
    def file_predict(self):
        result_file=open('./result.txt','a')
        result_file.write('\n')
        result_file.close()
        
        global imgName
        self.file.clear()
        self.file.repaint()
        
        self.file_path.clear()
        self.file_path.repaint()

        fileName=imgName.rstrip(imgName.split('/')[-1])
        
               
        img_paths = fileName
        imgs = os.listdir(img_paths)
        for index,item in enumerate(imgs):
            imgs[index] = img_paths + imgs[index]
       
        
        for img_path in imgs:
            print(img_path)
            result={}
            inputs = []
            images = []
            colour_label={'sex':'male','glasses':'none','hat':'none','mask':'none'}
            img = image.load_img(img_path, target_size=(300, 300))
            img = image.img_to_array(img)
            images.append(imread(img_path))
            inputs.append(img.copy())
        
            inputs = preprocess_input(np.array(inputs))
            preds = model.predict(inputs, batch_size=1, verbose=1)
            results = bbox_util.detection_out(preds)
            for i, img in enumerate(images):
    # Parse the outputs.
                det_label = results[i][:, 0]
                det_conf = results[i][:, 1]
                det_xmin = results[i][:, 2]
                det_ymin = results[i][:, 3]
                det_xmax = results[i][:, 4]
                det_ymax = results[i][:, 5]

    # Get detections with confidence higher than 0.6.
                top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]

                top_conf = det_conf[top_indices]
                top_label_indices = det_label[top_indices].tolist()
                top_xmin = det_xmin[top_indices]
                top_ymin = det_ymin[top_indices]
                top_xmax = det_xmax[top_indices]
                top_ymax = det_ymax[top_indices]
                for i in range(top_conf.shape[0]):
                    xmin = int(round(top_xmin[i] * img.shape[1]))
                    ymin = int(round(top_ymin[i] * img.shape[0]))
                    xmax = int(round(top_xmax[i] * img.shape[1]))
                    ymax = int(round(top_ymax[i] * img.shape[0]))
                    #score = top_conf[i]
                    label = int(top_label_indices[i])
                    label_name = zhongxing_class[label - 1]
         ###decide colour 
                    if label_name=='male' or label_name=='female':
                        colour_label['sex']=label_name
                    if label_name!='male' and label_name!='female':
                
#                        if label_name=='glasses':
#                            xmax=xmax-(xmax-xmin)/2
                        w=xmax-xmin
                        h=ymax-ymin
                        cut_rate=0.2
                        if (len(img.shape)==3):
                            image_new=img[int(ymin+h*cut_rate):int(ymax-h*cut_rate),int(xmin+cut_rate*w):int(xmax-w*cut_rate),:]
                            image_new = image_new.reshape((image_new.shape[0] * image_new.shape[1], 3))

                        else:
                            image_new=img[int(ymin+h*cut_rate):int(ymax-h*cut_rate),int(xmin+cut_rate*w):int(xmax-w*cut_rate)]

# cluster the pixel intensities
                        clt = KMeans(n_clusters = 5)
                        clt.fit(image_new)
                        hist = utils.centroid_histogram(clt)
                        label_RGB=clt.cluster_centers_[hist.argmax()]
                
  ####distinguish colour
              
                        label_HSV=rgb2hsv(label_RGB[0],label_RGB[1],label_RGB[2])                
                        colour_label[label_name]=hsv2colour(label_HSV[0],label_HSV[1],label_HSV[2])
    
#                    if label_name=='glasses':
#                         xmax=xmax+w
               
        
#        display_txt = '{:0.2f}, {}'.format(score, label_name)
#        coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1 
#        color = colors[label]
#        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
#        currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})
#    
#    plt.show()
                    if(colour_label['glasses']!='none' and colour_label['glasses']!='黑色'):
                        colour_label['glasses']='透明'
                    #print(imgs[count][-10:],colour_label)
                    result[(img_path.split('/')[-1])]=colour_label
                
                write2txt(result)
        print('Done!')
        self.file_signal_end.emit()


    def endOfFilePredict(self):
#        self.file.clear()

        self.file.append('Done!')
#        self.file_path.clear()
        self.file_path.append (os.path.abspath('.')+'\\'+'result.txt')
    
       
        
        
if __name__=="__main__":
    global imgName,colour_label
    colour_label={'sex':'male','glasses':'none','hat':'none','mask':'none'}
    result_file=open('./result.txt','w')
    result_file.write(u'图片名称         性别       眼镜       镜片颜色   口罩      口罩颜色   帽子    帽子颜色')
    result_file.write('\n')
    result_file.close()



    np.set_printoptions(suppress=True)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.45
    set_session(tf.Session(config=config))


    zhongxing_class = ['male', 'female', 'glasses', 'hat', 'mask']
    NUM_CLASSES = len(zhongxing_class) + 1
    input_shape=(300, 300, 3)
    model = SSD300(input_shape, num_classes=NUM_CLASSES)
    
    if getattr(sys,'frozen',None):
        tmp=sys._MEIPASS
    else:
        tmp=os.path.dirname(__file__)
    filename=os.path.join(tmp,'checkpoints/weights.17-1.10.hdf5')
    model.load_weights(filename, by_name=True)
    bbox_util = BBoxUtility(NUM_CLASSES)
    
    app=0
    app = QtWidgets.QApplication(sys.argv)
    #app.aboutToQuit.connect(quit)
    window = mywindow()
    
    window.show()
    sys.exit(app.exec_())
   
 

