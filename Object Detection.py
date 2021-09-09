import tensorflow as tf
from darkflow.net.build import TFNet
import cv2
import numpy as np
import time
from tkinter import *
from tkinter import filedialog
from os import path


root=Tk()
root.geometry('500x450')
root.resizable(0,0)

frame = Frame(root, relief=RIDGE, background='light blue', borderwidth=2)
frame.pack(fill=BOTH,expand=1)

icon = PhotoImage(file='icon.png')
root.iconphoto(False, icon)
root.title('OD')

label = Label(frame, text="OBJECT DETECTION",bg='light blue',font=('Times 35 bold'))
label.pack(side=TOP)

#filename = PhotoImage(file="demo.png")
#background_label = Label(frame,image=filename)
#background_label.pack(side=TOP)

"""-------------------------"""
options = {"model": 'D:/Roshan/Project/Darkflow/darkflow/cfg/yolo.cfg',
           "load": 'D:/Roshan/Project/Darkflow/darkflow/yolo.weights',
           "threshold": 0.6,
           #"gpu": 1.0
           }
tfnet = TFNet(options)


"""-----------------------------------------------------------------------------------------"""
def detect_camera():
    #mobile_cam = ("https://192.168.0.103:8080/video")
    video_cam = cv2.VideoCapture(0)
    while True:
        ret, frame = video_cam.read()
        cam_vid = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # new_vid = new_vid.astype('uint8')
        
        if ret:
            res = tfnet.return_predict(cam_vid)
            for i in res:
                top_x = i['topleft']['x']
                top_y = i['topleft']['y']
            
                btm_x = i['bottomright']['x']
                btm_y = i['bottomright']['y']
            
                confidence = i['confidence']
                label = i['label'] + " " + str(round(confidence*100, 3))
            
                if confidence > 0.5:
                    cam_vid = cv2.rectangle(cam_vid, (top_x, top_y), (btm_x, btm_y), (0,0,255), 4)
                    cam_vid = cv2.putText(cam_vid, label, (top_x, top_y-5), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                
                cv2.imshow('Captured video', cam_vid)
                cam_vid = cv2.resize(cam_vid, (350,350))
                #save.write(new_vid)
                if cv2.waitKey(1) & 0xff == ord('q'):
                    cv2.destroyAllWindows()
                    break
    
    cam_vid.release()
    video_cam.release()
    cv2.destroyAllWindows()

def detect_image(imagefile):
    res = tfnet.return_predict(imagefile)
    new_img = np.copy(imagefile)

    for i in res:
        top_x = i['topleft']['x']
        top_y = i['topleft']['y']
    
        btm_x = i['bottomright']['x']
        btm_y = i['bottomright']['y']
    
        confidence = i['confidence']
        label = i['label'] + " " + str(round(confidence*100, 3))
    
        if confidence > 0.5:
          new_img = cv2.rectangle(new_img, (top_x, top_y), (btm_x, btm_y), (0,0,255), 4)
          new_img = cv2.putText(new_img, label, (top_x, top_y-5), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
    
    new_img = cv2.resize(new_img, (0,0), fx=0.5, fy=0.5)
    save = "predicted_image_" + time.strftime("%Y_%m_%d_%H_%M") + ".png"
    print(save)
    cv2.imwrite(save, new_img)
    cv2.imshow('Predicted image',new_img)
    
    new_img.release()
    cv2.destroyAllWindows()
    
def detect_video(videofile):
    # w = videofile.get(cv2.CAP_PROP_FRAME_WIDTH)
    # h = videofile.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # fourcc = cv2.VideoWriter_fourcc('D','I','V','X') 
    # save = cv2.VideoWriter('predicted_video.mp4',fourcc,20.0,(int(w),int(h)))
    
    while True:
        ret, frame = videofile.read()
        new_vid = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # new_vid = new_vid.astype('uint8')
        
        if ret:
            res = tfnet.return_predict(new_vid)
            for i in res:
                top_x = i['topleft']['x']
                top_y = i['topleft']['y']
            
                btm_x = i['bottomright']['x']
                btm_y = i['bottomright']['y']
            
                confidence = i['confidence']
                label = i['label'] + " " + str(round(confidence*100, 3))
            
                if confidence > 0.5:
                    new_vid = cv2.rectangle(new_vid, (top_x, top_y), (btm_x, btm_y), (0,0,255), 4)
                    new_vid = cv2.putText(new_vid, label, (top_x, top_y-5), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                
                new_vid = cv2.resize(new_vid, (350,350))
                cv2.imshow('Detected video', new_vid)
                #save.write(new_vid)
                if cv2.waitKey(1) & 0xff == ord('q'):
                    cv2.destroyAllWindows()
                    break
    
    videofile.release()
    new_vid.release()
    #save.release()
    cv2.destroyAllWindows()

def BrowseFiles():
    img_file = [".jpg", ".png", ".jpeg"]
    filename = filedialog.askopenfilename(initialdir = "/",
                                          title = "Select a File",
                                          filetypes = [("image file", "*.jpg;*.png;*.jpeg"),
                                                       ("video file", "*.mp4;*.avi;*.mpeg")])
    
    name, extention = path.splitext(filename)
    
    if(extention in img_file):
        filename = filename.replace("/", "\\")
        filename = cv2.imread(filename)
        detect_image(filename)
    else:
        filename = filename.replace("/", "\\")
        filename = cv2.VideoCapture(filename)
        detect_video(filename)

cam=Button(frame,
           padx=5,
           pady=5,
           width=39,
           bg='white',
           fg='black',
           relief=GROOVE,
           command=detect_camera,
           text='Open Camera & Detect',
           font=('helvetica 15 bold'))
cam.place(x=5,y=100)

search=Button(frame,
              padx=5,
              pady=5,
              width=39,
              bg='white',
              fg='black',
              relief=GROOVE,
              command=BrowseFiles,
              text='Search File & Detect',
              font=('helvetica 15 bold'))
search.place(x=5,y=200)

exitt=Button(frame,padx=5,
             pady=5,
             width=5,
             bg='white',
             fg='black',
             relief=GROOVE,
             text='EXIT',
             command=root.destroy,
             font=('helvetica 15 bold'))
exitt.place(x=210,y=300)

root.mainloop()
