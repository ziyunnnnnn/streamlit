

import streamlit as st
import cv2
from PIL import Image,ImageEnhance
import numpy as np
import pandas as pd
import glob
import random
import os
import streamlit as st
import pandas as pd
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os
#import tensorflow as tf
#import tensorflow_hub as hub
import time ,sys
#from streamlit_embedcode import github_gist
import urllib.request
import urllib
#import moviepy.editor as moviepy
import cv2
import numpy as np
import time
import sys
import sqlite3 
import hashlib

# @st.cache(persist=True)
# def load_image(img):
#     im = Image.open(img)
#     return im


# @st.cache(persist=True)
# def yolo_objectdetection():

#     # Load Yolo
#     net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_testing.cfg")

#     # Name custom object
#     classes = ["Land"]

#     # Images path
#     global images_path
#     images_path = glob.glob(r"...\uploads")


#     layer_names = net.getLayerNames()
#     output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
#     colors = np.random.uniform(0, 255, size=(len(classes), 3))

#     # Insert here the path of your images
#     random.shuffle(images_path)
#     # loop through all the images
#     for img_path in images_path:
#         # Loading image
#         img = cv2.imread(img_path)
#         img = cv2.resize(img, None, fx=3.5, fy=3.5)
#         height, width, channels = img.shape

#         # Detecting objects
#         blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

#         net.setInput(blob)
#         outs = net.forward(output_layers)

#         # Showing informations on the screen
#         class_ids = []
#         confidences = []
#         boxes = []
#         for out in outs:
#             for detection in out:
#                 scores = detection[5:]
#                 class_id = np.argmax(scores)
#                 confidence = scores[class_id]
#                 if confidence > 0.3:
#                     # Object detected
#                     print(class_id)
#                     center_x = int(detection[0] * width)
#                     center_y = int(detection[1] * height)
#                     w = int(detection[2] * width)
#                     h = int(detection[3] * height)

#                     # Rectangle coordinates
#                     x = int(center_x - w / 2)
#                     y = int(center_y - h / 2)

#                     boxes.append([x, y, w, h])
#                     confidences.append(float(confidence))
#                     class_ids.append(class_id)

#         indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
#         print(indexes)
#         font = cv2.FONT_HERSHEY_PLAIN
#         for i in range(len(boxes)):
#             if i in indexes:
#                 x, y, w, h = boxes[i]
#                 label = str(classes[class_ids[i]])
#                 color = colors[class_ids[i]]
#                 cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
#                 cv2.putText(img, label, (x, y + 30), font, 3, color, 2)

#         cv2.imshow('Image',img)
#         key = cv2.waitKey(0)

#     cv2.destroyAllWindows()


# def main():
#     """
#     Land Detection App
#     """
#     st.title('이륜자동차 위험물 탐지')
#     #st.text('Build with Streamlit,Yolov3 and OpenCV')

#     menu = ['이미지','동영상', '실시간 영상']
#     choice = st.sidebar.selectbox('Menu',menu)

#     if choice == '이미지':
#         st.subheader(' 위험물 탐지')
#         image_file = st.file_uploader("Upload Image",type=['jpg', 'png', 'jpeg'])

#         if image_file is not None:
#             our_image = Image.open(image_file)
#             st.text('Original Image')
#             # st.write(type(our_image))
#             st.image(our_image)

#         enhance_type = st.sidebar.radio('Enhance Type', ['Original', 'Gray-Scale', 'Contrast', 'Brightness', 'Blurring'])

#         if enhance_type == 'Gray-Scale':
#             new_img = np.array(our_image.convert('RGB'))
#             img = cv2.cvtColor(new_img, 1)
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             # st.write(new_img)
#             st.image(gray)

#         if enhance_type == 'Contrast':
#             c_rate = st.sidebar.slider('Contrast', 0.5, 3.5)
#             enhancer = ImageEnhance.Contrast(our_image)
#             img_output = enhancer.enhance(c_rate)
#             st.image(img_output)

#         if enhance_type == 'Brightness':
#             c_rate = st.sidebar.slider('Brightness', 0.5, 3.5)
#             enhancer = ImageEnhance.Brightness(our_image)
#             img_output = enhancer.enhance(c_rate)
#             st.image(img_output)

#         if enhance_type == 'Blurring':
#             new_img = np.array(our_image.convert('RGB'))
#             blur_rate = st.sidebar.slider('Blurring', 0.5, 3.5)
#             img = cv2.cvtColor(new_img, 1)
#             blur_img = cv2.GaussianBlur(img, (11, 11), blur_rate)
#             st.image(blur_img)
#         else:
#             pass

#         # Land Detection
#         task = ['Land']
#         feature_choice = st.sidebar.selectbox("Find Features", task)
#         if st.button("Process"):
#             if feature_choice == 'Land':
#                 result_img = yolo_objectdetection()
#                 result_img = []
#                 st.image(result_img)

#                 st.success("Found {} Land".format(len(result_img)))

#             else:
#                 st.markdown('## Land not Detected')
#                 st.markdown('## Kindly upload correct image ')



#     elif choice == '동영상':
#         st.subheader(' 위험물 탐지')
#         video_file = st.file_uploader('video', type = ['mp4'])
#         cap = cv2.VideoCapture(video_file)


#     elif choice == '실시간 영상':
#         st.subheader(' 위험물 탐지')
#         video_file = st.file_uploader('video', type = ['mp4'])
#         cap = cv2.VideoCapture(video_file)



# if __name__ == '__main__':
#     main()
# -------------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os
import tensorflow as tf
import tensorflow_hub as hub
import time ,sys
from streamlit_embedcode import github_gist
import urllib.request
import urllib
import moviepy.editor as moviepy
import cv2
import numpy as np
import time
import sys

# def main():
#     st.sidebar.title("Select Activity")
#     choice  = st.sidebar.selectbox("MODE",("About","Object Detection(Image)","Object Detection(Video)"))
    
#     if choice == "Object Detection(Image)":
#         st.subheader(' 위험물 탐지')
#         file = st.file_uploader("Upload Image",type=['jpg', 'png', 'jpeg'])

#     if file!= None:
#         img1 = Image.open(file)
#         img2 = np.array(img1)

#         st.image(img1, caption = "Uploaded Image")

#         # 모델 관련
#         config_path = r'config_n_weights\yolov3.cfg'
#         weights_path = r'config_n_weights\yolov3.weights'
#         # net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

#         # img_detected = model.detect(img1)
#         st.image(img1, caption = "detected Image")   # 처리된 이미지
#         # my_bar = st.progress(0)
#         # confThreshold =st.slider('Confidence', 0, 100, 50)
#         # nmsThreshold= st.slider('Threshold', 0, 100, 20)
#         #classNames = []
#         # whT = 320
#         # url = "https://raw.githubusercontent.com/zhoroh/ObjectDetection/master/labels/coconames.txt"
#         # f = urllib.request.urlopen(url)
#         # classNames = [line.decode('utf-8').strip() for  line in f]
#         #f = open(r'C:\Users\Olazaah\Downloads\stream\labels\coconames.txt','r')
#         #lines = f.readlines()
#         #classNames = [line.strip() for line in lines]




#     elif choice == "Object Detection(Video)":
#         st.subheader(' 위험물 탐지')
#         video_file = st.file_uploader('video', type = ['mp4'])
#         cap = cv2.VideoCapture(video_file)
#         try:

#             clip = moviepy.VideoFileClip('detected_video.mp4')
#             clip.write_videofile("myvideo.mp4")
#             st_video = open('myvideo.mp4','rb')
#             video_bytes = st_video.read()
#             st.video(video_bytes)
#             st.write("Detected Video") 
#         except OSError:
#             ''

#     elif choice == "About":
#         print()
        

# if __name__ == '__main__':
# 		main()	

# 로그인 화면 
conn = sqlite3.connect('database.db')
c = conn.cursor()

import hashlib
def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False

def create_user():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')

def add_user(username,password):
	c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
	conn.commit()

def login_user(username,password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data

def main():

	#st.title("로그인 기능 테스트")

	menu = ["Login","signUp","Home"]
	choice = st.sidebar.selectbox("Login",menu)

    
	if choice == "Login":
		st.subheader("로그인 해주세요")

		username = st.sidebar.text_input("유저명을 입력해주세요")
		password = st.sidebar.text_input("비밀번호를 입력해주세요",type='password')
		if st.sidebar.checkbox("Login"):
			create_user()
			hashed_pswd = make_hashes(password)

			result = login_user(username,check_hashes(password,hashed_pswd))
			if result:

				st.success("{}님으로 로그인했습니다.".format(username))

			else:
				st.warning("사용자 이름이나 비밀번호가 잘못되었습니다.")

	elif choice == "signUp":
		st.subheader("새 계정을 만듭니다.")
		new_user = st.text_input("유저명을 입력해주세요")
		new_password = st.text_input("비밀번호를 입력해주세요",type='password')

		if st.button("signUp"):
			create_user()
			add_user(new_user,make_hashes(new_password))
			st.success("계정 생성에 성공했습니다.")
			st.info("로그인 화면에서 로그인 해주세요.")
    
	elif choice == "Home":
		st.subheader("새 계정을 만듭니다.")		
		selected_item = st.radio("Radio Part", ("A", "B", "C"))
		if selected_item == "A":
			st.write("A!!")
		elif selected_item == "B":
			st.write("B!")

if __name__ == '__main__':
	main()

