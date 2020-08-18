import sys
import cv2
from cv2 import *
import numpy as np
import sys

##### glabal 調校參數 #####

kernel_d = np.ones((3,3), np.uint8)	#做 dilate() 用的 kernel
kernel_e = np.ones((3,3), np.uint8)	#做 erode()  用的 kernel
kernel_gauss = (3,3)	#做 GaussianBlur()  用的 kernel

dilate_times = 13
erode_times = 5

is_blur = True
is_close = True
is_draw_ct = False

fac = 2

##### -------------- #####

def drawRectangle(frame, minus_frame):
	if(is_blur):
		minus_frame = GaussianBlur(minus_frame, kernel_gauss, 0)	#再來點高斯模糊

	minus_Matrix = np.float32(minus_frame)	#把 minus_frame 抓成 numpy-array: minus_Matrix
	
	if(is_close):
		for i in range(dilate_times):
			minus_Matrix = dilate(minus_Matrix, kernel_d)	#做 13次 dilate

		imshow('dilate', minus_Matrix)	#輸出在螢幕上 顯示做完 dilate的影像

		for i in range(erode_times):
			minus_Matrix = erode(minus_Matrix, kernel_e)	#做 3次 erode

		imshow('erode', minus_Matrix)	#輸出在螢幕上 顯示做完 erode的影像

	#為了能做 findContours() 做點格式修正
	minus_Matrix = np.clip(minus_Matrix, 0, 255)
	minus_Matrix = np.array(minus_Matrix, np.uint8)

	#用 findContours 找輪廓
	contours, hierarchy = findContours(minus_Matrix.copy(), RETR_TREE, CHAIN_APPROX_SIMPLE)

	for c in contours:	#針對 每一塊 找到的輪廓 畫框框

		(x, y, w, h) = boundingRect(c)		#計算外框範圍
		rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)	#畫出外框 貼在原圖上
		
		#可註解掉: 畫出原邊界（除錯用)
		if( is_draw_ct ):
			drawContours(frame, contours, -1, (0, 255, 255), 2)

	# 顯示偵測結果影像
	imshow('result', frame)

def objDetect(vdo):

	capture = VideoCapture(vdo);	#把影片用 opencv 抓成影片物件
	width = (int)( capture.get( CAP_PROP_FRAME_WIDTH )/fac )
	length = (int)( capture.get( CAP_PROP_FRAME_HEIGHT )/fac )
	try:
		if capture.isOpened():	#如果有成功開啟影片檔再來做事

			(ret_old, old_frame) = capture.read()	#把影片物件讀成 單一張影像: old_frame
			old_frame = resize( old_frame, ( width,length ),interpolation = INTER_CUBIC )
		
			gray_oldframe = cvtColor(old_frame, COLOR_BGR2GRAY)		#原本是 單張彩色影像，轉成 單張灰階影像 gray_oldframe
			if(is_blur):
				gray_oldframe = GaussianBlur(gray_oldframe, kernel_gauss, 0)	#再做點高斯模糊

			oldBlurMatrix = np.float32(gray_oldframe)	#把 gray_oldframe 抓成 numpy array
		
			#The function calculates the weighted sum of the input image gray_oldframe and the accumulator BlurMatrix 
			#so that oldBlurMatrix becomes a running average of a frame(0.003) sequence
			#數學表達: oldBlurMatrix(x,y) = (1-0.003)*oldBlurMatrix(x,y) + (0.003)*gray_oldframe(x,y)
			accumulateWeighted(gray_oldframe, oldBlurMatrix, 0.003)

			while(True):	#無窮迴圈，直到影片的影格跑完

				ret, frame = capture.read()	#把影片物件目前這張影格 讀成單一張影像: frame
				frame = resize( frame, ( width,length ),interpolation = INTER_CUBIC )
			
				gray_frame = cvtColor(frame, COLOR_BGR2GRAY)		#原本是 單張彩色影像，轉成 單張灰階影像 gray_oldframe

				if(is_blur):
					newBlur_frame = GaussianBlur(gray_frame, kernel_gauss, 0)	#再做點高斯模糊
				else:
					newBlur_frame = gray_frame
			
				newBlurMatrix = np.float32(newBlur_frame)			#把 newBlur_frame 抓成 numpy array: newBlurMatrix

				minusMatrix = absdiff(newBlurMatrix, oldBlurMatrix)	#相減陣列

				ret, minus_frame = threshold(minusMatrix, 60, 255.0, THRESH_BINARY)	#抓 (60, 255.0)

				#The function calculates the weighted sum of the input image gray_oldframe and the accumulator BlurMatrix 
				#so that oldBlurMatrix becomes a running average of a frame(0.02) sequence
				#數學表達: oldBlurMatrix(x,y) = (1-0.003)*newBlurMatrix(x,y) + (0.003)*oldBlurMatrix(x,y)
				#靠這個function 留住"上一個瞬間" 存在 oldBlurMatrix
				accumulateWeighted(newBlurMatrix,oldBlurMatrix,0.02)

				imshow("difference", minus_frame)	#輸出在螢幕上 變化差異處

				drawRectangle(frame, minus_frame)	#call抓出移動物件的function

				#影片的影格跑完，或是按下按鍵'q'結束程式
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
		
			capture.release(capture);

		else:	#影片檔開檔失敗，啥都不做
			pass

	except Exception:
		print("End of program.")


if __name__ == "__main__":
	videoPath = ""
	for arg in sys.argv[1:]:
		videoPath = arg	#第一個參數是 影片檔的 path, 同一個資料夾的話就打檔名就好
	
	objDetect(videoPath)
