import numpy as np 
import matplotlib.pyplot as plt

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

from PIL import Image 
import cv2

##lesson1##

img=Image.open('rock.jpg')
img.show()

img.rotate(-90)
img.show()

##check type of image
print(type(img))

img_array=np.asarray(img)
print(type(img_array))

print(img_array.shape)

plt.imshow(img_array)

##lesson2##

img_test=img_array.copy()
##only red channel
plt.imshow(img_test[:,:,0])

##scale red channel to gray
plt.imshow(img_test[:,:,0],cmap='gray')

##only green channel
plt.imshow(img_test[:,:,1])

##only blue channel
plt.imshow(img_test[:,:,2])

##remove red colour 
img_test[:,:,0]=0
plt.imshow(img_test)

##remove green colour 
img_test[:,:,1]=0
plt.imshow(img_test)

##remove blue colour 
img_test[:,:,2]=0
plt.imshow(img_test)


img=cv2.imread('rock.jpg')
print(type(img))
print(img.shape)
plt.imshow(img)

#opencv is reading the channels as BGR
#we will convert opencv to the channels  of the photo

img_fix=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img_fix)

#scale it to GRAY and check the SHAPE 
img_gray=cv2.imread('rock.jpg',cv2.IMREAD_GRAYSCALE)
print(img_gray.shape)
plt.imshow(img_gray,cmap='gray')
 #resize the image
img_new=cv2.resize(img_fix,(1000,400))
plt.imshow(img_new)

#hresize with ratio
width_ratio=0.5
height_ratio=0.5
img2=cv2.resize(img_fix, (0,0),img_fix,width_ratio,height_ratio)
plt.imshow(img2)

print(img2.shape)

##flip on horizontal Axis
img3=cv2.flip(img_fix,0)
plt.imshow(img3)

##flip on vertical Axis
img4=cv2.flip(img_fix,1)
plt.imshow(img4)


##flip on  horizontal and on vertical Axis
img5=cv2.flip(img_fix,-1)
plt.imshow(img5)

##change the size of our canva

last_img=plt.figure(figsize=(10,7))
ilp=last_img.add_subplot(111)
ilp.imshow(img_fix)

################# DRAW SHAPES ON IMAGES #######################

#create a black image to work on
black_img=np.zeros(shape=(512,512,3),dtype=np.int16)

#get the shape of the image
print(black_img.shape)
plt.imshow(black_img)

#draw a circle
cv2.circle(img=black_img,center=(400,100),radius=50,color=(255,0,0),thickness=8)
plt.imshow(black_img)

#filled circle
cv2.circle(img=black_img,center=(400,200),radius=50,color=(0,255,0),thickness=-1)
plt.imshow(black_img)

#draw a rectangle
cv2.rectangle(black_img,pt1=(200,200),pt2=(300,300),color=(0,255,0),thickness=5)
plt.imshow(black_img)

#draw triangle
vertices=np.array([[10,450],
                   [110,350],
                   [180,450]],
                  np.int32)
pts=vertices.reshape(-1,1,2)
cv2.polylines(black_img,[pts],isClosed=True,color=(0,0,255),thickness=3)
plt.imshow(black_img)

#Draw a line
cv2.line(black_img,pt1=(512,0),pt2=(0,512),color=(255,0,255),thickness=3)
plt.imshow(black_img)

#write text 
font=cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(black_img,text='Rhyme',org=(210,500),fontFace=font,fontScale=3,color=(255,255,0),thickness=3,lineType=cv2.LINE_AA)    
plt.imshow(black_img)

################# DRAW SHAPES WITH MOUSE & EVENT CHOICES FOR THE MOUSE #######################

#function
def draw_circle(event,x,y,flags,param):
    #left button down
    if event==cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img,(x,y),100,(35,69,78),-1)
        
    #right  button down
    elif event==cv2.EVENT_RBUTTONDOWN:
        cv2.circle(img,(x,y),50,(251,75,131),-1)
        
#connect the function with  the collback
cv2.namedWindow(winname="my_drawing")

#callback
cv2.setMouseCallback("my_drawing",draw_circle)

#using openCV to show the image 
img=np.zeros((512,512,3),np.int8)
while True:
    cv2.imshow("my_drawing",img)
    if (cv2.waitKey(5) & 0xFF==ord('q')):
        break
    
cv2.destroyAllWindows()

################# mouse functionality #######################


#variables
drawing=False
ex=-1
ey=-1

#function
def draw_rectangle(event,x,y,flags,params):
    
    global ex,ey,drawing
    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        ex,ey=x,y
     
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            cv2.rectangle(img,(ex,ey),(x,y),(255,0,255),-1)

        
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        cv2.rectangle(img,(ex,ey),(x,y),(255,0,255),-1)
        
#connect the function with  the collback
img=np.zeros((512,512,3),np.int8)
cv2.namedWindow(winname="my_drawing")

#callback
cv2.setMouseCallback("my_drawing",draw_rectangle)

#using openCV to show the image 

while True:
    cv2.imshow("my_drawing",img)
    if (cv2.waitKey(5) & 0xFF==ord('q')):
        break
    
cv2.destroyAllWindows()



























