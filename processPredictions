import os
import pandas as pd
import glob
from keras.preprocessing import image
from imageio import imread
import numpy as np
from pascal_voc_writer import Writer
####################################################################
# This script parses the .csv of HIT results, visualizes the results
# provides a GUI to accept or reject the results and creates a result
# .csv to upload on Mech Turk
####################################################################

import initialize_model
import cv2
import pandas as pd
import sys
from PIL import Image, ImageTk, ImageDraw, ImageFont
import tkinter
import json
import shutil
from tkinter import Tk,Button,HORIZONTAL
from tkinter.ttk import Progressbar

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)

# For ssd model
img_height = 300
img_width = 300

# original image
original_image_height = 0
original_image_width = 0
####################################################################
# This script parses the .csv of HIT results, visualizes the results
# provides a GUI to accept or reject the results and creates a result
# .csv to upload on Mech Turk
####################################################################


# Variables#
# path of the HIT results

#hit_path = "../resources/Batch_3408616_batch_results.csv"
data_path = "/media/auv/DATA/Data/Fish_Training_Data/Test/PMFS/images"
annotation_path = "/media/auv/Untitled/200510_PMFS/Dive05_DaisyBank/annotations"
# edited - Sahil
#accepted = "../resources/acceptedAnnotations/"
#rejected = "../resources/rejectedImages/"
prediction_thresh = []
dummy_path = "/home/auv/Pictures/dummy_pmfs/"
fields = [""]
counter = 0

# get image list
def get_images():
    image_list = []
    for im_path in os.listdir(data_path):
        img = os.path.join(data_path, im_path)
        image_list.append(img)
    return image_list

def createImageArray(path):
    files = glob.glob(path+'/*.jpg')
    return files

###
# Method which takes Annotation dataframe as input and gets the paths of the corresponding images in datafolder into a list
###
def get_images_annotations(df, path_lb):
    hit_df_filtered = df[["HITId", "Input.image_url", "Answer.annotation_data"]]
    image_annotation_list = []
    for index, row in hit_df_filtered.iterrows():
        # print(row["Input.image_url"], row["Answer.annotation_data"])
        worker_answer = json.loads(row["Answer.annotation_data"])
        path_lb.insert(tkinter.END, row["Input.image_url"])
        img = os.path.join(data_path, row["Input.image_url"])
        image_annotation_list.append((index, img, worker_answer))
        #to process image in order
    return image_annotation_list[::-1]

def prediction(path):
	orig_image = [] # Store the images here
	input_images = [] # Store resized versions of the images here.
	orig_image.append(imread(path))
	inference_img = image.load_img(path, target_size=(img_height, img_width))
	inference_img = image.img_to_array(inference_img)
	input_images.append(inference_img)
	input_images = np.array(input_images)
	model = initialize_model.get_model()
	y_pred = model.predict(input_images)
	confidence_threshold = 0.5
	y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

	return y_pred_thresh, orig_image

def tk_image(path, w, h):
    img = Image.open(path)
    draw = ImageDraw.Draw(img)
    # get a font
    fnt = ImageFont.truetype('../resources/arial.ttf', 27)

    global prediction_thresh
    prediction_thresh, orig_image = prediction(path)

    np.set_printoptions(precision=2, suppress=True, linewidth=90)
    print("Predicted boxes:\n")
    print('   class   conf xmin   ymin   xmax   ymax')
    print(prediction_thresh)
    # draw.rectangle([left, top, right, bottom])
    classes = ['background','fish', 'starfish','coral', 'mud']
    
    if(len(prediction_thresh[0] > 0)):
	    for box in prediction_thresh[0]:
	        xmin = float(box[2] * orig_image[0].shape[1] / img_width)
	        print("top", xmin)
	        ymin = float(box[3] * orig_image[0].shape[0] / img_height)
	        print("left", ymin )
	        xmax = float(box[4] * orig_image[0].shape[1] / img_width)
	        print("bottom", xmax )
	        ymax = float(box[5] * orig_image[0].shape[0] / img_height)
	        print("right", ymax )
	        label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
	        draw.line([(xmin, ymin), (xmax, ymin), (xmax, ymax), 
	               (xmin, ymax), (xmin,ymin)], width=3)
	        draw.text((xmin, ymin-10), label , font=fnt, fill=(255,255,255,128))
    
    img = img.resize((w, h))
    global counter
    counter += 1
    filename = dummy_path + str(counter) + ".jpg"
    print(filename)
    img.save(filename)
    #cv2.imwrite(dummy_path + filename, img)
    #print("width", w )
    storeobj = ImageTk.PhotoImage(img)
    #print(type(storeobj))
    return storeobj


# Creating Canvas Widget
class PictureWindow():
    def __init__(self, master, **kwargs):

        self.imagelist_p = []
        self.current_image = ''
        self.result_Dictionary = {}
        self.master = master
        self.counter = 0

        for key, val in kwargs.items():
            if key == "width":
                self.w = int(val)
            elif key == "height":
                self.h = int(val)
        self.result = tkinter.IntVar()
        self.setup_gui()
        # Dataframe to hold the annotation data
        #self.annotation_df = pd.read_csv(hit_path)
        self.current_hit = ""
        # get the list of images from the annotation frame
        self.imagelist= createImageArray(data_path)
        self.original_imglist = self.imagelist.copy()
        # 
        self.counter_frame = tkinter.Frame(self.master, height=100, bg="Black")
        #tkinter.Label(self.counter_frame, text="Total Images: "+ str(self.annotation_df.count(axis='rows')[0])).pack(side=tkinter.LEFT)
        self.imagesDone = tkinter.Label(self.counter_frame, fg="red")
        self.imagesDone.pack(side=tkinter.LEFT)
        self.counter_frame.pack(side=tkinter.TOP)
        self.done = 0
        # debug statement
        #print(self.annotation_df.count(axis='rows')[0])


    def on_close(self):
        print("in closing")
        #self.annotation_df.to_csv("modified.csv")
        return

    def reverse(tuple):
        new_tuple = ()
        for x in reversed(tuple):
            new_tuple = new_tuple + (k,)
        return new_tuple

    def show_image(self, pop_tuple):
        print("showing image")
        path = pop_tuple
        print(path)
        #bbox = pop_tuple[2]
        self.imagesDone.config(text="Images Done: " + str(len(self.result_Dictionary)))
        print(self.w)
        img = tk_image(path, self.w, self.h)
        self.img_canvas.delete(self.img_canvas.find_withtag("bacl"))
        self.allready = self.img_canvas.create_image(self.w / 2, self.h / 2, image=img, anchor='center', tag="bacl")

        self.image = img
        
        self.current_hit = pop_tuple
        #print(self.img_canvas.find_withtag("bacl"))
        #print(self.current_hit)
        self.master.title("Image Viewer ({})".format(path))

        # edited - Sahil
        '''
        if self.result_Dictionary.get(self.current_hit) is not None:
            self.result.set(self.result_Dictionary.get(self.current_hit))
        else:
            self.result.set(0)
        '''

        return

    def previous_image(self):
        try:
            pop = self.imagelist_p.pop()
            #previous_result = self.result.pop()
            self.show_image(pop)
            self.imagelist.append(pop)
        except:
            pass
        return

    def next_image(self):
        try:
            pop = self.imagelist.pop()
            print(pop)
            self.current_image = pop
            self.show_image(pop)
            self.imagelist_p.append(pop)
        except EOFError as e:
            pass
        return

    def onRightkey(self, event):
        print (" Next pressed")
        self.next_image()
        return

    def onLeftkey(self, event):
        print ("Prev pressed")
        self.previous_image()
        return

    def onQkey(self, event):
        self.result.set(1)
        self.update_approval()
        return

    def onEkey(self, event):
        self.result.set(2)
        self.update_approval()
        return

    def select_image(self, event):
        print("image selection")
        items = self.path_listbox.curselection()
        if len(items) == 1:
            ind = int(items[0])
            pop_ind = self.current_hit
            print("current hit ind :"+str(self.current_hit) +"  required ind :"+ str(ind))

            if(self.current_hit < ind):
                #next image until finding the ind
                while( pop_ind < ind):
                    pop = self.imagelist.pop()
                    self.imagelist_p.append(pop)
                    pop_ind = pop[0]
                    print("popped ind :"+ str(pop_ind))
                    if (pop_ind == ind):
                        self.show_image(pop)
                        break;

            elif (self.current_hit > ind):
                # previosdu image until finidng the ind
                # next image until finding the ind
                while (pop_ind > ind):
                    pop = self.imagelist_p.pop()
                    self.imagelist.append(pop)
                    pop_ind = pop[0]
                    print("popped ind :" + str(pop_ind))
                    if (pop_ind == ind):
                        self.show_image(pop)
            else:
                #nothing
                return
            if (pop_ind != ind):
                print("ERROR: didnt find the image")
        return

    def update_approval(self):
    	# Writer(path,width,height)
    	global prediction_thresh
    	writer = Writer(self.current_image,1280,1024)
    	classes = ['background','fish', 'starfish','coral', 'mud']
    	print(prediction_thresh)
    	if(len(prediction_thresh[0] > 0)):
    		for box in prediction_thresh[0]:
    			xmin = float(box[2] * 1280 / img_width)
    			ymin = float(box[3] * 1024 / img_height)
    			xmax = float(box[4] * 1280 / img_width)
    			ymax = float(box[5] * 1024 / img_height)
    			label = '{}'.format(classes[int(box[0])])
    			writer.addObject(label,xmin,ymin,xmax,ymax)
    		current_annotation_path = os.path.splitext(self.current_image)[0]
    		writer.save(current_annotation_path + '.xml')
    	return

    def setup_gui(self):
        #this is the canvas to sho image
        self.img_canvas = tkinter.Canvas(self.master, width=self.w, height=self.h)
        #bind root widget to keys to go to next image and previous image
        self.master.bind("<Right>", self.onRightkey)
        self.master.bind("<Left>", self.onLeftkey)
        self.master.bind("<q>", self.onQkey)
        self.master.bind("<e>", self.onEkey)

        #create buttons to go next and previous
        self.create_buttons(self.img_canvas)

        self.img_canvas.pack(side=tkinter.LEFT)

        self.control_frame = tkinter.Frame(self.master)

        result_frame = tkinter.Frame(self.control_frame)
        #tkinter.Radiobutton(result_frame, text="Save", \
        #                    indicatoron=0,width=20,height = 10,\
        #                    variable=self.result, value= 3, \
        #                    command=self.on_close).pack(side=tkinter.LEFT)
        tkinter.Radiobutton(result_frame, text="Accept",\
                            indicatoron = 0,width = 20,height = 10,  \
                            variable=self.result, value= 1, \
                            command=self.update_approval).pack(side=tkinter.LEFT)
        tkinter.Radiobutton(result_frame, text="Reject", \
                            indicatoron=0,width=20,height = 10,\
                            variable=self.result, value= 2, \
                            command=self.update_approval).pack(side=tkinter.RIGHT)
        

        list_frame = tkinter.Frame(self.control_frame)
        scrollbar = tkinter.Scrollbar(list_frame, orient=tkinter.VERTICAL)

        self.path_listbox = tkinter.Listbox(list_frame, height=30, width=20,yscrollcommand=scrollbar.set)
        self.path_listbox.bind("<Double-Button-1>", self.select_image)

        scrollbar.pack(side=tkinter.RIGHT)
        self.path_listbox.pack(fill=tkinter.X)

        result_frame.pack(side=tkinter.TOP, fill=tkinter.X)
        list_frame.pack(side=tkinter.BOTTOM, fill=tkinter.X)

        self.control_frame.pack(side=tkinter.LEFT)

        # self.window_settings()


    def create_buttons(self, c):
        tkinter.Button(c, text=" > ",width = 10,height = 5, command=self.next_image).place(x=(self.w / 1.1), y=(self.h / 2))
        tkinter.Button(c, text=" < ", width = 10, height = 5, command=self.previous_image).place(x=20, y=(self.h / 2))
        c['bg'] = "white"
        return


# Main Function
def main():
    # Creating Window
    root = tkinter.Tk(className=" Image Viewer")
    # Creating the main Widget
    app = PictureWindow(root, width=1280, height=1024)
    # Not Resizable
    root.resizable(width=1280, height=1024)

    # Window Mainloop
    root.mainloop()
    app.on_close()
    return


# Main Function Trigger
if __name__ == '__main__':
    main()
