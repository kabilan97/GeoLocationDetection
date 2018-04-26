import numpy as np
import cv2
from video_tools import *
import argparse
import pickle
import sys
import image_search
import os.path
import PIL.Image
from PIL.ExifTags import TAGS
from PIL.ExifTags import GPSTAGS
import math
import webbrowser


parser = argparse.ArgumentParser(description="Query tool to query the database created by the database tool (dbt.py). Retrieve Geo location with input database and video.")
parser.add_argument("database", help="Path to the database to execute the query on.")
parser.add_argument("video", help="Query video")
parser.add_argument("frameint", help="Frame interval")
parser.add_argument("siftM", help="Number of sift matches per frame")

args = parser.parse_args()


base = os.path.splitext(args.database)[0]

search = image_search.Searcher(args.database)

def get_sift_features(im_list):
    """get_sift_features accepts an image and computes the sift descriptos. It returns a dictionary with descriptor as value and image name as key """
    sift = cv2.xfeatures2d.SIFT_create()
    features = {}
    count = 0
    kp, desc = sift.detectAndCompute(im_list, None)
    features[0] = desc
    count += 1
    return features

def call_Sift(imageName):
        fname = base + '_sift_vocabulary.pkl'
        # Load the vocabulary to project the features of our query frame on
        with open(fname, 'rb') as f:
            sift_vocabulary = pickle.load(f)
	
        sift_query = get_sift_features(imageName)[0]
	if sift_query != None:
        # Get a histogram of visual words for the query image
        	image_words = sift_vocabulary.project(sift_query)
        	print 'Query database with another frame of the video...'
        # Use the histogram to search the database
		sift_candidates = search.query_iw('sift', image_words)
		sift_winners = [search.get_filename(cand[1]) for cand in sift_candidates][0:int(args.siftM)]
        	return sift_winners
	return 'No Descriptors' 
	  

# starting point
S = 0 # seconds
# stop at
E = 60 # seconds
# Retrieve frame count. We need to add one to the frame count because cv2 somehow 
# has one extra frame compared to the number returned by avprobe.
frame_count = get_frame_count(args.video)
frame_rate = get_frame_rate(args.video)

# create an cv2 capture object
cap = cv2.VideoCapture(args.video)

# store previous frame
prev_frame = None

# set video capture object to specific point in time
cap.set(cv2.CAP_PROP_POS_MSEC, S*1000)

#getting the exif data with value field.
def get_field (exif,field) :
  for (k,v) in exif.iteritems():
     if TAGS.get(k) == field:
        return v
  return None

#Compute the euclidean distance between 2 vectors
def euclidean_distance(p,q):
	d = 0
	for i in range(len(p)):
		d+= (p[i]-q[i])**2
	return math.sqrt(d)

#Cluster the input data into a 3-means clustering and return the mean of the biggest cluster
def cluster(data):
	data.sort(key=lambda data: data[0])
#	first = randint(0,len(data) - 1)
	mean1 = data[0]

	#second = randint(int(len(data)/2))
	#while second == first:
#		second = randint(0,len(data) - 1)
	mean2 = data[int(len(data)/2)]

#	third = randint(0,len(data) - 1)
#	while third == first or third == second:
#		third = randint(0,len(data) - 1)
	mean3 = data[len(data) - 1]

	cluster1 = []
	cluster2 = []
	cluster3 = []

	while True:

		for coord in range(len(data)):
			distance1 = euclidean_distance(mean1,data[coord])
			distance2 = euclidean_distance(mean2,data[coord])
			distance3 = euclidean_distance(mean3,data[coord])
			index = 1
			if distance2 < distance1:
				index = 2
			if distance3 < distance1 and distance3 < distance2:
				index = 3
		
			if index == 1:
				cluster1.append(data[coord])
				
			if index == 2:
				cluster2.append(data[coord])
			if index == 3:
				cluster3.append(data[coord])

		index = 1
		if len(cluster1) < len(cluster2):
			index = 2
		if len(cluster2) < len(cluster3) and len(cluster1) < len(cluster3):
			index = 3

		averagex1 = 0
		averagey1 = 0
		for coord in cluster1:
			 averagex1 += coord[0]
			 averagey1 += coord[1]
		if len(cluster1) > 0:
			averagex1 = averagex1 / len(cluster1)
			averagey1 = averagey1 / len(cluster1)
		else:
			averagex1 = mean1[0]
			averagey1 = mean1[1]

		averagex2 = 0
		averagey2 = 0
		for coord in cluster2:
			 averagex2 += coord[0]
			 averagey2 += coord[1]
		if len(cluster2) > 0:
			averagex2 = averagex2 / len(cluster2)
			averagey2 = averagey2 / len(cluster2)
		else:
			averagex2 = mean2[0]
			averagey2 = mean2[1]

		averagex3 = 0
		averagey3 = 0
		for coord in cluster3:
			 averagex3 += coord[0]
			 averagey3 += coord[1]
		if len(cluster3) > 0:
			averagex3 = averagex3 / len(cluster3)
			averagey3 = averagey3 / len(cluster3)
		else:
			averagex3 = mean3[0]
			averagey3 = mean3[1]

		if index == 1:
			difference1 = euclidean_distance(mean1,[averagex1,averagey1])
			if difference1 < 0.001:
				return [averagex1,averagey1]

		if index == 2:
			difference1 = euclidean_distance(mean2,[averagex2,averagey2])
			if difference1 < 0.001:
				return [averagex2,averagey2]

		if index == 3:
			difference1 = euclidean_distance(mean3,[averagex3,averagey3])
			if difference1 < 0.001:
				return [averagex3,averagey3]

		mean1 = [averagex1,averagey1]
		mean2 = [averagex2,averagey2]
		mean3 = [averagex3,averagey3]
		
#Keep a counter for the amount of frames passed and a list which will hold all the gps-locations
counter = 0 
coordinates = []
while(cap.isOpened() and cap.get(cv2.CAP_PROP_POS_MSEC) < (E*1000)):

    # 
    retVal, frame = cap.read()
    counter += 1 

    # 
    if retVal == False:
        break

    #For each 30th frame, sift will look for matches in the database and append the gps-location in the coordinates list.
    if counter==int(args.frameint):
    	counter = 0
	winners  = call_Sift(frame)
    	if winners != 'No Discriptors':
		win = []
		for image in winners:
			img = PIL.Image.open(image)
			#print image
			data_exif = img._getexif()
			geotag = get_field(data_exif, "GPSInfo")
			if geotag == None:
				continue
			coordinate = [-1,-1]
			for key in geotag.keys():
    				decode = GPSTAGS.get(key,key)
				if decode == 'GPSLatitude':
					tuples = geotag[key]
					degree = float(tuples[0][0])/float(tuples[0][1])
					minutes = float(tuples[1][0])/float(tuples[1][1])
					seconds = float(tuples[2][0])/float(tuples[2][1])
					realDeg  = degree + (minutes/60) + (seconds/3600)
					coordinate[0] = realDeg
				if decode == 'GPSLongitude':
					tuples = geotag[key]
					degree = float(tuples[0][0])/float(tuples[0][1])
					minutes = float(tuples[1][0])/float(tuples[1][1])
					seconds = float(tuples[2][0])/float(tuples[2][1])
					realDeg  = degree + (minutes/60) + (seconds/3600)
					coordinate[1] = realDeg
			if (coordinate[0] != -1) or (coordinate[1] != -1):
    				coordinates.append(coordinate)
   

    prev_frame = frame

#Open the webbrowser and query the coordinates in googlemaps
endResult = cluster(coordinates)
print endResult
webbrowser.open('https://www.google.com/maps/search/?api=1&query=' +str(endResult[0])+',' + str(endResult[1]))
cap.release()
cv2.destroyAllWindows()
