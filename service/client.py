import requests
import cv2
import json
import base64 
import ntpath

host = "http://127.0.0.1:5000/yolo"

def get_model_info(model_name):
	return requests.get(host + "/info").content

def set_thresh(thresh):
	return requests.post(host + "/config/thresh/" + str(thresh)).content

def get_proposals(path, class_name):
	image = open(path, 'rb') #open binary file in read mode
	image_read = image.read()
	image_64_encode = base64.encodestring(image_read)

	print ntpath.basename(path)
	data = {"filename" : ntpath.basename(path)}
	data['img'] = image_64_encode
	json_data = json.dumps(data)
	return requests.get(host + "/class/" + class_name, data = json_data).content

def set_proposals(path, class_name, proposals):
	image = open(path, 'rb') #open binary file in read mode
	image_read = image.read()
	image_64_encode = base64.encodestring(image_read)

	data = {"filename": ntpath.basename(path)}
	data['img'] = image_64_encode
	data['proposals'] = proposals
	json_data = json.dumps(data)
	return requests.post(host + "/class/" + class_name, data = json_data).content

def main():
	print "Requesting model info from service...\n"
	info = get_model_info(host)
	print info

	print "Requesting region proposals for image test.jpg"
	set_thresh(0.16)
	rois = get_proposals("test.jpg", "rugs")
	print rois

	print set_proposals("test.jpg", "rugs", [[20,40,100,80],[0,0,35,40],[100,120,80,75]])

if __name__ == "__main__":
	main()
