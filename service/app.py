from flask import Flask, jsonify, make_response, request
import argparse
import os
import json
import cv2
import yolopy

app = Flask(__name__)

def calc_yolo_annotation(anno, width, height):
	x,y,w,h = anno
	xc = (x+w / 2.0) / width
	yc = (y+h / 2.0) / height
	wc = w / float(width)
	hc = h / float(height)
	return [xc, yc, wc, hc]

@app.route("/yolo/info")
def get_model_info():
	description = "model name: " + modelname + "\n"
	description += "description: " + desc + "\n"
	description += "weights: " + weights + "\n"
	description += "cfg: " + cfg + "\n"
	description += "names: " + names + "\n"
	description += "classes: " + str(classes) + "\n"
	return description

@app.route("/yolo/config/thresh/<float:thresh>", methods=['POST'])
def set_threshold(thresh):
	yolo.set_thresh(thresh)
	return ""

@app.route("/yolo/class/<string:class_name>", methods=['GET'])
def get_proposal(class_name):
	if class_name not in classes:
		return None

	cls_idx = classes.index(class_name)
	
	#receive json
	data = json.loads(request.data)
	img_to_decode = data["img"]

	filename = data["filename"]

	#extract img
	temp_img_path = os.path.join("/tmp",str(filename))
	with open(temp_img_path, "wb") as fh:
    		fh.write(img_to_decode.decode('base64'))

	#communicate with yolopy
	count = yolo.detect(temp_img_path, cls_idx)

	rois = []
	#convert proposals to internal format
	for i in range(0,count):
		rois.append([yolo.comp(i,0),yolo.comp(i,1),yolo.comp(i,2),yolo.comp(i,3)])
	return jsonify(rois)

@app.route("/yolo/class/<string:class_name>", methods=['POST'])
def set_annotation(class_name):
	if class_name not in classes:
		return None

	cls_idx = classes.index(class_name)

	#receive json
	data = json.loads(request.data)
	img_to_decode = data["img"]
	annotation = data["proposals"]
	filename = data["filename"]

	#extract img to the images folder
	temp_img_path = os.path.join(img_target, str(filename))
	with open(temp_img_path, "wb") as fh:
    		fh.write(img_to_decode.decode('base64'))

	#get image size
	img = cv2.imread(temp_img_path)
	w = img.shape[1]
	h = img.shape[0]

	#save annotation in the correct format
	anno_file = os.path.splitext(filename)[0]+'.txt'
	anno_file = os.path.join(ann_target, anno_file)

	f = open(anno_file,"w+")
	for anno in annotation:
		anno_y = calc_yolo_annotation(anno, w, h)
		f.write(str(cls_idx) + " " + str(anno_y[0]) + " " + str(anno_y[1]) + " " + str(anno_y[2]) + " " + str(anno_y[3]) + "\n")
	return jsonify(annotation)

@app.route("/yolo/retrain", methods=['POST'])
def retrain_model(model_name):
	print "retraining model "

@app.errorhandler(500)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 500)

if __name__ == "__main__":
	global modelname, desc, cfg, names, weights, classes, yolo, target

	parse = argparse.ArgumentParser()
	parse.add_argument("--config", help="config file", required=True)
	args = parse.parse_args()

	with open(args.config) as data_file:
	    config = json.load(data_file)

	modelname = config["info"]["modelname"]
	desc = config["info"]["desc"]
	cfg = str(config["path"]["cfg"])
	weights = str(config["path"]["weights"])
	names = str(config["path"]["names"])
	target = str(config["path"]["target"])
	classes = config["classes"]

	img_target = os.path.join(target, "images")
	ann_target = os.path.join(target, "labels")

	if not os.path.exists(img_target):
	    os.makedirs(img_target)

	if not os.path.exists(ann_target):
	    os.makedirs(ann_target)

	yolo = yolopy.YoloPython()
	yolo.init(weights, names, cfg)

	app.run(debug=False)


