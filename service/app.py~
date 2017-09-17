from flask import Flask, jsonify, make_response, request
import argparse
import os
import json
import yolopy

app = Flask(__name__)

@app.route("/yolo/info")
def get_model_info():
	description = "model name: " + modelname + "\n"
	description += "description: " + desc + "\n"
	description += "weights: " + weights + "\n"
	description += "cfg: " + cfg + "\n"
	description += "names: " + names + "\n"
	description += "classes: " + str(classes) + "\n"
	return description

@app.route("/yolo/config/thresh/<int:thresh>", methods=['POST'])
def set_threshold(thresh):
	yolo.set_thresh(thresh)
	return ""

@app.route("/yolo/class/<string:class_name>", methods=['GET'])
def get_proposal(class_name):
	if class_name not in classes:
		return None

	#convert class string to idx
	cls_idx = classes.index(class_name)
	
	#receive json
	data = json.loads(request.data)
	img_to_decode = data["img"]

	#extract img
	temp_img_path = "/tmp/received.jpg"
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
def set_annotation(model_name, class_name):
	return "set annotation"

@app.route("/yolo/retrain", methods=['POST'])
def retrain_model(model_name):
	print "retraining model "

@app.errorhandler(500)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 500)

if __name__ == "__main__":
	global modelname, desc, cfg, names, weights, classes, yolo

	parse = argparse.ArgumentParser()
	parse.add_argument("--config", help="config file", required=True)
	args = parse.parse_args()

	with open(args.config) as data_file:
	    config = json.load(data_file)

	modelname = config["info"]["modelname"]
	desc = config["info"]["desc"]
	cfg = str(config["files"]["cfg"])
	weights = str(config["files"]["weights"])
	names = str(config["files"]["names"])
	classes = config["classes"]

	yolo = yolopy.YoloPython()
	yolo.init(weights, names, cfg)

	app.run(debug=False)


