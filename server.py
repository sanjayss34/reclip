import csv
import gzip
import json
import os
import random
import re
import sys
import numpy as np
from PIL import Image
from typing import List
import argparse
import base64
import torch
import io

from flask import Flask, render_template, request, send_from_directory

# image_root = "/Users/sanjays/Documents/refer/train2014/"
image_root = os.environ["TRAIN2014"]

app = Flask(__name__,
            static_url_path='/static',
            # static_folder='../../nlvr2_dataset/images')
            static_folder=image_root)
            # static_folder='/Users/sanjays/Documents/visual-genome/VG_100K/')
            # static_folder='/Users/sanjays/Documents/nlvr2_images/custom_test')

@app.route('/')
def index():
    questions = []
    page = int(request.args.get('page', 1))
    model = request.args.get('model', 'predictions')
    prediction_file_name = model+".json"
    try:
        f = open(prediction_file_name)
        data = json.load(f)
    except Exception as e:
        data = torch.load(prediction_file_name, map_location='cpu')
    results_per_page = 25

    data_to_display = []

    for datum in data[(page-1)*results_per_page:page*results_per_page]:
        if os.path.exists(os.path.join(image_root, datum["file_name"])):
            img = Image.open(os.path.join(image_root, datum["file_name"]))
            for i in range(len(datum["bboxes"])):
                datum["bboxes"][i] = [datum["bboxes"][i][0]/img.width, datum["bboxes"][i][1]/img.height, datum["bboxes"][i][2]/img.width, datum["bboxes"][i][3]/img.height]
            """output = io.BytesIO()
            img.save(output, format='JPEG')
            output.seek(0, 0)
            datum["img"] = base64.b64encode(output.getvalue())"""
            data_to_display.append(datum)

    page_params = {
        'examples': data_to_display, # data_to_display[(page-1)*results_per_page:page*results_per_page],
        'next_page': f'/?model={model}&page={str(page+1)}' if page*results_per_page < len(data) else None,
        'prev_page': f'/?model={model}&page={str(page-1)}' if page > 1 else None,
        'total': len(data_to_display),
        'prediction_file_name': prediction_file_name,
    }
    template_file = 'template.html'
    return render_template(template_file, **page_params)


if __name__ == '__main__':
    # with app.test_request_context(path="?model=gnn+detach_simple+pattend"):
    #     hello()
    app.run(host='0.0.0.0', port=8083, debug=True)
