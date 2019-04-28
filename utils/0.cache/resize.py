import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
from pathlib import Path
import os
from tqdm import tqdm
import argparse
import PIL

ap = argparse.ArgumentParser()
ap.add_argument("-wi", "--width", type=int, required=True, help="resize image width")
ap.add_argument("-hi", "--hight", type=int, required=True, help="resize image hight")
ap.add_argument("-s", "--source_dir", type=str, required=True, help="source dir")
ap.add_argument("-d", "--dist_dir", type=str, required=True, help="distination dir")
args = vars(ap.parse_args())

class resize_image():
    def __init__(self, w, h, src_dir, dist_dir, num_worker=4):
        self.w = w
        self.h = h
        self.src_dir = src_dir
        self.dist_dir = dist_dir
        self.num_worker = num_worker

        os.makedirs(self.dist_dir, exist_ok=True)

    def resize_img(self, path):
        PIL.Image.open(path).resize((self.w,self.h), resample=PIL.Image.BICUBIC).save(self.dist_dir+'/'+path.name)

    def run_resize(self):
        files = list(Path(self.src_dir).iterdir())
        for f in tqdm(files):
            if Path(f).is_file():
                self.resize_img(f)

resize_image(args['width'], args['hight'], args['source_dir'], args['dist_dir']).run_resize()
