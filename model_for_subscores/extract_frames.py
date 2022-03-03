#!/usr/bin/env python

from __future__ import print_function
import os, sys
import numpy as np
import cv2
import skvideo.io
import time

from os import listdir
from os.path import isfile, join, isdir


##############################################################################################################
def collect_files(dir_name, file_ext=".mp4", sort_files=True):
	allfiles = [os.path.join(dir_name,f) for f in listdir(dir_name) if isfile(join(dir_name,f))]

	these_files = []
	for i in range(0,len(allfiles)):
		_, ext = os.path.splitext(os.path.basename(allfiles[i]))
		if ext == file_ext:
			these_files.append(allfiles[i][-7:])

	if sort_files and len(these_files) > 0:
		these_files = sorted(these_files)

	return these_files


##############################################################################################################
def main(args=None, parser=None):

	data_dir = './diving/diving_samples_len_ori'
	images_dir = './frames'


	folder = images_dir

	file_dir = data_dir
	video_files = collect_files(file_dir, file_ext='.avi')
	# video_files = os.listdir(file_dir)
	# print video_files
	nVideos = len(video_files)

	start_time = time.time()
	for i in range(0,nVideos):
		print (i, '/', nVideos)

		vid_file = video_files[i]
		bn = os.path.basename(vid_file)
		print ("bn: ",bn )
		prefix = os.path.splitext(bn)[0]
		print (prefix)
		image_folder = join(folder, prefix)
		vid_file = join(data_dir, vid_file)
		print(vid_file)
		if not isdir(image_folder):
			os.mkdir(image_folder)
		videodata = skvideo.io.vread(vid_file)
	
		for id, img in enumerate(videodata):
			position = join(join(folder, prefix),"image_{:03d}.jpg".format(id))
			# print position
			cv2.imwrite(position, img)


	print('\nDONE\n')
	elapsed_time = time.time() - start_time
	print('time: ', elapsed_time)

	return 0


##############################################################################################################
if __name__ == '__main__':
	sys.exit(main())
