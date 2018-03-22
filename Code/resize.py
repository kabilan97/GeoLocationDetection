#!/usr/bin/env python

import glob
import cv2
import argparse
import pyexiv2

parser = argparse.ArgumentParser(description="Resize tool to resize a batch of images.")
parser.add_argument("image_path", help="The path to the images")
parser.add_argument("width", help="New width in pixels")
parser.add_argument("--height", help="New height in pixels, when preserve is enabled, this value is ignored", default=100)
parser.add_argument("-p", "--preserve", help="Preserve aspect ratio. set to 'False' to disable. Default is True", default='true')
parser.add_argument("-o","--output",  help="Output path for resize images. If not specified, original images will be overwritten", default='')

args = parser.parse_args()

print "\nResize tool"
print "============\n"

# retrieve image list

types = ('*.jpg', '*.JPG', '*.png')
image_list = []
for type_ in types:
    files = args.image_path + type_
    image_list.extend(glob.glob(files))	

for image in image_list:
    im = cv2.imread(image)

    h,w,d = im.shape
    new_width = int(args.width)
    if args.preserve.lower() == 'true':
        ratio = float(w)/new_width
        new_height = int(h / ratio)
    else:
        new_height = int(args.height)
    im = cv2.resize(im, (new_width,new_height))
    if (args.output != ''):
        outfile = args.output +'/'+ image.split('/')[-1]
    else:
        outfile = image
    print 'resized: ', outfile
    
    
    # copy and correct metadata
    source_image = pyexiv2.ImageMetadata(image)
    source_image.read()
    cv2.imwrite(outfile, im)
    
    dest_image = pyexiv2.ImageMetadata(outfile)
    dest_image.read()
    source_image.copy(dest_image)
    dest_image["Exif.Photo.PixelXDimension"] = new_width
    dest_image["Exif.Photo.PixelYDimension"] = new_height
    dest_image.write()

	
