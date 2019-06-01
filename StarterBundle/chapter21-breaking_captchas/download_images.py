# USAGE
# python download_images.py --output downloads --num-images 500
import argparse
import requests
import time
import os

# construct argument parser and parse argument
ap = argparse.ArgumentParser()
ap.add_argument('-o', '--output', required=True, help='path to output directory')
ap.add_argument('-n', '--num-images', type=int, default=500, help='number of images to download')
args= vars(ap.parse_args())

# initialize URL that contains captcha images that we will be downloading along with total number of images downloaded
url = 'https://www.e-zpassny.com/vector/jcaptcha.do'
total = 0

# loop over number of images to download
for i in range(args['num_images']):
    try:
        # grab new captcha images
        r = requests.get(url, timeout=60)

        # save image to disk
        p = os.path.sep.join([args['output'], f'{str(total).zfill(5)}.jpg']) # f'{total:05d}.jpg'
        f = open(p, 'wb')
        f.write(r.content)
        f.close()

        # update counter
        print(f'[INFO] downloaded: {p}...')
        total += 1

    # handle any error
    except:
        print('[INFO] error download image...')
    
    time.sleep(0.1)