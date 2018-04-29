import os
import glob
import re
import argparse
from sklearn.model_selection import train_test_split
import shutil

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path',help='The image path to split up')

    args = parser.parse_args()

    mask_files = []
    infiles = glob.glob(os.path.join(args.path, 'train','images','*.jpeg'))
    for f in infiles:
        m = re.search(r'.*\/(.*)_cam[0-9]_(\d+)\.jpeg',f)
        if m:
            mask_file = os.path.join(args.path, 'train', 'masks', "{}_mask_{}.png".format(m.group(1),m.group(2)))
            mask_files.append(mask_file)

   
    X_train, X_test, y_train, y_test = train_test_split(infiles, mask_files, test_size=0.2, random_state=42)

    for f in X_test:
        m = re.search(r'(.*)\/train\/(.*)',f)
        if m:
            vfile = os.path.join(m.group(1),'validation',m.group(2))
            shutil.move(f, vfile)
    for f in y_test:
        m = re.search(r'(.*)\/train\/(.*)',f)
        if m:
            vfile = os.path.join(m.group(1),'validation',m.group(2))
            shutil.move(f, vfile)


