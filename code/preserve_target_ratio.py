import glob
import os
import shutil
import sys
import argparse
import random
import math

import numpy as np
from scipy import misc

# Source derived from @royveshovda source code posted into slack

def get_cam3_files(files):
    is_cam3 = lambda x: x.find('cam3_') != -1
    return sorted(list(filter(is_cam3, files)))


def get_cam3_file_id(filename, base_path):
    return filename.replace(base_path,'').replace('/','').replace('cam3_','').replace('.png','')


def delete_all_cam_files(id, path):
    c1 = path+'/'+'cam1_'+id+'.png'
    c2 = path+'/'+'cam2_'+id+'.png'
    c3 = path+'/'+'cam3_'+id+'.png'
    c4 = path+'/'+'cam4_'+id+'.png'
    delete_file(c1)
    delete_file(c2)
    delete_file(c3)
    delete_file(c4)


def delete_file(filename):
    try:
        os.remove(filename)
    except OSError:
        pass

def contains_hero(filename):
    s = np.sum(misc.imread(filename)[:,:,0])
    return s < 16711680


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('path',
                        help='The image path to filter')
    parser.add_argument('heropct', help="The percentage of remaining files that should have the hero")

    args = parser.parse_args()

    base_path = args.path
    files = glob.glob(os.path.join(base_path, '*.png'))
    cam3 = get_cam3_files(files)
    no_hero_files = []
    hero_files = []
    for f in cam3:
        if(not contains_hero(f)):
            no_hero_files.append(f)
        else:
            hero_files.append(f)
    random.shuffle(no_hero_files)
    random.shuffle(hero_files)

    total_files_to_keep = math.ceil(len(hero_files) / float(args.heropct))
    num_no_hero_files_to_keep = total_files_to_keep - len(hero_files)
    for f in no_hero_files[num_no_hero_files_to_keep:]:
        id = get_cam3_file_id(f, base_path)
        delete_all_cam_files(id, base_path)
