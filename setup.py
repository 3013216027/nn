# -*- coding: utf-8 -*-
# @File    : setup.py
# @Author  : zhengdongjian
# @Time    : 2017/3/23 下午3:45
# @Desp    :
import os
import subprocess


def prepare_data():
    print('Downloading data...', end='')
    subprocess.run(['./data/get_data.sh'])
    print('Done!')


if __name__ == '__main__':
    prepare_data()
