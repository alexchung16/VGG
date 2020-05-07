#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : test_gpu.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/4/13 下午3:17
# @ Software   : PyCharm
#-------------------------------------------------------

import tensorflow as tf
import os
import math

if __name__ == "__main__":
    print(tf.test.is_gpu_available())
