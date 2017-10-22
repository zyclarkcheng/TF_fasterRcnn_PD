#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 11:21:50 2017

@author: clark
"""
import numpy as np


def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors

def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors
#
base_size=16
base_anchor = np.array([1, 1, base_size, base_size]) - 1
ratios=[0.5, 1, 2]
#ratio_anchors = _ratio_enum(base_anchor, ratios)

w, h, x_ctr, y_ctr = _whctrs(base_anchor)#0 0 15 15-> 16 16 7.5 7.5
#use center of base anchor to generate 9 anchors
size = w * h
size_ratios = size / ratios#256/[0.5,1,2]->512 256 128
ws = np.round(np.sqrt(size_ratios))#23 16 11
hs = np.round(ws * ratios)#12 16 22 
ratio_anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
scales=2**np.arange(3, 6)#8 16 32 
re_anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in xrange(ratio_anchors.shape[0])])

    
width=14
height=14
_feat_stride=16
shift_x = np.arange(0, width) * _feat_stride
shift_y = np.arange(0, height) * _feat_stride
shift_x, shift_y = np.meshgrid(shift_x, shift_y)
shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()
A =9 
K = shifts.shape[0]
anchors = re_anchors.reshape((1, A, 4)) + \
              shifts.reshape((1, K, 4)).transpose((1, 0, 2))
anchors = anchors.reshape((K * A, 4))

all_anchors=anchors
_allowed_border=0
im_info=[224,224,1]
inds_inside = np.where(
        (all_anchors[:, 0] >= -_allowed_border) &
        (all_anchors[:, 1] >= -_allowed_border) &
        (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
        (all_anchors[:, 3] < im_info[0] + _allowed_border)    # height
    )[0]

in_anchors = all_anchors[inds_inside, :]