import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2#for ckpt.meta
import matplotlib.pyplot as plt
import numpy as np
import os, sys, cv2
import argparse
import os.path as osp
import glob

this_dir = osp.dirname(__file__)
sys.path.insert(0, this_dir + '/..')
print(this_dir)

from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg
from lib.fast_rcnn.test import im_detect
from lib.fast_rcnn.nms_wrapper import nms
from lib.utils.timer import Timer

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')


#CLASSES=('__background__','Pedestrain','Car','Cyclist')
#CLASSES=('__background__','Van','Tram','Truck','Pedestrain','Car','Cyclist')



if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals


    # init session
    demo_net='VGGnet_test'


    model='/home/clark/cvProject1/data/VGGnet_fast_rcnn_iter_150000.ckpt'
   

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True
    sess = tf.Session(config=tfconfig)

    # load network
    net = get_network(demo_net)
    # load model
    print ('Loading network {:s}... '.format(demo_net)),
    saver = tf.train.Saver()
    saver.restore(sess, model)
    print (' done.')

    im_names = glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.png')) + \
               glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.jpg'))#lib_video
    for im_name in im_names:
        print ('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print( 'Demo for {:s}'.format(im_name))

        im = cv2.imread(im_name)

    # Detect all object classes and regress object bounds
        timer = Timer()
        timer.tic()
        scores, boxes = im_detect(sess, net, im)
        timer.toc()
        print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
        im = im[:, :, (2, 1, 0)]
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(im, aspect='equal')

        CONF_THRESH = 0.7
        NMS_THRESH = 0.3
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1  # because we skipped background
            cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            class_name=cls
            thresh=CONF_THRESH
            """Draw detected bounding boxes."""
            inds = np.where(dets[:, -1] >= thresh)[0]
            if len(inds) == 0:
                pass

            for i in inds:
                bbox = dets[i, :4]
                score = dets[i, -1]
                
                ax.add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], fill=False,edgecolor='red', linewidth=3.5))
                ax.text(bbox[0], bbox[1] - 2,'{:s} {:.3f}'.format(cls, score),bbox=dict(facecolor='blue', alpha=0.5),fontsize=14, color='white')
                print class_name,score

#            ax.set_title(('{} detections with ' 'p({} | box) >= {:.1f}').format(class_name, class_name,thresh),fontsize=14)
            plt.axis('off')
            plt.tight_layout()
            plt.draw()
    plt.show()

