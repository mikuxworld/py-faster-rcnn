import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
from voc_eval import voc_eval
from fast_rcnn.config import cfg


class mot(imdb):
    def __init__(self, image_set, devkit_path=None):
        imdb.__init__(self, image_set)
        self._image_set = image_set
        self._devkit_path = devkit_path
        self._data_path = os.path.join(self._devkit_path)
        self._classes = ('__background__', 'person')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_index = self._load_image_set_index('ImageList_MOT.txt')
        self._roidb_handler = self.gt_roidb
        self.config = {
            'cleanup': True,
            'use_salt': True,
            'top_k': 2000
        }
        assert os.path.exists(devkit_path), 'data path dose not exist: {}'.format(devkit_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, index)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self, imagelist):
        image_set_file = os.path.join(self._data_path, imagelist)
        assert os.path.exists(image_set_file), 'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        # NO CACHE
        # cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        # if os.path.exists(cache_file):
        #     with open(cache_file, 'rb') as fid:
        #         roidb = cPickle.load(fid)
        #     print '{} gt roidb loaded from {}'.format(self.name, cache_file)
        #     return roidb

        gt_roidb = self._load_mot_annotation()

        # with open(cache_file, 'wb') as fid:
        #     cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        # print 'wrote gt roidb to {}'.format(cache_file)
        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')  # if os.path.exists(cache_file):
        #     with open(cache_file, 'rb') as fid:
        #         roidb = cPickle.load(fid)
        #     print '{} ss roidb loaded from {}'.format(self.name, cache_file)
        #     return roidb
        roidb = self._load_mot_annotation()
        # with open(cache_file, 'wb') as fid:
        #     cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        # print 'wrote ss roidb to {}'.format(cache_file)
        return roidb

    def rpn_roidb(self):
        roidb = self._load_rpn_roidb(None)
        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
            'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        # useless
        # filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
        #                                       'selective_search_data',
        #                                       self.name + '.mat'))
        filename = os.path.join(self._data_path, 'ROISS_MOT.txt')
        assert os.path.exists(filename), \
            'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_mot_annotation(self):
        gt_roidb = []
        annotationfile = os.path.join(self._data_path, 'gt_bbox.txt')
        f = open(annotationfile)
        split_line = f.readline().strip().split()
        num = 1
        while (split_line):

            num_objs = int(split_line[1])
            boxes = np.zeros((num_objs, 4), dtype=np.int32)
            gt_classes = np.zeros((num_objs), dtype=np.int32)
            overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
            for i in range(num_objs):

                x1 = max(0,float(split_line[2 + i * 4]))
                y1 = max(0,float(split_line[3 + i * 4]))
                x2 = min(1919,float(split_line[4 + i * 4]))
                y2 = min(1079,float(split_line[5 + i * 4]))
                # x1 = float(split_line[2 + i * 4])
                # y1 = float(split_line[3 + i * 4])
                # x2 = float(split_line[4 + i * 4])
                # y2 = float(split_line[5 + i * 4])
                cls = self._class_to_ind['person']
                boxes[i, :] = [x1, y1, x2, y2]
                gt_classes[i] = cls
                overlaps[i, cls] = 1.0
            ds_utils.validate_boxes(boxes,1920,1080)


            overlaps = scipy.sparse.csr_matrix(overlaps)
            gt_roidb.append({'boxes': boxes, 'gt_classes': gt_classes, 'gt_overlaps': overlaps, 'flipped': False})
            split_line = f.readline().strip().split()

        f.close()
        return gt_roidb

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
                   else self._comp_id)
        return comp_id

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    from datasets.mot import mot
    d = mot('train', '/home/limingchen/exps/MOT17/train')
    res = d.roidb
    from IPython import embed; embed()
