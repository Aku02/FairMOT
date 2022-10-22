from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import os.path as osp
import cv2
import logging
import argparse
# import motmetrics as mm
import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
from tracker.multitracker import JDETracker
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
import lib.datasets.dataset.jde as datasets
from models.model import create_model, load_model

from tracking_utils.utils import mkdir_if_missing
from opts import opts


def write_results_score(filename, results):
    res_df = []
    for path, frame_id, tlwhs, track_ids, scores in results:
        for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
            if track_id < 0:
                continue
            x1, y1, w, h = tlwh
            x2, y2 = x1 + w, y1 + h
            res_df.append([path, frame_id, track_id, round(x1), round(y1),
                        round(x2), round(y2), round(w), round(h), score])

    res_df = pd.DataFrame(res_df, columns=['path', 'frame', 'track_id', 'left', 'top',
                                          'x2', 'y2', 'width', 'height', 'conf'])

    res_df.to_csv(filename, index=False)
    logger.info(f'save results to {filename}')


def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, save_image=True, frame_rate=30, use_cuda=True, model=None):
    if not torch.cuda.is_available():
        use_cuda = False

    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = JDETracker(opt, frame_rate=frame_rate, model=model)
    timer = Timer()
    results = []
    frame_id = 0
    cur_vid = ''
    for path, img, img0 in tqdm(dataloader):
        # path is: .../.../57594_000923_Endzone_1
        # this_vid = "_".join(os.path.basename(path).split('_')[:3])

        # run tracking
        timer.tic()
        if use_cuda:
            blob = torch.from_numpy(img).cuda().unsqueeze(0)
        else:
            blob = torch.from_numpy(img).unsqueeze(0)

        online_targets = tracker.update(blob, img0)
        online_tlwhs = []
        online_ids = []
        online_scores = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)
        timer.toc()
        results.append((path, frame_id + 1, online_tlwhs, online_ids, online_scores))

        if save_image or save_dir is not None:
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / timer.average_time)
        if save_dir is not None and save_image:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        frame_id += 1


    write_results_score(result_filename, results)
    return frame_id, timer.average_time, timer.calls


def main(opt):
    save_images = not opt.nosave
    save_videos = not opt.nosave

    logger.setLevel(logging.INFO)

    if os.path.exists(opt.output_dir):
        import shutil
        shutil.rmtree(opt.output_dir)
    os.makedirs(opt.output_dir)

    data_type = 'mot'

    # run tracking
    n_frame = 0
    timer_avgs, timer_calls = [], []

    dataloader = datasets.LoadImages(opt.image_dir, opt.img_size, video_id=opt.video_id)
    result_filename = os.path.join(opt.output_dir, 'fm_tracking.csv')
    nf, ta, tc = eval_seq(opt, dataloader, data_type, result_filename,
                            save_dir=opt.output_dir, save_image=save_images, frame_rate=opt.frame_rate)
    n_frame += nf
    timer_avgs.append(ta)
    timer_calls.append(tc)

    # eval
    if save_videos:
        output_video_path = osp.join(opt.output_dir, 'demo.mp4')
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(opt.output_dir, output_video_path)
        os.system(cmd_str)
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    print('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    opt = opts().init()
    main(opt)
