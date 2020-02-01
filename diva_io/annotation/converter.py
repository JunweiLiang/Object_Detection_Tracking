import os
import json
import argparse
import os.path as osp
from progressbar import progressbar
from concurrent.futures import ProcessPoolExecutor
from ..utils import get_logger
from .kf1 import KitwareAnnotation


def _get_video_list(annotation_dir):
    path = osp.join(annotation_dir, 'list-of-annotated-meva-clips.txt')
    with open(path) as f:
        video_list = [l.strip() for l in f][2:]
    return video_list


def _worker(job):
    video_name, annotation_dir = job
    annotation = KitwareAnnotation(video_name, annotation_dir)
    return annotation.get_activities_official()


def _get_official_format(video_list, annotation_dir):
    jobs = [(video_name, annotation_dir) for video_name in video_list]
    pool = ProcessPoolExecutor()
    activities = []
    for result in progressbar(pool.map(_worker, jobs)):
        activities.extend(result)
    reference = {'filesProcessed': video_list, 'activities': activities}
    file_index = {video_name: {'framerate': 30.0, 'selected': {0: 1, 9000: 0}}
                  for video_name in video_list}
    return reference, file_index


def _write_files(data_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logger = get_logger(__name__)
    for filename, data in data_dict.items():
        path = osp.join(output_dir, filename + '.json')
        if osp.exists(path):
            logger.warning('Overwriting file %s', path)
        with open(path, 'w') as f:
            json.dump(data, f)


def convert_annotation(annotation_dir, output_dir):
    video_list = _get_video_list(annotation_dir)
    reference, file_index = _get_official_format(video_list, annotation_dir)
    data_dict = {'reference': reference, 'file-index': file_index}
    _write_files(data_dict, output_dir)


def main():
    parser = argparse.ArgumentParser(
        'Annotation Converter for KF1, from Kitware YML format to '
        'ActEV Scorer JSON format.')
    parser.add_argument('annotation_dir')
    parser.add_argument('output_dir')
    args = parser.parse_args()
    convert_annotation(args.annotation_dir, args.output_dir)


if __name__ == "__main__":
    main()
