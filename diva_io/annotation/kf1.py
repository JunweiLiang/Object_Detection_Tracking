import yaml
import os.path as osp
from collections import defaultdict


FIELDS = ['activities', 'geom', 'types']


class KitwareAnnotation(object):

    def __init__(self, video_name: str, annotation_dir: str):
        # Please explore the structure of raw_data yourself
        self.video_name = video_name
        self.raw_data = self._load_raw_data(video_name, annotation_dir)

    def _split_meta(self, contents, key):
        meta = []
        i = 0
        while i < len(contents) and 'meta' in contents[i]:
            assert key not in contents[i]
            meta.append(contents[i]['meta'])
            i += 1
        data = [content[key] for content in contents[i:]]
        return meta, data

    def _load_file(self, video_name, annotation_dir, field):
        date, time_1, time_2 = video_name.split('.')[:3]
        for time in [time_1, time_2]:
            path = osp.join(annotation_dir, date, time[:2], '%s.%s.yml' % (
                video_name, field))
            if not osp.exists(path):
                continue
            with open(path) as f:
                contents = yaml.load(f, Loader=yaml.FullLoader)
            return contents
        path = osp.join(annotation_dir, date, time_1[:2], '%s.%s.yml' % (
            video_name, field))
        raise FileNotFoundError(path)

    def _load_raw_data(self, video_name, annotation_dir):
        raw_data = {'meta': {}}
        for field in FIELDS:
            contents = self._load_file(video_name, annotation_dir, field)
            key = field if field != 'activities' else 'act'
            raw_data['meta'][field], raw_data[field] = self._split_meta(
                contents, key)
        objs = defaultdict(dict)
        for obj in raw_data['geom']:
            obj['g0'] = [int(x) for x in obj['g0'].split()]
            objs[obj['id1']][obj['ts0']] = obj
        for obj in raw_data['types']:
            objs[obj['id1']]['type'] = [*obj['cset3'].keys()][0]
        for act in raw_data['activities']:
            for actor in act.get('actors', []):
                obj = objs[actor['id1']]
                geoms = []
                for ts in actor['timespan']:
                    start, end = ts['tsr0']
                    for time in range(start, end + 1):
                        geoms.append(obj[time])
                actor['geoms'] = geoms
                actor['type'] = obj['type']
        return raw_data

    def get_activities_official(self):
        activities = []
        for act in self.raw_data['activities']:
            act_id = act['id2']
            act_type = [*act['act2'].keys()][0]
            if act_type.startswith('empty'):
                continue
            start, end = act['timespan'][0]['tsr0']
            objects = []
            for actor in act['actors']:
                actor_id = actor['id1']
                bbox_history = {}
                for geom in actor['geoms']:
                    frame_id = geom['ts0']
                    x1, y1, x2, y2 = geom['g0']
                    bbox_history[frame_id] = {
                        'presenceConf': 1,
                        'boundingBox': {
                            'x': min(x1, x2), 'y': min(y1, y2),
                            'w': abs(x2 - x1), 'h': abs(y2 - y1)}}
                for frame_id in range(start, end + 1):
                    if frame_id not in bbox_history:
                        bbox_history[frame_id] = {}
                obj = {'objectType': 'Vehicle', 'objectID': actor_id,
                       'localization': {self.video_name: bbox_history}}
                objects.append(obj)
            activity = {
                'activity': act_type, 'activityID': act_id,
                'presenceConf': 1, 'alertFrame': start,
                'localization': {self.video_name: {start: 1, end + 1: 0}},
                'objects': objects}
            activities.append(activity)
        return activities
