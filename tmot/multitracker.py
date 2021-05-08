# coding=utf-8

import warnings
warnings.filterwarnings('ignore')  # remove numba warnings

from numba import jit
from collections import deque
from .kalman_filter import KalmanFilter
from . import matching
from .basetrack import BaseTrack, TrackState
import numpy as np

class STrack(BaseTrack):

    def __init__(self, tlwh, score, temp_feat, buffer_size=30, alpha=0.9):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.update_features(temp_feat)
        self.features = deque([], maxlen=buffer_size)
        self.alpha = alpha

        # use detection box directly
        self.cur_det_tlwh = np.asarray(tlwh, dtype=np.float)

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha *self.smooth_feat + (1-self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks, kalman_filter):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
#            multi_mean, multi_covariance = STrack.kalman_filter.multi_predict(multi_mean, multi_covariance)
            multi_mean, multi_covariance = kalman_filter.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        #self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )

        self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        # save the detection box as well
        self.cur_det_tlwh = new_track.cur_det_tlwh

    def update(self, new_track, frame_id, update_feature=True):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)

        # save the detection box as well
        self.cur_det_tlwh = new_track.cur_det_tlwh

    @property
    @jit
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    @jit
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    @jit
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    @jit
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    @jit
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class JDETracker(object):
    def __init__(self, conf_thres, track_max_second_lost=4.0,
                 emb_max_dist=0.7, iou_max_dist1=0.8, iou_max_dist2=0.9,
                 emb_smooth_alpha=0.9,
                 frame_gap=8.,
                 frame_rate=30.):

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.det_thresh = conf_thres


        #self.buffer_size = track_buffer
        # track will be removed after this number of frame
        # the frame is in the tracking frame
        # so this should be max_seconds * frame_rate / frame_gap
        self.max_frame_lost = track_max_second_lost * frame_rate / frame_gap

        # distance larger than these will be masked out
        self.emb_max_dist = emb_max_dist
        self.iou_max_dist1 = iou_max_dist1
        self.iou_max_dist2 = iou_max_dist2

        self.emb_smooth_alpha = emb_smooth_alpha

        self.kalman_filter = KalmanFilter()

    def reset(self):
        # finished running on one video
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.kalman_filter = KalmanFilter()

        BaseTrack._count = 0  # reset the track_id counter

    def update(self, detections):
        """
        Processes the NMSed outputs, # xywh, conf, feature


        Returns
        -------
        output_stracks : list of Strack(instances)
                         The list contains information regarding the online_tracklets for the recieved image tensor.

        """

        self.frame_id += 1 # local frame id count?
        activated_starcks = []      # for storing active tracks, for the current frame
        refind_stracks = []         # Lost Tracks whose detections are obtained in the current frame
        lost_stracks = []           # The tracks which are not obtained in the current frame but are not removed.(Lost for some time lesser than the threshold for removing)
        removed_stracks = []

        detections = [STrack(tlwhs, conf, f, buffer_size=30, alpha=self.emb_smooth_alpha)
                      for (tlwhs, conf, f) in detections]


        # print('Forward: {} s'.format(t2-t1))

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                # previous tracks which are not active in the current frame are added in unconfirmed list
                unconfirmed.append(track)
                # print("Should not be here, in unconfirmed")
            else:
                # Active tracks are added to the local list 'tracked_stracks'
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''
        # Combining currently tracked_stracks and lost_stracks
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool, self.kalman_filter)


        dists = matching.embedding_distance(strack_pool, detections)
        # dists = matching.gate_cost_matrix(self.kalman_filter, dists, strack_pool, detections)
        dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections)
        # The dists is the list of distances of the detection with the tracks in strack_pool
        matches, u_track, u_detection = matching.linear_assignment(
                dists, thresh=self.emb_max_dist)
        # The matches is the array for corresponding matches of the detection with the corresponding strack_pool

        for itracked, idet in matches:
            # itracked is the id of the track and idet is the detection
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                # If the track is active, add the detection to the track
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                # We have obtained a detection from a track which is not active, hence put the track in refind_stracks list
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # None of the steps below happen if there are no undetected tracks.
        ''' Step 3: Second association, with IOU'''
        detections = [detections[i] for i in u_detection]
        # detections is now a list of the unmatched detections
        r_tracked_stracks = [] # This is container for stracks which were tracked till the
        # previous frame but no detection was found for it in the current frame
        for i in u_track:
            if strack_pool[i].state == TrackState.Tracked:
                r_tracked_stracks.append(strack_pool[i])
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(
                dists, thresh=self.iou_max_dist1)
        # matches is the list of detections which matched with corresponding tracks by IOU distance method
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        # Same process done for some unmatched detections, but now considering IOU_distance as measure

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        # If no detections are obtained for tracks (u_track), the tracks are added to lost_tracks list and are marked lost

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(
                dists, thresh=self.iou_max_dist2)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])

        # The tracks which are yet not matched
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # after all these confirmation steps, if a new detection is found, it is initialized for a new track
        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        # If the tracks are lost for more frames than the threshold number, the tracks are removed.
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_frame_lost:
                track.mark_removed()
                removed_stracks.append(track)
        # print('Remained match {} s'.format(t4-t3))

        # Update the self.tracked_stracks and self.lost_stracks using the updates in this step.
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        # self.lost_stracks = [t for t in self.lost_stracks if t.state == TrackState.Lost]  # type: list[STrack]
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res

def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())

def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist<0.15)
    dupa, dupb = list(), list()
    for p,q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i,t in enumerate(stracksa) if not i in dupa]
    resb = [t for i,t in enumerate(stracksb) if not i in dupb]
    return resa, resb


