import av
#import queue
import logging
import numpy as np
import os.path as osp
#from typing import Tuple

from ..utils import get_logger
from .frame import Frame


class VideoReader(object):

    def __init__(self, video_path, parent_dir = '',
                 fix_missing = True, silence_warning = True):
        """Read frames from a video file.

        Parameters
        ----------
        video_path : str
            Path of the video file, will be joint with parent_dir if specified.

        parent_dir : str, optional
            Parent directory of the videos, convenient for path management, by
            default ''.

        fix_missing : bool, optional
            Whether to fix missing frames.

        silence_warning : bool, optional
            Whether to silence warnings from ffmpeg and warnings about missing
            frames when `fix_missing=True`. Warnings about missing frames are
            not silenced when `fix_missing=False`.

        Raises
        ------
        FileNotFoundError
            If the video file to read does not exist.
        """
        self.path = osp.join(parent_dir, video_path)
        if not osp.exists(self.path):
            raise FileNotFoundError(self.path)
        if silence_warning:
            self._logger = get_logger(
                '%s@%s' % (__name__, self.path), logging.WARNING)
            av.logging.set_level(av.logging.FATAL)
        else:
            self._logger = get_logger('%s@%s' % (__name__, self.path))
        self._assert_msg = ' Please report %s to Lijun.' % (self.path)
        self.fix_missing = fix_missing
        if not self.fix_missing:
            self._logger.warning('Not fixing missing frames.')
        self._init()
        self.length = self._stream.duration
        self.fps = float(self._stream.average_rate)
        self.height = self._stream.codec_context.format.height
        self.width = self._stream.codec_context.format.width
        self.shape = (self.height, self.width)

    def __del__(self):
        if hasattr(self, '_container'):
            self._container.close()

    def __iter__(self):
        """Iterator interface to use in a for-loop directly as:
        for frame in video:
            pass

        Yields
        -------
        Frame
            A Frame object.
        """
        self.reset()
        #yield from self._frame_gen
        for f in self._frame_gen:
            yield f

    def get_iter(self, limit= None, cycle = 1):
        """Get an iterator to yield a frame every cycle frames and stop at a
        limited number of yielded frames.

        Parameters
        ----------
        limit : int, optional
            Total number of frames to yield, by default None. If None, it
            yields until the video ends.

        cycle : int, optional
            The cycle length for each read, by default 1. If cycle = 1, no
            frames are skipped.

        Yields
        -------
        Frame
            A Frame object.
        """
        if limit is None or limit > self.length:
            limit = self.length
        for _ in range(limit):
            try:
                yield self.get_skip(cycle)
            except StopIteration:
                break

    def get_skip(self, cycle= 1):
        """Read a frame from the video every cycle frames. It returns the
        immediate next frame and skips cycle - 1 frames for the next call of
        get.

        Parameters
        ----------
        cycle : int, optional
            The cycle length for each read, by default 1. If cycle = 1, no
            frames are skipped.

        Returns
        -------
        Frame
            A Frame object.

        Raises
        -------
        StopIteration
            When the video ends.
        """
        frame = self.get()
        try:
            for _ in range(cycle - 1):
                self.get()
        except StopIteration:  # will be raised in the next call of get
            pass
        return frame

    def get(self):
        """Read the next frame from the video.

        Returns
        -------
        Frame
            The frame object.

        Raises
        -------
        StopIteration
            When the video ends.
        """
        return next(self._frame_gen)

    def read(self):
        """Read the next frame from the video. Following the API of
        cv2.VideoCapture.read() for consistency in old codes. For new codes,
        the get method is recommended.

        Returns
        -------
        bool
            True when the read is successful, False when the video ends.

        numpy.ndarray
            The frame when successful, with format bgr24, shape (height, width,
            channel) and dtype int.
        """
        try:
            frame = next(self._frame_gen)
            frame = frame.numpy()
        except StopIteration:
            frame = None
        return frame is not None, frame

    def reset(self):
        """Reset the internal states to load the video from the beginning.
        """
        self._container.close()
        self._init()

    def seek(self, frame_id):
        """Seek to a specific position in the video.

        Parameters
        ----------
        frame_id : int
            Target frame id for next `get` call.

        Raises
        -------
        ValueError
            If the frame_id does not exist.
        """
        if frame_id >= self.length:
            raise ValueError(
                'Cannot seek frame id %d in a video of length %d' % (
                    frame_id, self.length))
        self._frame_gen = self._get_frame_gen(frame_id)

    def get_at(self, frame_id):
        """Get a specific frame.

        Parameters
        ----------
        frame_id : int
            Target frame id for next `get` call.

        Raises
        -------
        ValueError
            If the frame_id does not exist.
        """
        self.seek(frame_id)
        return self.get()

    def _init(self):
        self._container = av.open(self.path)
        self._stream = self._container.streams.video[0]
        #self._packets = [*self._container.demux(self._stream)][:-1]
        self._packets = [o for o in self._container.demux(self._stream)][:-1]
        self._key_frame_ids = [
            i for i, p in enumerate(self._packets) if p.is_keyframe]
        assert len(self._packets) == self._stream.duration
        #assert [p.pts for p in self._packets] == [*range(
            #1, len(self._packets) + 1)], \
            #'Packets pts not in order' + self._assert_msg
        assert [p.pts for p in self._packets] == [o for o in range(
            1, len(self._packets) + 1)], \
            'Packets pts not in order' + self._assert_msg
        self._frame_gen = self._get_frame_gen()

    def _get_frame_gen(self, start_frame_id=0):
        for key_frame_index, frame_id in enumerate(self._key_frame_ids):
            if frame_id > start_frame_id:
                break
        frame = None
        key_frame_id = self._key_frame_ids[key_frame_index]
        while frame is None and key_frame_index > 0:
            key_frame_index -= 1
            key_frame_id = self._key_frame_ids[key_frame_index]
            frame = self._decode(key_frame_id)
        prev_frame = frame
        for frame_id in range(key_frame_id + 1, start_frame_id):
            frame = self._decode(frame_id)
            if frame is not None:
                prev_frame = frame
        for frame_id in range(start_frame_id, self.length):
            frame = self._decode(frame_id)
            if frame is not None:
                yield frame
                prev_frame = frame
            else:
                if self.fix_missing:
                    self._logger.info('Missing frame %d, used frame %d',
                                      frame_id, prev_frame.frame_id)
                    assert prev_frame is not None
                    yield prev_frame
                else:
                    self._logger.warning('Missing frame %d, skipped', frame_id)

    def _decode_one(self, packet):
        frames = packet.decode()
        if len(frames) == 0:
            raise RuntimeError('Empty packet')
        assert len(frames) <= 1, 'More than one frame in a packet.' + \
            self._assert_msg
        frame = frames[0]
        assert isinstance(frame, av.VideoFrame)
        frame = Frame(frame)
        return frame

    def _decode(self, frame_id):
        packet = self._packets[frame_id]
        try:
            decode_frame_id = -1
            while decode_frame_id < frame_id:
                frame = self._decode_one(packet)
                if frame.frame_id == decode_frame_id:
                    break
                decode_frame_id = frame.frame_id
            assert frame.frame_id == frame_id, \
                'Frame id should be %d, but is %d.' % (
                    frame_id, frame.frame_id) + self._assert_msg
            return frame
        except (av.AVError, RuntimeError):
            self._logger.info('Decode failed for frame %d', frame_id)
            return None
