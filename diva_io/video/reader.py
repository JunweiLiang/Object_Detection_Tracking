import av
import heapq
import logging
import numpy as np
import os.path as osp

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
            self._logger = get_logger(
                '%s@%s' % (__name__, self.path), logging.INFO)
            av.logging.set_level(av.logging.INFO)
        self._assert_msg = ' Please report %s to Lijun.' % (self.path)
        self.fix_missing = fix_missing
        if not self.fix_missing:
            self._logger.warning('NOT fixing missing frames.')
        self._init()
        self.length = self._stream.duration
        self.fps = float(self._stream.average_rate)
        self.height = self._stream.codec_context.format.height
        self.width = self._stream.codec_context.format.width
        self.shape = (self.height, self.width)

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

    def get_iter(self, limit=None, stride=1):
        """Get an iterator to yield a frame every stride frames and stop at a
        limited number of yielded frames.

        Parameters
        ----------
        limit : int, optional
            Total number of frames to yield, by default None. If None, it
            yields until the video ends.

        stride : int, optional
            The stride length for each read, by default 1. If stride = 1, no
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
                yield self.get_skip(stride)
            except StopIteration:
                break

    def get_skip(self, stride=1):
        """Read a frame from the video every stride frames. It returns the
        immediate next frame and skips stride - 1 frames for the next call of
        get.

        Parameters
        ----------
        stride : int, optional
            The stride length for each read, by default 1. If stride = 1, no
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
            for _ in range(stride - 1):
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

    def reset(self):
        """Reset the internal states to load the video from the beginning.
        """
        self._container.close()
        self._init()

    def __del__(self):
        if hasattr(self, '_container'):
            self._container.close()

    def _init(self, video_stream_id=0):
        self._container = av.open(self.path)
        self._stream = self._container.streams.video[video_stream_id]
        self._frame_gen = self._get_frame_gen()

    def _get_frame_gen(self, start_frame_id=0, retry=5, retry_step=120):
        if start_frame_id == 0:
            for f in self._fix_missing(start_frame_id):
                yield f
        else:
            seek_frame_id = start_frame_id
            for _ in range(retry):
                if seek_frame_id != start_frame_id:
                    self._logger.warn(
                        'Failed to seek to frame %d, retrying with frame %d',
                        start_frame_id, seek_frame_id)
                self._container.seek(seek_frame_id, stream=self._stream)
                success = False
                for frame in self._fix_missing(start_frame_id):
                    success = True
                    if frame.frame_id >= start_frame_id:
                        yield frame
                if success:
                    return
                seek_frame_id -= retry_step
            raise RuntimeError('Failed to seek to frame %d' % (start_frame_id))

    def _fix_missing(self, start_frame_id):
        frame_gen = self._reorder()
        try:
            frame = next(frame_gen)
        except StopIteration:
            return
        if frame.frame_id > start_frame_id:
            for f in self._fix_missing_one(
                start_frame_id, frame.frame_id - 1, frame):
                yield f
        yield frame
        while True:
            prev_frame = frame
            try:
                frame = next(frame_gen)
                next_frame_id = frame.frame_id
            except StopIteration:
                frame = None
                next_frame_id = self.length
            frame_gap = next_frame_id - prev_frame.frame_id
            assert frame_gap >= 1, \
                'Unreordered bidirectional frame occured.' + self._assert_msg
            if frame_gap > 1:
                for f in self._fix_missing_one(
                    prev_frame.frame_id + 1, next_frame_id - 1, prev_frame):
                    yield f
            if frame is not None:
                yield frame
            else:
                return

    def _fix_missing_one(self, start_frame_id, end_frame_id, src_frame):
        if self.fix_missing:
            self._logger.info(
                'Missing frames from %d to %d, used frame %d',
                start_frame_id, end_frame_id, src_frame.frame_id)
            for frame_id in range(start_frame_id, end_frame_id + 1):
                offset = frame_id - src_frame.frame_id
                missing_frame = Frame(src_frame.frame, offset)
                yield missing_frame
        else:
            self._logger.warn(
                'Missing frames from %d to %d, skipped',
                start_frame_id, end_frame_id)

    def _reorder(self, buffer_size=5):
        buffer = []
        for frame in self._decode():
            heapq.heappush(buffer, (frame.frame_id, frame))
            if len(buffer) > buffer_size:
                _, frame = heapq.heappop(buffer)
                yield frame
        while len(buffer) > 0:
            _, frame = heapq.heappop(buffer)
            yield frame

    def _decode(self):
        for packet in self._container.demux():
            try:
                for frame in packet.decode():
                    yield Frame(frame)
            except av.AVError:
                self._logger.info('Decode failed for packet %s', packet)
