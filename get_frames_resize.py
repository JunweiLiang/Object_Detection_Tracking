# coding=utf-8
# given a list of videos, get all the frames and resize
# note that the frames are 0-indexed

import sys,os,argparse
from tqdm import tqdm
import cPickle as pickle

def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("videolist")
  parser.add_argument("despath")

  parser.add_argument("--size",default=800,type=int)
  parser.add_argument("--maxsize",default=1333,type=int)

  parser.add_argument("--resize",default=False,action="store_true")

  parser.add_argument("--job",type=int,default=1,help="total job")
  parser.add_argument("--curJob",type=int,default=1,help="this script run job Num")
  parser.add_argument("--statspath",default=None,help="path to write videoname.p to save some stats for that video")
  parser.add_argument("--use_2level",action="store_true",help="make videoname/frames dir")
  parser.add_argument("--name_level",type=int,default=None,help="add the top level folder name to the videoname")
  parser.add_argument("--cv2path",default=None)

  parser.add_argument("--use_moviepy", action="store_true")
  parser.add_argument("--use_lijun", action="store_true")

  return parser.parse_args()

def get_new_hw(h,w,size,max_size):
  scale = size * 1.0 / min(h, w)
  if h < w:
    newh, neww = size, scale * w
  else:
    newh, neww = scale * h, size
  if max(newh, neww) > max_size:
    scale = max_size * 1.0 / max(newh, neww)
    newh = newh * scale
    neww = neww * scale
  neww = int(neww + 0.5)
  newh = int(newh + 0.5)
  return neww,newh


if __name__ == "__main__":
  args = get_args()
  if args.cv2path is not None:
    sys.path = [args.cv2path] + sys.path


  if args.use_moviepy:
    from moviepy.editor import VideoFileClip
  elif args.use_lijun:
    from diva_io.video import VideoReader

  # still need this to write image
  import cv2
  print "using opencv version:%s"%(cv2.__version__)

  if not os.path.exists(args.despath):
    os.makedirs(args.despath)

  if  args.statspath is not None and not os.path.exists(args.statspath):
    os.makedirs(args.statspath)

  count=0
  for line in tqdm(open(args.videolist,"r").readlines()):
    count+=1
    if((count % args.job) != (args.curJob-1)):
      continue

    video = line.strip()

    stats = {"h":None,"w":None,"fps":None,"frame_count":None,"actual_frame_count":None}

    videoname = os.path.splitext(os.path.basename(video))[0]

    targetpath = args.despath

    if args.use_2level:
      targetpath = os.path.join(args.despath,videoname)
      if not os.path.exists(targetpath):
        os.makedirs(targetpath)

    if args.name_level is not None:
      foldernames = video.split("/")
      prefixes = foldernames[-1-args.name_level:-1]
      videoname = "__".join(prefixes + [videoname])

    if args.use_moviepy:
      vcap = VideoFileClip(video)
      frame_count = int(vcap.fps * vcap.duration)  # uh
      vcap_iter = vcap.iter_frames()
    elif args.use_lijun:
      vcap = VideoReader(video)
      frame_count = int(vcap.length)
    else:
      try:
        vcap = cv2.VideoCapture(video)
        if not vcap.isOpened():
          raise Exception("cannot open %s"%video)
      except Exception as e:
        raise e

      if cv2.__version__.split(".") != "2":
        frame_width = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        fps = vcap.get(cv2.CAP_PROP_FPS)
        frame_count = vcap.get(cv2.CAP_PROP_FRAME_COUNT)
      else:
        frame_width = vcap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        frame_height = vcap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)

        fps = vcap.get(cv2.cv.CV_CAP_PROP_FPS)
        frame_count = vcap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
      stats['h'] = frame_height
      stats['w'] = frame_width

      stats['fps'] = fps



    stats['frame_count'] = frame_count

    cur_frame=0
    count_actual=0
    while cur_frame < frame_count:
      if args.use_moviepy:
        suc = True
        frame = next(vcap_iter)
      else:
        suc, frame = vcap.read()

      if not suc:
        cur_frame+=1
        tqdm.write("warning, %s frame of %s failed"%(cur_frame,videoname))
        continue
      count_actual+=1
      frame = frame.astype("float32")

      if args.resize:
        neww,newh = get_new_hw(frame.shape[0],frame.shape[1],args.size,args.maxsize)

        frame = cv2.resize(frame,(neww,newh),interpolation=cv2.INTER_LINEAR)

      cv2.imwrite(os.path.join(targetpath,"%s_F_%08d.jpg"%(videoname,cur_frame)),frame)

      cur_frame+=1

    stats['actual_frame_count'] = count_actual

    if args.statspath is not None:
      with open(os.path.join(args.statspath,"%s.p"%videoname),"wb") as fs:
        pickle.dump(stats,fs)
    if not args.use_moviepy and not args.use_lijun:
      vcap.release()

