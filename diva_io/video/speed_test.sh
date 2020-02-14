#! /bin/zsh

video_dir=$1
cd $(pwd)/$(dirname $0)/../..

echo "diva_io.video.VideoReader(fix_missing=True)"
time python -c "from diva_io.video.test import speed_test_divaio; speed_test_divaio(\"$video_dir\", True)"

echo "diva_io.video.VideoReader(fix_missing=False)"
time python -c "from diva_io.video.test import speed_test_divaio; speed_test_divaio(\"$video_dir\", False)"

echo "moviepy.editor.VideoFileClip"
time python -c "from diva_io.video.test import speed_test_moviepy; speed_test_moviepy(\"$video_dir\")"

echo "cv2.VideoCapture"
time python -c "from diva_io.video.test import speed_test_opencv; speed_test_opencv(\"$video_dir\")"
