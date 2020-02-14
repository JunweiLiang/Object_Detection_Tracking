# Speed of diva_io.video.VideoReader

Test performed by [video/speed_test.sh](../video/speed_test.sh).

```sh
./video/speed_test.sh <video_dir>
```

## Overall Performance

Loading all frames of 7 videos from the [MEVA dataset](http://mevadata.org). Each video is 5-min long and 1080p at 30 fps.

|  | `diva_io.video. VideoReader (fix_missing=True)` | `diva_io.video. VideoReader (fix_missing=False)` | `pymovie.editor .VideoFileClip` | `cv2.VideoCapture` |
|:---------------:|:----------------------------------------------:|:-----------------------------------------------:|:-------------------------------:|:------------------:|
| User Time | 338.12s | 329.00s | 904.09s | 844.35s |
| System Time | 0.80s | 0.60s | 317.14s | 6.44s |
| CPU Utilization | 99% | 99% | 293% | 264% |
| Total Time | 338.98s | 329.60s | 416.31s | 321.06s |

## Detailed Results

| Video Name | Video Description | `diva_io.video .VideoReader (fix_missing=True)` | `diva_io.video .VideoReader (fix_missing=False)` | `pymovie.editor .VideoFileClip` | `cv2.VideoCapture` |
|:----------------------------------------------:|:-------------------------------------------------------:|:-----------------------------------------------:|:------------------------------------------------:|:-------------------------------:|:------------------:|
| 2018-03-11.16-30-08.16-35-08.hospital.G436.avi | No missing | 0:45 | 0:44 | 1:00 | 0:26 |
| 2018-03-07.16-55-06.17-00-06.school.G336.avi | Missing 104-109, 2294 | 0:55 | 0:53 | 0:59 | 0:26 |
| 2018-03-11.11-25-01.11-30-01.school.G424.avi | Missing 7391-7499 | 0:38 | 0:37 | 0:58 | 0:26 |
| 2018-03-11.16-25-00.16-30-00.school.G639.avi | Bidirectional frames, missing 1, 4 | 0:55 | 0:53 | 0:59 | 0:27 |
| 2018-03-11.11-35-00.11-40-00.school.G299.avi | Packet id and frame id unsychronized, missing 5789-5797 | 0:50 | 0:49 | 0:58 | 1:42 |
| 2018-03-11.11-35-00.11-40-00.school.G330.avi | Packet id and frame id unsychronized, missing 5755-5761 | 0:50 | 0:49 | 0:58 | 0:39 |
| 2018-03-12.10-05-00.10-10-00.hospital.G436.avi | First packet fail | 0:41 | 0:41 | 0:59 | 1:11 |
