# DIVA IO Package

Version 0.2

Author: Lijun Yu

Email: lijun@lj-y.com

## Version History

* 0.2
  * **Real** random access in video loader.
  * Add annotation converter.
  * Warning control option.
* 0.1
  * Initial release of video loader.

## Installation

### Integration

To use as a submodule in your git project, run

```sh
git submodule add https://github.com/Lijun-Yu/diva-io.git diva_io
```

### Requirements

Environment requirements are listed in `environment.yml`.
For the `av` package, I recommend you install it via `conda` by

```sh
conda install av -c conda-forge
```

as building from `pip` would require a lot of dependencies.

## Video Loader

A robust video loader that deals with missing frames in the [MEVA dataset](http://mevadata.org).

This video loader is developed based on [`PyAV`](https://github.com/mikeboers/PyAV) package.
The [`pims`](https://github.com/soft-matter/pims) package was also a good reference despite its compatibility issue with current `PyAV`.

For the videos in the MEVA, using `cv2.VideoCapture` would result in wrong frame ids as it never counts the missing frames.
If you are using MEVA, I suggest you change to this video loader ASAP.

### Replace `cv2.VideoCapture`

According to my test, this video loader returns the exact same frame as `cv2.VideoCapture` unless missing frame or decoding error occured.
To replace the `cv2.VideoCapture` objects in legacy codes, simply change from

```python
import cv2
cap = cv2.VideoCapture(video_path)
```

to

```python
from diva_io.video import VideoReader
cap = VideoReader(video_path)
```

`VideoReader.read` follows the schema of `cv2.VideoCapture.read` but automatically inserts the missing frames while reading the video.

### Iterator Interface

```python
video = VideoReader(video_path)
for frame in video:
    # frame is a diva_io.video.frame.Frame object
    image = frame.numpy()
    # image is an uint8 array in a shape of (height, width, channel[BGR])
    # ... Do something with the image
```

### Random Access

Random access of a frame requires decoding from the nearest key frame (approximately every 60 frames for MEVA).
Averagely, this introduces a constant overhead of 0.1 seconds, which is much faster than iterating from the beginning.

```python
start_frame_id = 1500
length = 100
video.seek(start_frame_id)
for frame in video.get_iter(length):
    image = frame.numpy()
    # ... Do something with the image
```

### Video Properties

```python
video.width # cap.get(cv2.CAP_PROP_FRAME_WIDTH)
video.height # cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
video.fps # cap.get(cv2.CAP_PROP_FPS)
```

### Other Interfaces

For other usages, please see the comments in `video/reader.py`.

## Annotation

An annotation loader and converter for Kitware YML format in [meva-data-repo](https://gitlab.kitware.com/meva/meva-data-repo).

Clone the meva-data-repo and set

```python
annotation_dir = 'path/to/meva-data-repo/annotation/DIVA-phase-2/MEVA/meva-annotations'
```

### Convert Annotation

This is to convert the annotation from Kitware YML format to ActEV Scorer JSON format.
Run the following command in shell outside the repo's director,

```sh
python -m diva_io.annotation.converter <annotation_dir> <output_dir>
```

### Read Annotation

```python
from diva_io.annotation import KitwareAnnotation
video_name = '2018-03-11.11-15-04.11-20-04.school.G300'
annotation = KitwareAnnotation(video_name, annotation_dir)
# deal with annotation.raw_data
```
