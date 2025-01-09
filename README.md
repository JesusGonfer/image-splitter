# 🥷 IMAGE SPLITTER 🖼️ 🗡️
 
## Overview
The `image-splitter` project is designed to take a scanned image containing multiple smaller images and split it into individual images. This can be particularly useful for organizing and processing scanned documents, photos, or any other type of image that contains multiple distinct sections.

## Features
- Automatically detect and split multiple images within a single scanned image.
- High accuracy in image segmentation.
- Easy to use with a simple command-line interface.

## Installation
To install `image-splitter`, clone the repository and install the required dependencies:
```bash
git clone https://github.com/JesusGonfer/image-splitter.git
cd image-splitter
pip install -r requirements.txt
```

## Usage
To use `image-splitter`, run the following command:

```bash
python main.py <input directory/file> <out directory> <picture margin (can be +/-)> <threadhold bind detections>
```
Common usage:
```bash
python main.py ./images/ ./output/ -5 750
```

This will process the scanned image and save the individual images in the output directory.


## Debugging
After running the `image-splitter`, a folder will be generated in the output directory. This folder serves as a debug tool to visualize how the images have been split. It contains intermediate steps and segmented images, allowing you to verify the accuracy of the splitting process.

## Contributing
This is an unsupported project, so improvements are welcome, but new features or bug fixes cannot be requested.
