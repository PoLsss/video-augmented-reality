# Video augmented reality on real-time webcam
This project is an augmented reality, created to research and implement augmented reality applications in the field of comuter vision used OpenCV lib.

## Requirements
Ensure you have the following installed:
- Python 3.6+
- OpenCV
- NumPy

## Installation
1. Clone this repository:
      
       git clone https://github.com/PoLsss/video-augmented-reality.git
   
       cd video-augmented-reality
2. Install the necessary libraries using the command below:

        pip install numpy opencv-python argparse
## Usage 
  Run the script with the following command:
  
        python OneTarget.py --image-path <your image path> --video-path <your video path> --threshold <your threshold>
  Arguments:
  - `--image-path`: Path to the target image.
  - `--video-path`: Path to the target video.
  - `--threshold`: Minimum number of feature matches required to overlay the video (Default: 20).
## Demo
One target augmented reality (OneTarget.py):



https://github.com/PoLsss/video-augmented-reality/assets/98370447/54e12f2c-abce-4dc4-b977-9d515eb5ab04


Multi target (MultiTarget.py):

 Run the script with the following command:
  
        python MultiTarget.py --image-path1 <your image path 1> --video-path1 <your video path 1> 
                              --image-path2 <your image path 2> --video-path1 <your video path 2> 
                              --image-path3 <your image path 3> --video-path1 <your video path 3> 
                              --image-path4 <your image path 4> --video-path1 <your video path 4> 
                              --threshold <your threshold>


https://github.com/PoLsss/video-augmented-reality/assets/98370447/2cdca8a3-4f91-4c75-b0d9-275ffa0194e3

And there are some more interesting demos.
---
This is a subject project, the result of the work of team members.
