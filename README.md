### Usage
`op_testing.py` should now accept original unprocessed video
 and produce the input data in their respective
 repositories required to run the program (`main.py`).
 
 You can now just run main.py, which will run
 the cleaned up support files.

```self.get_cadence()```
### Data:
Example data processed with Open Pose can be found here:

https://1drv.ms/u/s!Am7KReNR0aJnmVMnFB9eom9GWGjt?e=shrSjG

You can copy and move these folders into the same directory as the `extract_data.py` file.


### Installation
The required libraries can be installed by downloading the requirements.txt with pip:

```py -3.7-64 -m pip install requirements.txt```

or

```pip3 install requirements.txt```

Additionally the input directories should be created in the 
correct loccations and named correctly.
The program should then be ready to run after the installations.

Additionally, Open Pose must be installed to use `op_testing.py`

Documentation on how to install Open Pose can be found at https://github.com/CMU-Perceptual-Computing-Lab/openpose

However the documentation is... Not very good. Alternatively the example data can be used with `main.py`
### How to use the program

After running the command to start the program, the user interface should appear.
The interface shows two tabs: One for sagittal and one for coronal. You can select
as many measurements are required, and once ready press start to commence
processing.
 
 The progress bar will tell the user when the processing is complete, 
and the processed images will appear in `output_images` or `coronal_output_images`.
However, you must wait until a info window pops up to tell you that the
full operation is complete, as the program will continue to save the images to a video 
that can be found in `processed_video`. It is not recommended to start another process
until this is complete.

 Running `main.py` will assume you are using the default folders for data and images. Alternatively 
 you can specify the location of the data folders with parse arguments:
 
 ```-i ``` specificies saggital plane images
 <br>
 `-d` specifies saggital plane JSON data
 <br>
  ```-ci ``` specificies coronal plane images
 <br>
 `-cd` specifies coronal plane JSON data
 <br>
 Additionally:
 <br>
 `-fps` specifies output video fps (1 fps default)
 <br>
 `-a True` or `--anon True` adds a blur over patient's face for anonymity (Will increase processing time) 
 
 An example:
 <br>
 ```py -3.7-64 main.py -i op_images -d output1 -ci c_images -cd c_output --fps 20```
 
This command is telling the program to use the directory
op_images/ for the input images, output1/ for JSON data, and the other two arguments 
for the coronal plane data. This will also tell the program to set the output video to 20 fps (default 1 fps)

