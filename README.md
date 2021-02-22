# Perception Code Overview

Code Quality [![CodeFactor](https://www.codefactor.io/repository/github/berkeleyauv/perception/badge)](https://www.codefactor.io/repository/github/berkeleyauv/perception)

## Installation

We will use Conda for managing environments. We recommend installing Miniconda for Python 3.8 [here](https://docs.conda.io/en/latest/miniconda.html).
Then create an environment with

    conda create -n urobotics python=3.7

activate it with

    conda activate urobotics

and install all dependencies with

    pip3 install -r requirements.txt

Then clone the repo in a directory of your choice

    git clone https://github.com/berkeleyauv/perception.git

and install it

    pip3 install -e perception/

Also, our training data is stored here https://www.dropbox.com/sh/rrbfqfutrmifrxs/AAAfXxlcCtWZmUELp4wXyTIxa?dl=0 so download it and unzip it in the same folder as `perception`.

### Cython
To compile cythonized code, run the following commands after `cd`ing into the folder with Cython `setup.py`

    python setup.py build_ext --inplace
    cythonize file_to_cythonize.pyx


## misc:
Misc code, camera calibration etc.

## tasks:
Code for specific tasks like 

1. cross: cross detection
2. segmentation:
3. path_marker: path_marker detection
4. spinny_wheel_detection

etc

In order to create your own algorithm to test:

1. Create <your_algo>.py and put it in one of the specific task folders in perception/tasks.

2. Create a class which extends the TaskPerceiver class. perception/tasks/TaskPerceiver.py includes a template with documentation for how to do this.

## vis:
Visualization tools 
Code for testing tasks (Ideally this should be placed a separate folder called `tests`).

After writing the code for your specific task algorithm, you can do one of two things:

1. Add this to the end of <your algorithm>.py file:
    
        if __name__ == '__main__':
            from perception.vis.vis import run
            run(<list of file/directory names>, <new instance of your class>, <save your video?>)
    and then run
    
        python <your algorithm>.py
2. Add this to the perception/__init__.py file:

        import <path to your module>
        
        ALGOS = {
            'custom_name': <your module>.<your class reference>
        }
    and then run
    
        python vis.py --algorithm custom_name [--data <path to file/directory>] [--profile <function name>] [--save_video]
    The **algorithm** parameter is required. If **data** isn't specified, it'll default to your webcam. If **profile** isn't specified, it will be off by default. Add the **save_video** tag if you want to save your vis test as an mp4 file.

## wiki:
Flowchart on TaskPerceiver, TaskReceiver, AlgorithmRunner.
