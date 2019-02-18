# CUDA-Accelerated-NBody-Simulation
An nbody simulation of massive objects that uses cuda as a means of drastically increasing the simulation speed.

## Building
In order to build the project, you must install g++ version 6, and the Nvidia CUDA Toolkit.
If you're on Ubuntu, you can install these by using the commands:

```bash
sudo apt-get update && sudo apt-get install nvidia-cuda-toolkit g++-6
```
Once you have all the dependencies installed, build the project by using:
 ```bash
 ./build.sh
 ``` 
If everything is installed correctly, the build should finish, and the application binary should be located in the ```build``` directory within the project folder.

## Usage
Currently, the program does not display any graphics. It will print some statistics about the program's execution into the terminal. Run the simulation with:
 ```bash 
 ./simulation <Number of Bodies> <Threads Per Block> <Number of Cycles> <Save Frames?>
 ```

If no parameters are provided, the simulation starts with **1000** bodies, **256** threads per block, and runs for **1000** simulation cycles. Due to bandwith limitations, the simulation runs significantly slower when the frames are saved to the disk.

An example:
```bash
./simulation 2000 512 1000 # 2000 bodies with 512 threads per block. Will run for 1000 simulation cycles.
```

## Visualization
In order to see the results of the simulation, a quick and dirty python program (```visualize.py```) is provided. This was written with python3 and the requirements can be installed with:
```bash
pip3 install -r requirements.txt
```
This program requires that you have some form of the glfw library installed.