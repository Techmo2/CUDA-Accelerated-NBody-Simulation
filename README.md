# CUDA-Accelerated-NBody-Simulation
An nbody simulation of massive objects that uses cuda as a means of drastically increasing the simulation speed.

## Building
In order to build the project, you must install g++ version 6, and the Nvidia CUDA Toolkit.
If you're on Ubuntu, you can install these by using the commands:

```bash
sudo apt-get update && sudo apt-get install nvidia-cuda-toolkit g++-6
```
Once you have all the dependencies installed, ```cd``` to the project directory, and build the project by using:
 ```bash
 ./build.sh
 ``` 
If everything is installed correctly, the build should finish, and the application binary should be located in the ```build``` directory within the project folder.

## Usage
Currently, the program does not display any graphics. It will print some statistics about the program's execution into the terminal. Run the simulation with:
 ```bash 
 ./simulation <Number of bodies> <Number of CUDA threads> <Number of simulation cycles>
 ```

If no parameters are provided, the simulation starts with **1000** bodies, **256** cuda threads, and runs for **1000** simulation cycles.

An example:
```bash
./simulation 2000 512 1000 # 2000 bodies with 512 cuda threads. Will run for 1000 simulation cycles.
```
Running the simulation with too many cuda threads will result in some threads crashing, and may give erroneous results.