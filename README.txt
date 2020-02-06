This is a C++ implementation of the paper: 

Unmixing-Based Soft Color Segmentation for Image Manipulation

Yagiz Aksoy, Tunc Ozan Aydin, Aljosa Smolic and Marc Pollefeys, ACM Trans. Graph., 2017

Dependencies


Ubuntu 14.04 or 16.04 
OpenCV 3.x
Install the following dependencies from terminal:


sudo apt-get install libpthread-stubs0-dev libboost-all-dev


Installation:

1. Clone the code.

2. Build the project using the following commands:

	mkdir build
	cd build
	cmake ..
	make

3. Run the code as follow:

./SoftSegmentation <path/to/image/> <tau_parameterer(optional)> <results directory>

For example:
./SoftSegmentation ../4_small.png 15 results

Results are saved in the folder <results directory>.

A tau parameter of 11 gives good results. 
