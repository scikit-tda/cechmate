# phatcech (Pronounced "Fat Check")
Doing Cech Filtrations Wrapping around Phat (The Persistent Homology Algorithms Toolbox)

##Compiling PHAT
By default, the code calls a binary for phat from the command line, which you will need to compile.  This requires you to have CMake and OpenMP installed.  First, download and extract phat from this link: https://bitbucket.org/phat-code/phat.  Then, change into the phat directory and type

~~~~~ bash
mkdir build
cd build
cmake ..
make
~~~~~

This will generate a binary called "phat" in the build/ directory, which you will need to copy to the root of the PhatCech repository.  To test this, type
~~~~~ bash
python PhatCech.py
~~~~~

You should see the following image pop up:
![Example Rips Filtration](RipsExample.png "Rips on Noisy Circle")

