# Cechmate

This library provides easy to use  onstructors for custom filtrations. 
It is currently designed to produce filtrations suitable for use with [Phat](https://github.com/xoltar/phat). 
Phat currently provides a clean interface for persistence reduction algorithms for boundary matrices. 
This tool helps bridge the gap between data and boundary matrices.  
Currently, we support construction of Alpha and Rips filtrations, with more on the way.  

If you have a particular filtration you would like implemented, please feel free to reach out and we can work on helping with implementation and integration, so others can use it.

## Dependencies
* Numpy
* Matplotlib
* Phat

## Setup
You will need to install the Python wrapper for [Phat](https://github.com/xoltar/phat) using pip

```
pip install phat
```

Cechmate is currently not hosted on pypi, until it is, please install the library from a clone with the following commands:
```
git clone https://github.com/ctralie/cechmate
cd cechmate
pip install -e .
```

## Test examples

Please refer to the BasicUsage notebook in the examples/ directory
