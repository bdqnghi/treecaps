# TreeCaps: Tree-Structured Capsule Networks for Program Source Code Processing.

<p aligh="center"> This repository contains the code for TreeCaps introduced in the following paper <b>TreeCaps: Tree-Structured Capsule Networks for Program Source Code Processing (NeurIPS Workshops 2019) </b> </p>

## Usage

1. Install the required dependencies ```pip install -r requirements.txt```.

2. Download and extract the dataset ```python3 download_data.py```

3. Simply run ```python3 main_vts.py```.


### Datasets

We used 2 datasets in 2 programming languages to ensure cross-language robustness:

* [**SA Dataset:**]: 10 classes of sorting algorithms, with 64 training programs on average per class, written in Java. 
* [**OJ Dataset:**]: 104 classes of C programs, with 375 training programs on average per class. 

