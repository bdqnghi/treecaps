# TreeCaps: Tree-based Capsule Networks for Source Code Processing [Not Actively Maintained]

<p aligh="center"> This repository contains the code for TreeCaps introduced in the following paper <b>TreeCaps: Tree-based Capsule Networks for Source Code Processing. (NeurIPS 2020) </b> </p>

## Usage

1. Install the required dependencies ```pip install -r requirements.txt```.

2. Download and extract the dataset ```python3 download_data.py```. After this step, you can see two folders OJ_data and SA_data, noted that this data has been parsed into Srcml-based AST as protobuf format.

3. To train the model:
- Run ```python3 main_vts.py``` for VTS routing
- Run ```python3 main_drsw.py``` for DRSW routing



## Datasets

We used 2 datasets in 2 programming languages to ensure cross-language robustness:

* [**SA Dataset:**]: 10 classes of sorting algorithms, with 64 training programs on average per class, written in Java. 
* [**OJ Dataset:**]: 104 classes of C programs, with 375 training programs on average per class. 

