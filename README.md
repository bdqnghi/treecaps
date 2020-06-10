# TreeCaps: Tree-Structured Capsule Networks for Program Source Code Processing.

<p aligh="center"> This repository contains the code for TreeCaps introduced in the following paper <b>TreeCaps: Tree-Structured Capsule Networks for Program Source Code Processing (NeurIPS Workshops 2019) </b> </p>

## Usage

1. Install [requirements.txt](https://github.com/vinojjayasundara/treecaps/blob/master/requirements.txt) and the required dependencies ```pip install -r requirements.txt```.

2. Download and extract the [dataset](https://drive.google.com/open?id=1qdLNPjlNfGSLm9SdbQE6Me8K7Cp4W_ee) and the pre-trained [embedding](https://drive.google.com/open?id=10QTTj6Abhnpay7UPmDdS8uHPfAlxAq8m).

3. Simply run ```python main_vts.py```.


### Datasets

We used t datasets in three programming languages to ensure cross-language robustness:

* [**SA Dataset:**]: 10 classes of sorting algorithms, with 64 training programs on average per class, written in Java. 
* [**OJ Dataset:**]: 104 classes of C programs, with 375 training programs on average per class. 

