# TreeCaps: Tree-Structured Capsule Networks for Program Source Code Processing.

<p aligh="center"> This repository contains the code for TreeCaps introduced in the following paper <b>TreeCaps: Tree-Structured Capsule Networks for Program Source Code Processing (NeurIPS Workshops 2019) </b> </p>

## Usage

1. Install [requirements.txt](https://github.com/vinojjayasundara/treecaps/blob/master/requirements.txt) and the required dependencies ```pip install -r requirements.txt```.

2. Clone this repo: ```git clone https://github.com/vinojjayasundara/treecaps.git```.

3. Download and extract the [dataset](https://drive.google.com/open?id=1qdLNPjlNfGSLm9SdbQE6Me8K7Cp4W_ee) and the pre-trained [embedding](https://drive.google.com/open?id=10QTTj6Abhnpay7UPmDdS8uHPfAlxAq8m).

4. Simply run ```python job.py```.

5. Note the following in the ```job.py``` :

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; * Set ```training = 1``` for training the model and ```training = 0``` for testing. </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; * Uncomment the lines ```18-20``` in ```job.py``` to continue training with a reduced learning rate.

## Performance

### Datasets

We used three datasets in three programming languages to ensure cross-language robustness:

* [**SA Dataset:**]: 10 classes of sorting algorithms, with 64 training programs on average per class, written in Java. 
* [**OJ Dataset:**]: 104 classes of C programs, with 375 training programs on average per class. 

