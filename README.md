# DLCV Final Project ( Medical-Imaging )

# How to run your code?
### To infer testing data
To reproduce the results, it is easy to run the provided script:
```script
./omnipotent.sh <Path to Blood_test_data> <Path to the output csv file>
```
For example,
```script
./omnipotent.sh Blood_data/test/ submit_test.csv
```
### To train
#### STEP 1: train the CNN backbone
```
cd wubinray/base/
make train_resnet18
```
#### STEP 2: copy the CNN backbone model and fixed train/val set to the lstm folder
```
cp wubinary/base/checkpoints/resnet18/best.pth kaminyou/lstm/backbone_model/best.pth
cp wubinary/base/checkpoints/resnet18/train_set.pkl kaminyou/lstm/train_set.pkl
cp wubinary/base/checkpoints/resnet18/valid_set.pkl kaminyou/lstm/valid_set.pkl
```
#### STEP 3: revise the backbone model path and serialized train/val set path in config.yaml
```yaml
PREPROCESSING:
  BACKBONE_MODEL_PATH: ./backbone_model/best.pth
LSTM_TRAINING: 
  TRAIN_SET_PATH: ./train_set.pkl
  VAL_SET_PATH: ./valid_set.pkl
```
#### STEP 4: train the LSTM model
```
python3 preprocessing.py
python3 train.py
```
#### STEP 5: obtain the lstm model
The lstm model will be saved in the `kaminyou/lstm/experiment/datetime/lstm_best.pth`.
#### STEP 6: infer for the testing data
Simply move those two models in the root folder and following the steps in previous `To inference testing data` part. 

# Usage
To start working on this final project, you should clone this repository into your local machine by using the following command:

    git clone https://github.com/DLCV-Fall-2020/medical-imaging-<team_name>.git
Note that you should replace `<team_name>` with your own team name.

For more details, please click [this link](https://docs.google.com/presentation/d/1T8Wh9rM5zCiuMVCulDCZwX9JZZ9Mqgd0Yr3uqgPpe1I/edit?usp=sharing) to view the slides of Final Project - Medical Imaging. **Note that video and introduction pdf files for final project can be accessed in your NTU COOL.**

### Dataset
In the starter code of this repository, we have provided a shell script for downloading and extracting the dataset for this assignment. For Linux users, simply use the following command.

    bash ./get_dataset.sh
The shell script will automatically download the dataset and store the data in a folder called `Blood_data`. Note that this command by default only works on Linux. If you are using other operating systems, you should download the dataset from [this link](https://drive.google.com/file/d/1XY5twJuNLU-tJL-sr5-efTMPqbOtodS6/view?usp=sharing) and unzip the compressed file manually.

> ⚠️ ***IMPORTANT NOTE*** ⚠️  
> 1. Please do not disclose the dataset! Also, do not upload your get_dataset.sh to your (public) Github.
> 2. You should keep a copy of the dataset only in your local machine. **DO NOT** upload the dataset to this remote repository. If you extract the dataset manually, be sure to put them in a folder called `Blood_data` under the root directory of your local repository so that it will be included in the default `.gitignore` file.

### Evaluation
We will use F2-score to evaluate your model. Please refer to the introduction ppt for more details.

# Submission Rules
### Deadline
2021/1/22 11:59 GMT+8

### Late Submission Policy
#### Late Submission is NOT allowed for final project!

### Academic Honesty
-   Taking any unfair advantages over other class members (or letting anyone do so) is strictly prohibited. Violating university policy would result in an **F** grade for this course (**NOT** negotiable).    
-   If you refer to some parts of the public code, you are required to specify the references in your report (e.g. URL to GitHub repositories).      
-   You are encouraged to discuss homework assignments with your fellow class members, but you must complete the assignment by yourself. TAs will compare the similarity of everyone’s submission. Any form of cheating or plagiarism will not be tolerated and will also result in an **F** grade for students with such misconduct.

> 🆕 ***NOTE***  
> For the sake of conformity, please use the `python3` command to call your `.py` files in all your shell scripts. Do not use `python` or other aliases, otherwise your commands may fail in our autograding scripts.

### Packages
This homework should be done using python3.6. For a list of packages you are allowed to import in this assignment, please refer to `requirments.txt` for more details.

You can run the following command to install all the packages listed in `requirements.txt`:

    pip3 install -r requirements.txt

Note that using packages with different versions will very likely lead to compatibility issues, so make sure that you install the correct version if one is specified above. E-mail or ask the TAs first if you want to import other packages.


# Q&A
If you have any problems related to Final Project, you may
- Use TA hours
- Contact TAs by e-mail ([ntudlcv@gmail.com](mailto:ntudlcv@gmail.com))
- Post your question under Final Project FAQ section in FB group
