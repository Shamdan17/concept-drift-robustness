# concept-drift-robustness

## Windows PE datasets:

First create a folder named data if it does not exist: 

```
mkdir data
```

Then, download the data from the following links:

### BODMAS
```
https://drive.google.com/drive/folders/1Uf-LebLWyi9eCv97iBal7kL1NgiGEsv_
```

### EMBER
A version of EMBER preprocessed by BODMAS authors can be found here, only ember related files are needed:

```
https://drive.google.com/drive/folders/12DMPeh8DA2ukPATnHX4K__shWFJIiBN5
```

Place the .npz data and .csv metadata files in the data folder.

## Android Malware datasets:

### KRONODROID 

Download the zip files from the following link, and unzip them in the data folder: 

```
https://github.com/aleguma/kronodroid/tree/main/emulator
```

---
## Training

Training script can be used to train a specified model on a selected dataset, with optional arguments such as train_start_date, train_end_data, and more. For example, to train a decision tree model on the bodmas dataset with default arguments, use the following command:


```
python src/train.py --model_type DT --dataset bodmas 
```

# Replicating results:

First download the data from the above instructions, then run the following

```
python src/run_all_experiments.py
python src/kronodroid_experiments.py
```



