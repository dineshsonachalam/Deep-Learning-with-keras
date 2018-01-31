# Floydhub :

### 1. Create a .floydignore file to avoid data folder. Since it allows 100mb of code data size.

### 2. Create a dataset:

```
floyd data init av-digit-recognizer-cnn 
```

### 3. Upload the dataset:

```
floyd data upload
```

### 4. Go to Projects to create new projects:

```
floyd init av-digit-recognizer
```

### 5. Connecting data and code file

dataset --> niranjan2020/datasets/av-digit-recognizer-cnn/1

```
# Example 1:
floyd run --data floydhub/datasets/udacity-gan/1:/my_data "python my_script.py"

# Analytics vidhya cnn original example
floyd run --data av-digit-recognizer-cnn/1:data --mode jupyter --gpu

```

### 6. Then Kill the jobs --> go to jobs--> Cancel it--> For shut downing the project