Instruction for running the code:

First check that the datasets are located at 'Data/Datasets/Raw'.
* In case you would like to check another dataset please make sure that it does not contain any categorical values.

Please also check that the Base Image is also located is the right place - 'Data/Base Image/Rick&Morty.png'
* you can use any image you would like, even some random numbers in the right dimensions, witch is (224, 224, 3) - RGB with size 224x224.

The next stage is to transform the datasets from tabular to images, to do that you will need to run - PreProcess.py file.
1. If you would like to choose on witch data set to run the preprocess - go to Globals.py and change the list 'datasets_names',
    the list should contain the names of the datasets according to the file names in the 'Data/Datasets/Raw' folder.
2. To change the number of Fold you would like to create go to - PreProcess.py to 'preprocess' function and change 'n_splits' to the wanted number (line 76)

The method we have used to implement that can be found in the next article:
https://www.biorxiv.org/content/10.1101/2020.05.02.074203v1.full

After running PreProcess.py the desired datasets will be created in 'Data/Datasets/Processed', you will found a directory for each one of the datasets.
each directory contains 2 subdirectories one for train and one for test.
Each one of the train and test directories will contain files per each fold, one for the features 'x{fold_index}.npy',
and one for the labels 'y{fold_index}.npy'. The files contains a numpy array with all the relevant samples.

Now you can start the training and evaluations run:
Changing the parameters:
1. The the dataset the will be evaluated are the ones in the 'datasets_names' in Globals.py.
2. 'space' - is a dictionary that contains the hyper-parameters to evaluate and their values range (line 44 in Training.py):
    a. "epochs" - is the number of epochs values and range.
    b. "batch_size" - is the size of a training batch values and range.
    c. "optimizer" - is the options for different optimizers.
    d. "learning_rate" - is the optional values and range for the learning rate.
3. model_fun - is the variable that determines witch model will be running (line 51 in Training.py):
    a. get_xception - will use the Xception architecture.
    b. get_xception_dropout - will use the Xception architecture with the dropout layers we added (our improvement)
    c. get_inceptionV3 - will use the InceptionV3 architecture.
    * In the same line change the variable 'model_name' to match the model - it is used for creating the Results file.
4. number_of_folds - the number of fold you would like to run, this variable value should not exceed the number of fold that were created in the preprocess stage (line 52 in Training.py).
5. max_evals - the maximal number of evaluations for the hyper-parameters optimization (line 53 in Training.py).

You can find the results of the run in 'Results/Results/{model_name}-{ds}.csv', where ds - is the dataset name.
Be aware that the results is saved after each fold and the results file is updated after each fold.
Furthermore, after each dataset a new results file is created and it always contains the results from all the previous datasets.
We decided to implement it like that to keep track on the process and to save each evaluation in case there is some kind of exception, or an error that will terminate the running.