import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.metrics import Accuracy, AUC, TruePositives, TrueNegatives, FalsePositives, FalseNegatives, Precision
import numpy as np
import pandas as pd
import os
import time as t
from Models import get_xception, get_inceptionv3, get_Xception_dropout
import Globals as glob
from sklearn.model_selection import StratifiedKFold
from hyperopt import hp, tpe, Trials, fmin
import sys


def objective(params):
    """ Evaluating the Given Hyper-parameters with 3-Fold Cross Validation.
        This Function returns the mean loss value of the 3-validations. """
    global input_size, history, ds, x_train, y_train, model_fun, device, verbose
    print(f'Objective Params: {params}')
    param_eval = []

    skf = StratifiedKFold(n_splits=3)
    for train_index, test_index in skf.split(x_train, np.argmax(y_train, axis=1)):
        print(f'Objective Fold: {len(param_eval)+1}')
        # Create Model
        model = model_fun(input_size, classes_num=y_train.shape[1],
                          optimizer=params['optimizer'](learning_rate=params['learning_rate']))
        # Train Model
        model.fit(x_train[train_index], y_train[train_index], batch_size=int(params['batch_size']), epochs=int(params['epochs']),
                  verbose=verbose)
        # Evaluate Model
        f_eval = model.evaluate(x_train[test_index], y_train[test_index], verbose=verbose)
        print(f'Objective Fold-{len(param_eval)+1} Eval: {f_eval}')
        param_eval.append(f_eval[0])
    del model
    print(f'Objective Eval: {np.mean(param_eval)}')
    return np.mean(param_eval)


results_dir = '/Results/'
input_size = (224, 224, 3)
# Search Space for Hyper-Parameters Optimization
optimizers_choice = [Adam]
space = {
    "epochs": hp.quniform("epochs", 9, 21, 3),
    "batch_size": hp.quniform("batch_size", 32, 130, 32),
    "optimizer": hp.choice("optimizer", optimizers_choice),
    "learning_rate": hp.uniform("learning_rate", 1e-4, 0.002),
}
# Running Parameters:
model_fun, model_name = get_Xception_dropout, 'Xception_Dropout' # TODO: change
number_of_folds = 5
max_evals =10
verbose = 0
history = {}

# Write console to file
sys.stdout = open(f'Results/{model_name}-out.txt', 'w')
print(f'Model: {model_name}')
metrics = [Accuracy(), AUC(curve='ROC'), AUC(curve='PR'), TruePositives(), TrueNegatives(), FalsePositives(),
           FalseNegatives(), Precision()]


for ds in glob.datasets_names:
    history[ds] = {}

    for fold in range(number_of_folds):

        print(f'Starting... Dataset: {ds} Fold: {fold+1}')
        # Loads dataset
        x_train = np.load(f'Data/Datasets/Processed/{ds}/train/x{fold}.npy')
        x_test = np.load(f'Data/Datasets/Processed/{ds}/test/x{fold}.npy')
        y_train = np.load(f'Data/Datasets/Processed/{ds}/train/y{fold}.npy')
        y_test = np.load(f'Data/Datasets/Processed/{ds}/test/y{fold}.npy')

        trials = Trials()
        st = t.time_ns()
        # using TPE for Hyper-parameters evaluation by minimizing the 'objective' function
        best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
        hp_t = glob.get_time_passed(st)
        print(f'Dataset: {ds} Fold: {fold+1} Finished Params Evaluation - Time: {hp_t} min')

        model_params = {
            "input_size": input_size,
            "classes_num": y_train.shape[1],
            "metrics": metrics,
            "optimizer": optimizers_choice[best_params['optimizer']](learning_rate=best_params['learning_rate'])
        }
        train_params = {
            "x": x_train,
            "y": y_train,
            "epochs": int(best_params['epochs']),
            "batch_size": int(best_params['batch_size']),
            "verbose": verbose
        }
        # Train model on the Best Hyper-Parameters Found
        model = model_fun(**model_params)
        st = t.time_ns()
        model.fit(**train_params)
        t_t = glob.get_time_passed(st)
        print(f'Dataset: {ds} Fold: {fold+1} Finished Model Training - Time: {t_t} min')
        st = t.time_ns()
        # Evaluate model on Test set
        fold_eval = model.evaluate(x_test, y_test, batch_size=train_params['batch_size'], verbose=verbose, return_dict=True)
        e_t = glob.get_time_passed(st, u="sec")
        print(f'Dataset: {ds} Fold: {fold+1} Finished Model Evaluation: {fold_eval} - Time: {t_t} sec')
        # Saving results and relevant information
        history[ds][fold] = {
            'Dataset Name': [ds],
            'Algorithm Name': [model_name],
            'Cross Validation': [fold],
            'Hyper-Parameters values': [str([int(best_params['epochs']),
                                             int(best_params['batch_size']),
                                             model_params["optimizer"]._name,
                                             round(best_params['learning_rate'], 5)])],
            'Accuracy': [fold_eval['accuracy']],
            'TPR': [round(fold_eval['true_positives'] / (fold_eval['true_positives'] + fold_eval['false_negatives']), 3)],
            'FPR': [round(fold_eval['false_positives'] / (fold_eval['false_positives'] + fold_eval['true_negatives']), 3)],
            'Precision': [fold_eval['precision']],
            'AUC ROC': [fold_eval['auc']],
            'AUC PR': [fold_eval['auc_1']],
            'Hyper-Parameters Evaluation Time (min)': [hp_t],
            'Training Time (min)': [t_t],
            'Inference Time for 1000 Instances (sec)': [(e_t / y_test.shape[0]) * 1000]
        }
        # Saving result per dataset
        cols = list(history[ds][0].keys())
        df = pd.DataFrame(columns=cols)
        for ds in history:
            for fold in history[ds]:
                row = pd.DataFrame(history[ds][fold], columns=cols)
                df = df.append(row)

        df.to_csv(f'Results/{model_name}-{ds}.csv')

    print(f'Finished Dataset: {ds}')


# scancel 140052, 140197,
# 140198 140200
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)
# tf.config.experimental.get_memory_usage('GPU:0')