import os
import pandas as pd


datasets_meta = {}
errors = []
all_samples = []
all_features = []
dir_path = f'Data/Datasets/Raw/'


for root, dirs, files in os.walk(dir_path):
    for file in files:
        if '.csv' in file:
            try:

                df = pd.read_csv(f'{root}{file}')
                if df.shape[0] >= 1000 and df.shape[1] >= 10:
                    datasets_meta[file[:-4]] = {
                        'Number Of Samples': df.shape[0],
                        'Number Of Features': df.shape[1],
                        'Number of Classes': len(df[df.columns[-1]].unique())
                    }
                    all_samples.append(df.shape)
                    all_features.append((df.shape[1], df.shape[0]))
            except:
                errors.append(file)



