import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from generate_data.generate_data import generate_more_data



def main():
    data_csv_path = "../data/question.csv"
    df = pd.read_csv(data_csv_path)

    kfold_split = StratifiedKFold(n_splits=5)
    X, y = df["Utterance"].values,  df["Intent"].values

    for i, (train_index, val_index) in enumerate(kfold_split.split(X, y)):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        train_df = generate_more_data(pd.DataFrame({"Utterance": X_train, "Intent": y_train}))
        # val_df = generate_more_data(pd.DataFrame({"Utterance": X_val, "Intent": y_val}))
        val_df = pd.DataFrame({"Utterance": X_val, "Intent": y_val})
        train_df.to_csv(f"../data/train_df_fold_{i+1}.csv", index=False)
        val_df.to_csv(f"../data/val_df_fold_{i+1}.csv", index=False)

if __name__ == "__main__":
    main()
