import pandas as pd
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt







if __name__ == "__main__":

    # Import, cleanup, and partition dataset.
    col_names = [
        "sepal_length", "sepal_width", "petal_length", "petal_width", "species"
    ]
    categorical_species = {
        "species" : {
            "Iris-setosa" : 1,
            "Iris-versicolor" : 2,
            "Iris-virginica" : 3
        }
    }

    main_df = pd.read_csv("iris.data", header=None, names=col_names)
    main_df = main_df.replace(
        categorical_species
    ).reindex(
        np.random.permutation(main_df.index)
    )
    train_df = main_df[:100]
    test_df = main_df[100:125]
    valid_df = main_df[125:]
    print(train_df)

