import pandas as pd
import numpy as np

from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras import models
import matplotlib.pyplot as plt
from numpy import float32

# def create_model():


if __name__ == "__main__":
    # Import, cleanup, and partition dataset.
    col_names = [
        "sepal_length", "sepal_width", "petal_length", "petal_width", "species"
    ]
    categorical_species = {
        "species": {
            "Iris-setosa": 0,
            "Iris-versicolor": 1,
            "Iris-virginica": 2
        }
    }

    main_df = pd.read_csv("iris.data", header=None, names=col_names)
    main_df = main_df.replace(categorical_species)
    main_df = main_df.reindex(np.random.permutation(main_df.index))

    one_hot_species = to_categorical(main_df["species"])

    X_train, X_test = \
        np.array(main_df.iloc[:125, :4].values.tolist()), \
        np.array(main_df.iloc[125:, :4].values.tolist())

    Y_train, Y_test = \
        np.array(one_hot_species[:125]), \
        np.array(one_hot_species[125:])

    print(X_train[:1])
    print(Y_train[:1])

    """ 
    didnt get this part ngl google
    
        sep_width = tf.feature_column.numeric_column("sepal_width")
        sep_length = tf.feature_column.numeric_column("sepal_length")
        pet_width = tf.feature_column.numeric_column("petal_width")
        pet_length = tf.feature_column.numeric_column("petal_length")
    
        feature_cols = [sep_width, sep_length, pet_length, pet_width]
    
        feature_layer = layers.DenseFeatures(feature_cols)
        feature_layer(dict(main_df))
    """

    model = models.Sequential()
    model.add(layers.Dense(20, activation="relu", input_shape=(4,)))
    model.add(layers.Dense(3, activation="softmax"))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train, Y_train,
                        epochs=500,
                        batch_size=10,
                        validation_data=(X_test, Y_test))

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, 'bo', label='training loss')
    plt.plot(epochs, val_loss, 'b', label='validation loss')
    plt.title('model loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()



    weights = model.get_weights()










