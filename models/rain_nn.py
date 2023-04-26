import os
import pandas as pd

from sklearn.metrics import confusion_matrix, classification_report

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras import callbacks


# Class for the neural network model to predict if it will rain tomorrow
class RainNN:
    """
    Class for the neural network model to predict if it will rain tomorrow.
    """
    
    def __init__(self):
        """
        Initialise the model. If the model is already trained, load it from disk.
        Otherwise, train the model and save it to disk.
        """
        model_path = os.path.join(os.path.dirname(__file__), 'rain_nn.h5')

        if os.path.exists(model_path):
            self.model =load_model(model_path)
            print("Model loaded from disk")
        else:
            self.train()
            print("Model trained and saved to disk")
        

    def train(self):
        """
        Train the neural network model and save it to disk.
        """

        # Reading the datasets and splitting into X and y
        train_df = pd.read_csv("dataset\cleaned_weatherAUS_train.csv")
        test_df = pd.read_csv("dataset\cleaned_weatherAUS_test.csv")
        X_train = train_df.drop(["RainTomorrow"], axis=1)
        y_train = train_df["RainTomorrow"]
        X_test = test_df.drop(["RainTomorrow"], axis=1)
        y_test = test_df["RainTomorrow"]

        # Early stopping
        # early_stopping = callbacks.EarlyStopping(
        #     min_delta=0.001, # minimum amount of change to count as an improvement
        #     patience=20, # how many epochs to wait before stopping
        #     restore_best_weights=True,
        # )

        # Initializing the NN and adding the input layers
        self.model = Sequential()
        self.model.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu', input_dim = 26))
        self.model.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'))
        self.model.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))
        # self.model.add(Dropout(0.25))
        self.model.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
        # self.model.add(Dropout(0.5))
        self.model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

        # Compiling the NN
        opt = Adam(learning_rate=0.00009)
        self.model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])

        # Train the NN
        self.model.fit(X_train, 
                       y_train, 
                       batch_size = 32, 
                       epochs = 200, 
                    #    callbacks=[early_stopping], 
                       validation_split=0.2)

        # Predicting the test set results
        y_pred = self.model.predict(X_test)
        y_pred = (y_pred > 0.5)

        # print confusion matrix classification_report
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        # save the model to file
        self.model.save('models/rain_nn.h5')

        # 1st survey: to pick probability between 0-0.25 or 0.75-1, and the outcome will be 0 or 1 randomly.
        # 2nd survey:70% accuracy on test set without calibration



    def predict(self, X):
        """
        Predict the probability of rain for a given set of features. 
        :param X: features to predict on 
        :return: probability of rain for each sample in X 
        """
        return self.model.predict(X)
