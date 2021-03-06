import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from constant.constant_data_make import *
import matplotlib.pyplot as plt
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class FFN:
    def __init__(self, input_shape, train_feature, train_label, test_feature, test_label, epochs=2000, unit=30, hidden=5, save_path=None, load_path=None, check_seed=None):
        self.input_shape = input_shape
        self.train_feature = train_feature
        self.train_label = train_label
        self.test_feature = test_feature
        self.test_label = test_label
        self.epochs = epochs
        self.unit = unit
        self.hidden = hidden
        self.save_path = save_path
        self.load_path = load_path
        self.check_seed = check_seed
        # self.fold = fold

    # model load
    def load(load_path, test_feature):
        json_file = open(load_path + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = tf.keras.models.model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(load_path + ".h5")
        # model = load_model(load_path)
        # score = model.evaluate(self.test_feature, self.test_label)
        pred = model.predict(x=test_feature)

        return pred[0][0]

    def run(self):
        model = self.init_model()

        if self.check_seed is not None:
            checkpoint = tf.keras.callbacks.ModelCheckpoint("./check/model checkpoint epoch2_" + str(self.check_seed) + "_{epoch:02d}.h5", monitor='loss',
                                                            save_freq=100)
            csv_logger = tf.keras.callbacks.CSVLogger('./2_model_history.csv')
            callbacks = [csv_logger, checkpoint]

        # self.train_feature = np.array(self.train_feature)
        # self.train_label = np.array(self.train_label)
        # self.test_feature = np.array(self.test_feature)
        # self.test_label = np.array(self.test_label)

        history = model.fit(self.train_feature, self.train_label,
                  batch_size=np.size(a=self.train_feature, axis=0),
                  epochs=self.epochs, verbose=1)
        # history = model.fit(self.train_feature, self.train_label,
        #                     batch_size=int(len(self.train_feature)/10),
        #                     epochs=self.epochs, verbose=0)

        score_train = model.evaluate(self.train_feature, self.train_label)
        print('train score : ', score_train)
        score = model.evaluate(self.test_feature, self.test_label)
        print('test score : ', score)
        y_pred = model.predict(x=self.test_feature)


        plt.plot(history.history['loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        # plt.legend(['Train', 'Test'], loc='upper left')
        # plt.show()

        # save
        if self.save_path is not None:
            # tf.keras.models.save_model(model=model, filepath=base_path + 'save_model/model_' + self.save_path + '.h5')
            model_json = model.to_json()
            with open(base_path + "save_model/model_" + self.save_path + ".json", 'w') as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights(base_path + 'save_model/model_' + self.save_path + '.h5')
        return score, y_pred, model

    def init_model(self):
        # 1) Initialize Sequential
        model = tf.keras.Sequential()

        # 2) Add layer
        model.add(layers.Dense(units=20, input_dim=self.input_shape))
        for i in range(1, self.hidden):
            model.add(layers.LeakyReLU())
            model.add(layers.Dense(units=self.unit, kernel_initializer='glorot_uniform',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        model.add(layers.Dense(units=1, kernel_initializer='glorot_uniform',
                               kernel_regularizer=tf.keras.regularizers.l2(0.01)))


        # 3) Initialize cost function, optimization algorithm, and metrics
        model.compile(optimizer='adam', loss='mean_absolute_percentage_error')
        return model

    def accuracy(self, target, pred):
        target = K.round(K.clip(x=target, min_value=0, max_value=1))
        pred = K.round(K.clip(pred, 0, 1))
        accuracy = K.mean(K.equal(target, pred))
        return accuracy

    def recall(self, target, pred):
        """ Calculate recall
            Args:
                target: target values of data
                pred: prediction values of model
            1) Clip target and pred appropriately
            2) Calculate true positive
            3) Calculate false negative + true positive
            4) Calculate recall
        """
        # 1) Clip target and pred appropriately
        target = K.round(K.clip(x=target, min_value=0, max_value=1))
        pred = K.round(K.clip(pred, 0, 1))
        """ tf.keras.backend.clip
                Element-wise value clipping.
                Args:
                    x: Tensor or variable.
                    min_value: Python float or integer.
                    max_value: Python float or integer.
        """
        """ tf.keras.backend.round
                Element-wise rounding to the closest integer.
                In case of tie, the rounding mode used is "half to even".
                Args:
                    x: Tensor or variable.
        """
        # 2) Calculate true positive
        # Recall that every value is 1 or 0 so, sum means amount of data which is 1.
        # And target * pred means AND(target, pred) so, they activate only when they are true positive
        n_true_positive = K.sum(target * pred)

        # 3) Calculate false negative + true positive
        n_true_positive_false_negative = K.sum(target)

        # 4) Calculate recall
        # recall =  (true Positive) / (true Positive + false Negative)
        # We add very small value by using K.epsilon() to prevent division by zero error
        recall = n_true_positive / (n_true_positive_false_negative + K.epsilon())
        return recall

    def precision(self, target, pred):
        """ Calculate precision
            Args:
                target: target values of data
                pred: prediction values of model
            1) Clip target and pred appropriately
            2) Calculate true positive
            3) Calculate amount of retrieved data
            4) Calculate precision
        """
        # 1) Clip target and pred appropriately
        target = K.round(K.clip(target, 0, 1))
        pred = K.round(K.clip(pred, 0, 1))
        """ tf.keras.backend.clip
                    Element-wise value clipping.
                    Args:
                        x: Tensor or variable.
                        min_value: Python float or integer.
                        max_value: Python float or integer.
            """
        """ tf.keras.backend.round
                Element-wise rounding to the closest integer.
                In case of tie, the rounding mode used is "half to even".
                Args:
                    x: Tensor or variable.
        """
        # 2) Calculate true positive
        # Recall that every value is 1 or 0 so, sum means amount of data which is 1.
        # And target * pred means AND(target, pred) so, they activate only when they are true positive
        n_true_positive = K.sum(target * pred)

        # 3) Calculate amount of retrieved data
        # amount of retrieved data = false positive + true positive
        n_retrieved_data = K.sum(pred)

        # 4) Calculate precision
        # precision = (true Positive) / (false positive + true positive)
        # We add very small value by using K.epsilon() to prevent division by zero error
        precision = n_true_positive / (n_retrieved_data + K.epsilon())
        return precision