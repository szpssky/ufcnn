from tensorflow.contrib.keras.python.keras.models import Model
from tensorflow.contrib.keras.python.keras.layers import Conv1D,Input, add



class ufcnn:
    def __init__(self, time_step, input_size, output_size,n_filters,n_filter_length):
        self.time_step = time_step
        self.input_size = input_size
        self.output_size = output_size
        self.filters = n_filters
        self.filter_length = n_filter_length
        self.model = self.build_model(self.filters, self.filter_length)

    def build_model(self, filters, filter_length, optimizer='rmsprop', loss='mse'):
        input_ = Input(shape=(self.time_step, self.input_size))
        conv_h1 = Conv1D(filters, filter_length, padding='causal', activation='relu')(input_)

        conv_h2 = Conv1D(filters, filter_length, padding='causal', activation='relu')(conv_h1)

        conv_h3 = Conv1D(filters, filter_length, padding='causal', activation='relu')(conv_h2)

        conv_g3 = Conv1D(filters, filter_length, padding='causal', activation='relu')(conv_h3)

        merge_h2_g3 = add([conv_h2, conv_g3])

        conv_g2 = Conv1D(filters, filter_length, padding='causal', activation='relu')(merge_h2_g3)

        merge_h1_g2 = add([conv_h1, conv_g2])

        conv_g1 = Conv1D(filters, filter_length, padding='causal', activation='relu')(merge_h1_g2)

        conv_g0 = Conv1D(self.output_size, filter_length, padding='causal', activation='relu')(conv_g1)

        model = Model(input_, conv_g0)

        model.compile(optimizer, loss)

        model.summary()

        return model

    def train(self,train_data_x,train_data_y,n_epochs=1,validation_split=0.2,callbacks=None):
        self.model.fit(train_data_x,train_data_y,batch_size=128,epochs=n_epochs,validation_split=validation_split,
                  callbacks=callbacks)
    def load_weight(self,check_point):
        self.model.load_weights(check_point)

    def predict(self,x):

        return self.model.predict(x,1)

