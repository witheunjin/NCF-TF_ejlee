import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

class GMP:

    def __init__(self, user_num, movie_num):

        latent_features = 8

        # User embedding
        user = Input(shape=(1,), dtype='int32')
        user_embedding = Embedding(user_num, latent_features, input_length=user.shape[1])(user)
        user_embedding = Flatten()(user_embedding)

        # movie embedding
        movie = Input(shape=(1,), dtype='int32')
        movie_embedding = Embedding(movie_num, latent_features, input_length=movie.shape[1])(movie)
        movie_embedding = Flatten()(movie_embedding)

        # Merge
        concatenated = Multiply()([user_embedding, movie_embedding])

        # Output
        output_layer = Dense(1, kernel_initializer='lecun_uniform', name='output_layer')(concatenated) # 1,1 / h(8,1)초기화

        # Model
        self.model = Model([user, movie], output_layer)
        self.model.compile(optimizer='adam', loss='binary_crossentropy')

    def get_model(self):
        model = self.model
        return model





