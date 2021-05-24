import numpy as np
from sklearn.utils import shuffle
from Loader import Loader
from Metric import Metric
from model.NeuMF import NeuMF

class Run:

    def __init__(self):

        # data 로드
        loader = Loader()

        print('start data load..')

        num_neg = 4
        uids, mids, self.df_train, self.df_test, \
        self.df_neg, self.users, self.movies, movie_lookup = loader.load_dataset()
        user_input, movie_input, labels = loader.get_train_instances(uids, mids, num_neg, len(self.movies))

        print('end data load..')

        # input data 준비
        user_data_shuff, movie_data_shuff, label_data_shuff = shuffle(user_input, movie_input, labels)
        self.user_data_shuff = np.array(user_data_shuff).reshape(-1,1)
        self.movie_data_shuff = np.array(movie_data_shuff).reshape(-1,1)
        self.label_data_shuff = np.array(label_data_shuff).reshape(-1,1)

    def run(self):

        nmf = NeuMF(len(self.users), len(self.movies))  # Neural Collaborative Filtering
        self.model = nmf.get_model()
        self.model.fit([self.user_data_shuff, self.movie_data_shuff], self.label_data_shuff, epochs=20,
                       batch_size=256, verbose=1)

        return self.model

    def calculate_top_k_metric(self):
        metric = Metric()
        hit_lst = metric.evaluate_top_k(self.df_neg, self.df_test, self.model, K=10)
        hit = np.mean(hit_lst)

        return hit

if __name__ == '__main__':

    ncf = Run()
    model = ncf.run()

    # top-k metric
    top_k_metric = ncf.calculate_top_k_metric()
    print('metric:', top_k_metric)

    # user 한 명에 대한 prediction 예시
    user_id = 0
    user_candidate_movie = np.array([134, 6783, 2788, 8362, 25]).reshape(-1, 1)
    user_input = np.full(len(user_candidate_movie), user_id, dtype='int32').reshape(-1, 1)
    predictions = model.predict([user_input, user_candidate_movie])
    predictions = predictions.flatten().tolist()
    movie_to_pre_score = {movie[0]: pre for movie, pre in zip(user_candidate_movie, predictions)}  # 후보 아이템 별 예측값
    movie_to_pre_score = dict(sorted(movie_to_pre_score.items(), key=lambda x: x[1], reverse=True))

    recommend_movie_lst = list(movie_to_pre_score.keys())
    print('recommend:', recommend_movie_lst)
