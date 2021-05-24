import numpy as np
import heapq

class Metric:

    def __init__(self):
        pass

    def get_hits(self, k_ranked, holdout):
        """
        hit 생성 함수
        hit := holdout(df_test의 movie)이 K순위 내에 있는지 여부
        """
        for movie in k_ranked:
            if movie == holdout:
                return 1
        return 0

    def eval_rating(self, idx, test_ratings, test_negatives, K, model):
        """
        holdout(df_test의 movie)이 K순위 내에 있는지 평가하는 함수
        """
        movies = test_negatives[idx]      # negative movies [neg_movie_id, ... ] (1,100)
        user_idx = test_ratings[idx][0]  # [user_id, movie_id][0]
        holdout = test_ratings[idx][1]   # [user_id, movie_id][1]
        movies.append(holdout)            # holdout 추가 [neg_movie_id, ..., holdout] (1,101)

        # prediction
        predict_user = np.full(len(movies), user_idx, dtype='int32').reshape(-1, 1)  # [[user_id], ...], (101, 1)
        np_movies = np.array(movies).reshape(-1, 1)                                   # [[movie_id], ... ], (101, 1)

        predictions = model.predict([predict_user, np_movies])
        predictions = predictions.flatten().tolist()
        movie_to_pre_score = {movie:pre for movie, pre in zip(movies, predictions)}

        # 점수가 높은 상위 k개 아이템 리스트
        k_ranked = heapq.nlargest(K, movie_to_pre_score, key=movie_to_pre_score.get)

        # holdout이 상위 K 순위에 포함 되는지 체크
        # {1:포함, 0:포함x}
        hits = self.get_hits(k_ranked, holdout)

        return hits

    def evaluate_top_k(self, df_neg, df_test, model, K=10):
        """
        TOP-K metric을 사용해 모델을 평가하는 함수
        """
        hits = []
        test_u = df_test['user_id'].values.tolist()
        test_i = df_test['movie_id'].values.tolist()

        test_ratings = list(zip(test_u, test_i))
        df_neg = df_neg.drop(df_neg.columns[0], axis=1)
        test_negatives = df_neg.values.tolist()  # [[(user_id, movie_id=holdout)], neg_movie,... ] (1,100)

        # user 샘플링
        sample_idx_lst = np.random.choice(len(test_ratings), int(len(test_ratings) * 0.3))
        for user_idx in sample_idx_lst:  # 전체 사용: range(len(test_ratings))

            hitrate = self.eval_rating(user_idx, test_ratings, test_negatives, K, model)
            hits.append(hitrate)  # ex. [1,0,1,1,0,...] (1, df_test.shape[0])

        return hits
