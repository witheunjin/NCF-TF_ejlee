# Neural_CF_ejlee

* NCF-TF_ejlee (Neural Collaborative Filtering Model with TensorFlow)

* Reference: https://github.com/LeeHyeJin91/Neural_CF

* My Repository URL: https://github.com/witheunjin/NCF-TF_ejlee

## [DATASETS]
```
|__ path: ~/NCF-TF_ejlee/data
      |__ /100K/ratings.csv: 100K 개의 Ratings data
      |__ /1M/ratings.csv: 1M 개의 Ratings data
      |__ /20M/ratings.csv: 20M 개의 Ratings data (Github에는 용량문제로 인해 미포함)
      |__ /25M/ratings.csv: 25M 개의 Ratings data (Github에는 용량문제로 인해 미포함)
      |__ /27M/ratings.csv: 27M 개의 Ratings data (Github에는 용량문제로 인해 미포함)
```     

## [HOW TO USE: 1. Dataset 조회(Dataset Lookup)]
`~/NCF-TF_ejlee$ python Run.py --help` 명령어를 통해 아래와 같은 Training에 사용할 수 있는 Dataset의 종류와 상세정보를 출력하여 확인할 수 있다.
(You can look up specifications of datasets that you can use when training this model by using `python Run.py --help` command)

`~/NCF-TF_ejlee$ python Run.py --help`
**RESULT**
```
usage: Run.py [-h] [--data_size DATA_SIZE]

Run NCF.

optional arguments:
  -h, --help            show this help message and exit
  --data_size DATA_SIZE
                       Data Size(ex.NAME(Ratings|Movies|Users))
                        |__100K(100,000|9,000|600)
                        |__1M(1,000,000|4,000|6,000)
                        |__20M(20,000,000|27,000|138,000)
                        |__25M(25,000,000|62,000|162,000)
                        |__27M(27,000,000|58,000|280,000)
```

## [HOW TO USE: 2. 실행방법(How to execute)]
~/NCF-TF_ejlee에서 다음과 같은 명령어를 사용하여 실행
* `$python Run.py --data_size ###` : ###부분에 원하는 데이터크기값(100K, 1M, 20M, 25M, 27M 택 1)을 넣어준 후 실행(ex. `python Run.py --data_size 100K`)

## [RESULTS]
* NCF-TF_ejlee_100K_result: 100K Dataset에 대한 Training 결과(Epoch 20)
* NCF-TF_ejlee_1M_result: 1M Dataset에 대한 Training 결과(Epoch 20)

