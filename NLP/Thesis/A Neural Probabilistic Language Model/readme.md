# 논문 리뷰

### 자연어 처리 분야에는 2가지 기반의 모델이 있습니다

1. 통계 기반
2. neural network 기반

이번에 읽은 A Neural Probabilistic Language Model라는 논문은 neural network 기반 언어 모델을 처음으로 제안한 논문 중 하나입니다. 이전의 언어 모델은 n-gram과 같은 통계 기반 모델이 대부분이었지만, 이 논문에서는 단어 시퀀스를 다루기 위해 신경망 모델을 사용하였습니다.

## 논문 요약
논문에서는 단어 시퀀스의 확률 분포를 추정하기 위한 뉴럴 네트워크 모델인 Neural Probabilistic Language Model을 제안합니다. NPLM은 단어 시퀀스를 입력받아 다음 단어를 예측하는 작업을 수행하며, 각 단어의 확률을 출력합니다. NPLM 모델은 단어들을 distributed representation으로 표현하고, 이를 통해 단어 간 유사도를 계산합니다. 또한, n-gram 방식과는 달리 이전에 나온 모든 단어를 고려하며, 이를 위해 feedforward neural network를 사용합니다.

NPLM은 backpropagation 알고리즘을 사용해 학습되며, 학습 과정에서 noise-contrastive estimation(NCE)을 사용하여 빠른 학습을 가능하게 한다. 실험 결과, NPLM은 기존 n-gram 모델들과 비교해 좋은 성능을 보여주었고, NPLM이 학습한 distributed representation이 다른 자연어 처리 작업에서도 좋은 성능을 보인다는 것을 보여

> 논문에서 제시한 강점
>- n-gram 방식에서 발생하는 정보 손실 문제를 해결한다
>- 타 자연어 처리 작업에서도 좋은 성능을 보인다
>- NCE를 사용하기 때문에 빠른 학습이 가능하다

> 문제점
>- 데이터셋을 학습하는데 시간이 오래 걸린다
>- 큰 사이즈의 어뤼를 다루는데 있어 문제가 발생한다

### 논문의 기여
- 모든 단어를 고려한 distributed representation을 제안함으로써 자연어 처리 분야에서의 성능을 크게 향상시켰다
- NCE를 사용하여 학습 속도를 크게 향상시켰습니다.
- 해당 논문 출간 이후, 자연어 처리 분야에서도 distributed representation을 사용하는 뉴럴 네트워크 모델이 많이 연구되게 되었다
