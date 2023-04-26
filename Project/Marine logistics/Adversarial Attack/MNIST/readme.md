## MNIST DATASET을 활용하여 적대적 예제 생성

### Code review

MNIST 데이터셋을 사용하여 적대적 예제를 생성하고 시각화하며, 모델의 예측 결과와 정확도를 출력합니다

- 먼저, np.random.choice 함수를 사용하여 x_test에서 10개의 이미지를 무작위로 선택합니다.
  - 이때, 선택한 이미지들에 대해 for 루프를 수행하며, image와 해당 image의 레이블 label을 변수로 저장합니다
  
- generate_adversarial_example 함수를 사용하여 image에 대한 적대적 예제 adv_image를 생성합니다.
  - epsilo 값은 0.1로 설정한다

- image와 adv_image를 시각화하고, model에 입력하여 예측 결과를 출력
  - 원래 이미지의 예측 결과는 pred_image에, 적대적 예제의 예측 결과는 pred_adv_image에 저장
  - matplotlib 라이브러리를 사용하여 시각화

- 정확도를 측정하여 출력
  - accuracy 함수 사용해서 원래 이미지와 적대적 예제 각각에 대해 정확도 계산 후 출력
