## First Class

- 인공지능 > 머신러닝 > 딥러닝

<img src= "https://user-images.githubusercontent.com/107015573/217162431-fe8bcf9f-cf31-4cbf-9e61-ab18ce5c19d7.png" width="400" height="200">



## 그럼 딥러닝과 머신러닝의 차이는 뭘까?
<img src= "https://user-images.githubusercontent.com/107015573/217162663-6ac8665c-6489-4636-a9ea-0522504aa128.png" width="600" height="350">

- 머신러닝
  - Data에 대한 특장(Feature)을 사람(사용자)이 뽑아서 모델에게 주는 방식
- 딥러닝 
  - 특징(Feature)조차도 학습을 시킨 후 모델에서 특징을 고르고 Data를 학습, 분류(Classification)까지 스스로 진행
  - 데이터와 환경만 맞춰주면 모델이 스스로 학습을 진행한다

 ### Deep Neural Network
 - Hidden Layer가 2개 이상인 NN
 <img src= "https://user-images.githubusercontent.com/107015573/217228413-cdcbf628-8171-47ca-ab73-4873ad8dd62e.png" width="600" height="280">

이런 DNN을 학습하는 것이 바로 Deep Learning

> Deep Learning 용어
>- Dataset : 데이터가 들어있는 집합
>   - 학습이나 모델을 검증할 때 사용하는 것 : Train / Validation / test set
>- label : Data를 표현하는 것
>- Model : DeepLearning 어떠한 Network를 뜻
>- Hyper-parameter / Parameter
>- Epoch
>- Batch
>- Loss
>- Optimization / Optimizer
>- Overfiting / Underfiting

------
### What is Parameter / Hyper-parameter
#### Parameter
- 모델 내부에서 결정되는 변수, 데이터를 통해 산출이 가능한 값
- 예측을 수행할 때 모델에 의해 요구되는 갓
    - ex : 가중치(Weight), 편향(Bias), 선형회귀의 결정 계수
- 측정하거나 데이터로부터 학습된다

#### Hyper-Parameter
- 주로 알고리즘 사용자가 경험에 의해 설정하는 값
- 여러번 수행하여 최적을 값을 찾는다 (중요)
    - 여러번 실행하여 이 네트워크에는 이 값이 적절한 것 같다 라고 경험을 해서 만들어지는 값
- 데이터 분석을 통해 얻어지는 값이 아니다
    - EX) Learning Rate(학습률), Epoch(학습 횟수)
    
### What is Epoch / Batch Size / Iteration
 <img src= "https://user-images.githubusercontent.com/107015573/217231219-03ad692f-cc20-4cb9-889f-c4d2696f7ef6.png" width="800" height="180">

전체 데이터셋이 있을 때, 컴퓨터상에 메모리 등의 한계로 인해 한번에 모델에 학습을 시키지 않는다

- Batch : 큰 데이터 셋을 조각내서 학습시킬 때 한 조각을 나타내는 용어
- Epoch : 전체 데이터를 한번 다 도는 것
- Batch Size : Dataset을 얼마나 쪼갠 후 학습을 할지 의미
- Iteration : BatchSize가 10이라고 가정하고 데이터가 100개일 때 Batch는 10개가 생길 것이다. 이때 하나의 Batch를 학습 시키는걸 1 Iteration이라고 한다
    - 퀴즈 : 100개의 Data가 있고 Batchsize는 10일 때 1 Epoch는 몇 Iteration일까?
        - 정답 : 10 Iteration
        
------
## 학습을 할 때 전체적인 흐름
![image](https://user-images.githubusercontent.com/107015573/217232428-a9dbf42a-6755-460a-ad1a-6b794f8ac407.png)
- Input을 할 때, 네트워크에 맞는 데이터 형태를 하기 위해 Data Perprocessing(데이터 전처리)과정을 진행한다

만약 Image Data가 들어왔다는 가정을 할 경우 Data를 Nomalize시키거나 증감시키는 등 데이터를 전처리하는 과정을 거친다

> Hidden & Network Size
>- 모델을 결정하는 단계
>   - 학습 시킬 모델을 어떻게 결정시킬지
>> 용어 정리
>>- Depth(네트워크의 깊이) : Layer의 수 
>>- Width(네트워크의 폭) : Layer별 뉴런 수

> Loss Funtion
>- 네트워크를 정하게 되면 Loss Function을 정하게 된다

> Output
>- 모델이 예측한 값과 정답을 비교해보며 각 Task에 맞는 Evaluation Matrix를 정한다
>- 해당 값을 보면서 학습이 잘 되었는지 못되었는지 평가 진행
>   - 평가 지표에 따라서 Weight, Bias를 지속적으로 업데이트한다

-----------
모델을 위해서 아키텍쳐를 구성하고 나면 Training Roop라는 것을 통과하게 된다

### model.train()
- 모델이 지금 Train 상태인지, Test 상태인지 선언을 해주어야한다
  - 모델을 향상시키기 위해서 다양한 방법을 쓰는데 이게 Train에서 작용하는거랑 Test에서 작용하는거랑 달라지는 경우가 있다
  - 그 다음부터 본격적으로 진행하는 것

### 모델 학습 과정
> Tr_1. Training Mode
> - model.train()
>> For each train batch
>>- Tr_2 . Transfer Data to GPU if available
>>- Tr_3 . Clear Gradients
>>- Tr_4 . forward pass
>>- Tr_5 . compute loss
>>- Tr_6 . compute gradients
>>- Tr_7 adjust learnable parameters
>>- Tr_8 [Opt]Calculate total loss
>>- Tr_9 [Opt]Save weight for every C Cycle

#### Ex Code
![image](https://user-images.githubusercontent.com/107015573/217235580-6ff1824e-e0dc-4fb9-8289-bcc663896d23.png)

- Red dot Part : 몇번 돌지 정하는 Epoch 수
- Red Part : Train인지 Test인지 선언해주는 부분
- Green Part : 예측할 값을 뽑아내는 것
- Blue Part : 예측값과 내가 가진 실체 라벨 값을 비교해서 평가
  - 내가 쓰는 모델이 얼마나 정답에 근접한지를 판단할 수 있는 척도
  - Loss가 줄어들어야 모델의 학습이 잘 진행되고 있는 것
- Yellow Part : Optimizer라고 불리는 값을 Zero_grad함수를 통해 Reset
  - Optimizer : 뒤에 BackPropagation에서 자세하게
- Pink Part : Loss같은 Backpropagation을 진행
- Last : Optimizer을 재조절 후 다시 맨 위부터 Loop을 다시 도는 것
----------------
### Overfitting, Underfitting, Optimal
![image](https://user-images.githubusercontent.com/107015573/217237044-4144eabc-0b4a-4753-b9b9-553347b6db18.png)

보통 Underfitting보다는 Overfitting이 더욱 자주 발생한다

> Overfitting이 발생할 수 있는 사유
>- 데이터가 부족하다
>- 모델이 복잡하다

Overfitting이란?
- 모델이 데이터에 너무 편향되게 학습되는 것
- 이 경우 Test Data에서는 Loss가 적은, 학습이 잘 되는 것 처럼 보이지만, Train 데이터 등 타 데이터를 넣을 경우 갑작스럽게 Loss가 증가하고 값이 이상해지는 현상이 발생한다
    - ex) 답안지를 외웠는데 막상 시험에서 시험지가 바뀌어서 출제가 되어 시험을 망치는 것

이러한 Overfitting을 규제하거나 모델을 단순화시켜서 해결을 하고자 노력한다

