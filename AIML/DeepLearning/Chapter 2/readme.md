# Deep Neural Network(DNN)
- 3층 신경망 + softmax

![image](https://user-images.githubusercontent.com/107015573/218976379-f151784d-171e-4d49-80b4-18773987f9b6.png)

입력층에 $x_1, x_2$이 있고 다음 Layer인 Hidden Layer 1에 $a_1, a_2, a_3$이 있다

입력층 2번째 Node에서 다음 hidden Layer 1층으로 가는데 표기하는 법이 위 그림에 표시가 되어있다
- Input Layer에 2번째 뉴런에서 다음 층 1번째 뉴런으로 이동, 위에 가중치를 표시한다

그렇다면 만약에 w<sup>1</sup><sub>2, 2</sub> 일 경우에는 무슨 의미일까요?
- 정답
  - 앞 층 2번째 뉴런에서 다음 층 2번째 뉴런으로 연결된다, 가중치는 1이라는 의미

> 배우는 이유
> - 신경망을 수식으로도 표현할 수 있다

### 신경망 수식 표현
<img src= "https://user-images.githubusercontent.com/107015573/218979848-b3028942-b9d3-4063-810c-6949d0446311.png" width="400" height="220">
<img src= "https://user-images.githubusercontent.com/107015573/218979855-4f6e9410-3b4a-4d2b-b5f8-bc582ff45bd4.png" width="200" height="30">

> 그림 설명
> - a<sup>(1)</sup><sub>1</sub>는 이전 Input Layer의 1,2,3번째 노드에 가중치를 곱한 값의 합이다
> - 마지막에 있는 b<sup>(1)</sup><sub>1</sub>은 편향, bias 값이다
>     - bias? : 뉴런의 활성화 난이도를 조절해주는 상수
>> 용어 의미
>> - 가중치(Weight)
>>    - 노드와 노드 간 연결 강도를 조정하는 값
>>    - 입력이 원하는 연산결과와 유사한 결과가 나올 수 있도록 중요도를 부여
>> - 편향(Bias)
>>    - 가중치와 신호를 곱한 값 : 가중합
>>    - 가중합이 임계점을 넘을 시 뉴런이 활성화 된다
>>    - 주어지지 않은 입력의 출력을 예측하는 것
>>      - 즉, 일반화의 성능을 높이기 위해서 만약의 상황에 대한 추가적인 가정(Additional Assumptions)

### Add Activation Function
<img src= "https://user-images.githubusercontent.com/107015573/218982999-6be116ca-214d-4e70-a0c7-d6c742b2f571.png" width="600" height="320">

- 신경망의 비선형성을 위해서  Activation Function을 넣는다
  - h( )는 Activation을 뜻한다
  - a<sup>(1)</sup><sub>1</sub>은 가중치와 입력을 곱한 가중합을 의미한다
  - z<sup>(1)</sup><sub>1</sub>은 가중합이 Activation을 통과한 값을 의미한다
<img src= "https://user-images.githubusercontent.com/107015573/218984141-35060cbc-ddf4-481e-8529-8cf4a13309ec.png" width="600" height="320">

이러한 과정을 반복하면 OUTPUT이 정상적으로 나오는 것을 확인할 수 있을 것 이다

![image](https://user-images.githubusercontent.com/107015573/218983639-40a63253-bcf3-44e8-8151-b69a1b9efc02.png)
- Activation Function 중 Sigmoid 함수의 예시

![image](https://user-images.githubusercontent.com/107015573/218983970-8aceb265-23c6-447b-911b-3fde76eda845.png)
- 그 외 다양한 Activation Function 함수의 종류

---

## 실습 코드 설명

#### Sigmoid 함수, 항등 함수 작성
```python
# 활성화 함수 
def sigmoid(x):

    return 1 / (1 + np.exp(-x))

# 항등 함수 
def identity_func(x):

    return x
```

### 2-3-2구조
- input : 2개, hidden 3개, output 2개가 나오는 구조
```python
network = {}    #빈딕셔너리 하나 생성

network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]) #weight(가중치)
# layer가 2인 곳에서 hidden layer(3)으로 가기 때문에 (2, 3)으로 크기를 만들었다

network['b1'] = np.array([0.1, 0.2, 0.3]) 
# bias에 값이 들어가야 하기 때문에 1,3짜리로 만들어주었다

network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]) #(3,2)
# hidden layer(3)에서 output layer(2)로 가기 때문에 이에 맞추서 (3,2)로 만든다

network['b2'] = np.array([0.1, 0.2]) 
# 이에 맞춰서 bias로 (1,2)로 만들어준다

network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]]) #(2, 2)
# 최종 output이 2가 나오도록 하기 위해서 (2,2)로 할당을 한다

network['b3'] = np.array([0.1, 0.2])
```

```python
#각각의 네트워크에 알맞게 포워딩을 해준다
W1, W2, W3 = network['W1'], network['W2'], network['W3']
b1, b2, b3 = network['b1'], network['b2'], network['b3']

#W1, W2, W3에 각각 알맞는 W1, W2, W3을 할당
#b1, b2, b3에 각각 알맞는 b1, b2, b3을 할당

a1 = np.dot(x, W1) + b1
#np.dot = 내적을 하기 위해서 사용
#뒤에 바이어스를 추가함으로써 각각의 중요도를 구분한다

#(2,3), (3,2)를 내적하면 (2,2)가 나온다

z1 = sigmoid(a1)
#activation function의 일종인 sigmoid에 summation 된 값을 넣어 z1이라는 결과값을 도출해낸다

a2 = np.dot(z1, W2) + b2
z2 = sigmoid(a2)
#앞선 방법과 동일하게 진행된다(layer만 다를 뿐)


```
