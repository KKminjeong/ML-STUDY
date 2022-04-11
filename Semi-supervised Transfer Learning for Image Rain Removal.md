# Semi-supervised Transfer Learning for Image Rain Removal

- Image Rain Removal을 위한 Semi-supervised Transfer Learning

## Semi-supervised Transfer Learning for Image Rain Removal 제안 이유

- Deep Learning 방식으로 이미 Rain Removal 문제에 대해 효과적인 성과를 얻을 수 있었다. 하지만 이전 Deep Learning 방식은 원하는 성능을 내기 위해서 rain streaks가 합성된 이미지, rain streaks가 없는 깨끗한 이미지 쌍의 데이터들이 엄청나게 많이 필요하다. (**데이터 부족 문제**)
- 그리고 특정 rain streaks 패턴에 편향된 학습을 해서 일반화가 잘 안되는 문제가 있다. (**일반화 문제**)

## 제안 방안

- 위 문제들을 어느 정도 해결하기 위해서 추가적으로 network 훈련 과정에  실제 rain streaks 이미지를 준다. (**이전 방식들과의 가장 큰 차이점**)
- 실제 rain streaks 이미지(input image) - 실제 rain streaks 이미지가 network에 들어가서 나오는 예측 이미지 = residual
  - **이** **residual를 특정 파라미터로 정교하게 공식화 시켜서 학습에 참여 시킨다.**
- network는 synthesized rain 이미지와 clean 이미지 쌍으로 학습되고, 이 학습이 실제 rain streaks 이미지에 transfer learning이 되어 학습이 된다.
  - 데이터 부족 문제와 supervised sample에 편향되는 문제를 모두 완화를 시킬 수 있다.

![Untitled1](https://user-images.githubusercontent.com/86338750/162739230-3b4ecbc8-4cfb-4199-b3a9-36a19e98e2ae.png)

## 이전 방식들의 문제를 더 구체적으로 살펴보기

- 합성 데이터 문제

  - 실제 비가 오는 시나리오에서 clean / rainy 이미지 쌍을 얻기 힘들기 때문에, 이전 방법들은 대체 방법으로 Photoshop으로 “Fake” rain streaks를 합성하는 방식을 채택했다.

  ![Untitled2](https://user-images.githubusercontent.com/86338750/162739818-ad35bf89-e35e-44a3-8b7c-97c6fbc469a9.png)

  - 합성된 rainy 이미지는 (b)와 (c)에서 확인할 수 있다. ((a) 원본, (b)와 (c)는 합성된 이미지)
  - 합성 시 다양한 rain streaks 방향과 밀도를 적용했지만, 실제 rain streaks 이미지에서 더 다양한 범위의 패턴들을 포함하지 않는다.

  ![Untitled3](https://user-images.githubusercontent.com/86338750/162739868-c92324bc-560b-4e58-8cfa-551eb5032f7a.png)
  
  - (d)의 경우, rain streaks는 바람에 의해 영향을 받아 한 frame안에서 여러 방향을 가지고 있다.
  - (e)의 경우, rain streaks는 rain streaks와 카메라까지 거리가 다르기 때문에 multi-layer를 가진다.
  - (f)의 경우, rain streaks가 fog 또는 mist와 유사한 효과를 만들어낸다.
  - 위 3가지의 경우를 합성 rainy 이미지들로는 표현하지 못한다.
    - 합성된 훈련 데이터와 실제 테스트 데이터 사이에는 명백한 bias가 존재하고 이 bias때문에 실제 rainy 이미지에 일반화가 잘 안된다.

- Deep Learning 방식의 주된 문제들 중 하나는 많은 supervised sample들이 필요하다는 것이다.

  - Deraining network를 위해서 이 데이터셋들을 생성하는 것은 일반적으로 너무 많은 시간이 소모되고, 이상적인 데이터 쌍을 모으기 너무 힘들다.
  - 하지만 clean 이미지 없이 진짜 비가 오는 이미지는 쉽게 많이 얻을 수 있다.
    - Semi-supervised learning을 채택한 이유 중 하나

## 위 문제들을 해결하기 위해 시도해 볼 수 있는 방법들

- supervised dataset을 더 모으는 것 대신에, unsupervised input으로 overfitting문제를 방지할 수 있다.

  - **왜?** 합성 비 이미지가 가지는 한정된 패턴의 한계를 보완할 수 있기 때문이다.

- 논문에서 Image restoration task를 위한 supervised 그리고 unsupervised knowledge를 동시에 활용하기 위한 일반적인 방법론을 제시한다.

- Supervised의 경우

  - network output 이미지들과 ground truth의 깨끗한 이미지들 사이의 least square loss를 채택한다.

- Unsupervised의 경우

  - residual에 대한 domain 이해를 기반으로 설계된 파라미터화 시킨 분포에 부과된 likelihood 항을 통해 예상되는 clean image들 출력과 원래 노이즈가 있는 이미지(비가 내리는 이미지) 사이의 residual를 공식화한다.

  ![Untitled4](https://user-images.githubusercontent.com/86338750/162739876-8bd41152-dd90-4e79-b7e4-c042dc644680.png)

- 논문에서 제안한 모델을 해결하기 위해 gradient descent strategy를 모으는 Expectation Maximization algorithm 설계

  ![Untitled5](https://user-images.githubusercontent.com/86338750/162739887-ffaca16a-d564-44ca-bacc-aea23ddd9868.png)

  - rain distribution parameter들과 network parameter들이 각 epoch에서는 sequence로 인해 최적화될 수 있다.

- 합성된 rainy image들에서 학습이 진행되고, 합성된 rainy image로부터 학습된 지식들을 이용해서 실제 비 이미지들을 transfer learning해서 실제 비 패턴에 대해서도 학습이 가능하므로 제안된 방법은 최신 기술에 비해 우수하다.

## Unsupervised data에 대한 공식

![Untitled6](https://user-images.githubusercontent.com/86338750/162739894-e9bc226b-d78d-4d74-b230-94924563b2bc.png)

- 그림 (d), (e), (f) → 실제 비는 합성된 비와 비교해서 상대적으로 더 복잡한 패턴과 표현을 가진다. 하지만 기술적인 문제로 실제 비의 label(clean image)는 만들 수 없다.
- 실제 비 이미지에서부터 clean background 뿐만 아니라 rain layer를 힘들고 어렵게 추출할 수는 있지만, real rainy image의 stochastic configuration들을 적절하게 근사화하는 parameterized distribution을 설계한다.
- 비는 카메라에서 다른 거리 위치에서 발생해서 일반적으로 multi-model structure들을 포함하기 때문에, Gaussian Mixture Model로 비를 근사하게 표현할 수 있다.

![Untitled7](https://user-images.githubusercontent.com/86338750/162739902-bc9c4de0-1794-4124-99e6-d4d7ce4e16f5.png)

- Mixture model들은 parameter들이 적절히 학습되어 있다면, 어떤 Continuous function으로도 균일한 근사화가 가능하다, 그러므로 Mixture model은 input rainy image에서부터 뽑아내는 rain streake들을 묘사하는데 적절히 활용된다.
- 이 Unsupervised sample들에 부과된 negative log likelihood function은 다음과 같이 쓸 수 있다.

![Untitled8](https://user-images.githubusercontent.com/86338750/162739913-e57f3810-ae9e-41d2-b0cd-569a3d470cb0.png)

- 위의 encoding 방법을 활용함으로써, Unsupervised rainy image들에 대한 objective function을 구성할 수 있다. 이는 더 나아가 objective function의 gradient들을 network layer들로 back-propagating하는 것을 통해 network parameter들을 fine-tune하기 위해 사용될 수 있다.

## Supervised data에 대한 공식

- supervised sample들에 대한 loss function을 공식화하기 위해 **DerainNet(deep convolutional neural network)**의 **network structure**와 **negative residual mapping skill(?)**을 따른다. $f_ω (ㆍ)$로 정의되는 DerainNet(w → network parameters)는 input image의 rain streaks를 제거하고 rain-free image를 출력한다고 가정한다.
- 일반적인 CNN의 loss function은 derain output으로 기대되는 $f_ω (x_i )$와 ground truth label $y_i$사이의 least square loss를 최소화 시킨다. 즉, supervised sample들에 부과되는 loss function은 다음과 같다.

![Untitled9](https://user-images.githubusercontent.com/86338750/162739925-8fa2edb0-d9b4-4244-b013-e60966f63a63.png)

## KL-Divergence

- GMM은 모든 continuous distribution에 적용할 수 있기 때문에, **real rain sample들에 더 잘 맞도록 하기 위해서 Synthesized rain data 영역과 real rain data 영역 사이의 불일치가 너무 크지 않도록** 합성 비에서 학습한 훈련 중 small controlling parameter를 가지고 실제 비에서 학습한 **Gaussian들의 Kullback-Leibler divergence를 최소화 시킴으로써 제약 조건을 추가**한다.
- 이 제약 조건을 추가함으로써 논문의 모델이 합성된 비에서 임의의 영역이 아닌 실제 비로 transfer될 것으로 예상하고 있다.
- 이 KL-Divergence는 분석적으로 다루기 어렵기 때문에 각 구성 요소 사이의 KL-Divergence의 최소 값을 실제 샘플에서 학습한 GMM의 하나 이상의 구성 요소가 비와 유사하도록 경험적이고 간단한 대체물로 사용한다.

![Untitled10](https://user-images.githubusercontent.com/86338750/162739932-128f4862-70c4-46f4-b7b0-b9cd34857d5f.png)

## Total Variation Regularizer

- real rain image가 network를 통과해 예측된 이미지(clean image라고 예측)에서 혹시 남아있을지 모르는 빗줄기를 제거하기 위해서, 이미지를 약간 부드럽게 만들어준다.

## 전체 Objective function

- MAP model(likelihood + regularizer)은 unsupervised rainy image의 예상 network 출력에 대해 공식화한다.

  - 이는 해당 clean image에 대한 특정 명시적 지침 없이도 unsupervised data의 network training에 대한 gradient descent에 대해 올바른 방향으로 갈 수 있도록 해준다.

![Untitled11](https://user-images.githubusercontent.com/86338750/162739938-32435aa4-5f88-4927-bc60-49a972b93132.png)

- 이 objective function의 마지막 항을 통해, unsupervised data는 supervised data에 부과된 같은 network에 공급될 수 있다.

- α, β, λ가 0이면 논문의 model은 원래의 supervised deep learning model이 된다.

- 이런 objective 설정을 사용함으로써, network는 annotated supervised data에 잘 훈련될 수 있을 뿐만 아니라 rain streaks distribution의 사전 정보를 완전히 encoding하여  unsupervised input들도 잘 훈련될 수 있다.
