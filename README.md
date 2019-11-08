# Implementation of CNN Classifier

## Introduction

지난 프로젝트에서는, k-NN classifier 와 SVM 을 이용한 linear classifier 를 파이썬 코드로 구현하여 CIFAR-10 dataset 에 적용하고 그 결과를 분석하였다. 결과 k-NN classifier 는 image 와 같은 고차원 데이터를 분류하는 분류기로는 적절하지 않았고, SVM classifier 의 경우는 image 의 회전과 확대/축소와 같은 변화에 항상성(invariance)을 가지지 못하여 계산된 거리와 실제 인지적 거리가 차이가 있었다.

이번 프로젝트에서는 이미지 정보 처리에 있어서 가장 유리하다고 알려진 Convolutional neural network(CNN) 을 기반으로 하는 다양한 CNN classifier 들을 바탕으로 좀 더 최적화된 형태로 변형해가는 과정을 여러 주목받은 리서치들과 함께 따라가 볼 것이다. 먼저, CNN 이 처음 소개된 논문 [1] 에서 시작하여, 동 저자의 LeNet5 [2], ILSVRC 의 AlexNet [3], Zeiler 의 ZFNet 및 Visualizing [4], GoogLeNet [5], VGGNet [6], ResNet [7], DenseNet [8] 등의 주요 아키텍처들을 중심으로 네트워크의 구조가 어떻게 변화해왔고, 변화해가고 있는지를 알아볼 것이다.

또한, Nvidia 의 GPU-accelerated 라이브러리인 CUDA Deep Neural Network library(cuDNN) 를 적용하여 제한이 있었던 환경을 개선하였고, 오픈 소스 신경망 라이브러리인 Keras 를 활용하여 좀 더 직관적이고 간결한 코드로 구현하였다. </br></br>

## Devices and Programs

![table1](/table1.PNG)
<br/>
Batch-size 의 경우 tensorflow-gpu 사용시 사용된 GPU 의 메모리에 크게 영향을 받는다. 위 디바이스 환경에서 batch-size 가 128인 경우 OOM(Out of Memory) Error 가 간헐적으로 발생하여 모든 코드에서 batch-size 를 64로 설정하였다. 나중에 또 언급할 것이지만 batch-size 는 학습속도에 크게 영향을 주는 파라미터이므로 자신의 디바이스 환경을 잘 알고 허용되는 범위에서 설정해주어야 한다.

![table2](/table2.PNG)
<br/>

모든 코드는 파이썬 3을 기준으로 작성되었고, 파이썬 2 환경에서 실행하면 오류가 발생할 수 있다. 만약 2에서 실행하여야 한다면 __future__ 를 이용하면 된다.

## CIFAR-10

학습에 사용될 데이터는 어떻게 보면 학습 그 자체보다 훨씬 중요한 존재라고 볼 수 있다. 아무리 잘 최적화된 아키텍처라 할지라도 제공된 데이터의 양과 질이 좋지 않다면 학습의 성능이 크게 떨어지기 때문이다. 따라서 학습의 방향과 구조를 정하기에 앞서 앞으로 다룰 데이터셋의 성격을 정확하게 파악하는 것이 가장 중요하다.

CIFAR-10 dataset 은 10개의 클래스와 각 클래스별로 각각 5000개의 train data 와 1000개의 test data 를 포함하는 image dataset 이다. 모든 이미지는 32 x 32 x 3 사이즈로 규격화되어 있다. 여기서 생각해볼 수 있는 것은 과연 5000개의 이미지가 하나의 클래스의 특징들을 유효하게 추출하는데 충분한 양인가이다. 이것은 그 학습의 한계를 결정짓는 아주 중요한 요소이다. 만약 5000개로는 부족할 때 두가지 상황이 발생할 수 있는데 첫번째는, 구현한 아키텍처는 크지만 데이터의 양이 그에 따라주지 못하여 overfitting 이 일어나는 경우이고 두번째는, 구현한 아키텍처와 데이터의 양이 적절하게 맞아 떨어지지만 데이터의 양의 한계로 더 이상 추출될 수 있는 특징이 부족하여 최종적으로 학습은 특정 수렴점에서 더 이상 분류 성능이 올라가지 않는 것이다. 또한, 이미지들의 성격이 특정 방향으로 치우쳐져 있는 경우라면 해당 클래스의 특징을 정확하게 추출하는데 어려움이 있을 것이다. 나중에 확인할 것이지만 CIFAR-10 dataset 은 연구 목적으로 수집된 데이터들이기 때문에 일반적인 경우보다 데이터들의 질과 양이 더 좋긴 하지만 그럼에도 부족하다는 것을 알 수 있다.

이러한 training data 의 한계를 극복하기 위해 Data augmentation 을 수행할 수 있다. 데이터를 증강시키기 위해 여러가지 방법을 쓸 수 있는데 AlexNet 에서는 256 x 256 size images 로부터 224 x 224 patches 를 무작위로 선택하여(crops) 2048배 늘어난 training set 을 획득하였고, 거기에 추가로 PCA 를 수행하여 RGB 채널을 조작하였다. GoogLeNet 은 이미지의 비율을 일정 범위 내에서 유지하면서 다양한 크기의 patch 들을 이용하고, photometric distortion of Andrew Howard [9] 을 사용하여 학습 데이터를 늘렸다. VGGNet 의 경우에는 두가지 crops 을 사용하였는데 첫번째는 single scale 에서이고, 두번째는 multi scale 에서이다. Single scale 에서는 AlexNet, ZFNet 과 마찬가지로 size 가 256 인 경우와 추가로 384 인 경우로 rescale 해서 patches 를 선택하였다. Multi scale 에서는 256~512 사이의 size 에서 무작위로 선택하여 다양한 크기에 대한 대응이 가능해져 정확도가 올라갔다. 이때 속도를 올리기 위해 384 size 에서 미리 학습 시킨 후 무작위로 선택하여 fine tuning 하는 scale jittering 을 수행하였다.

이렇게 data 를 처리하여 overfitting 과 같은 문제점을 개선하려는 시도는 training 뿐만 아니라 testing 시에도 나타난다. 일반적으로 쓰이는 방법은 test images 에 대해 training 과 마찬가지로 여러장의 crop images 를 생성한 후 testing 시에 평균 혹은 voting 방식으로 클래스를 결정한다.

이 프로젝트에서는 먼저 data augmentation 과정 없이 학습을 수행한 후 발생하는 문제점들을 확인하고, 다시 data augmentation 과정을 포함하여 학습을 수행하였다. 기본적인 과정은 Keras 의 ImageDataGenerator 모듈을 이용하였고, rotation, shift, flip 등의 기법을 사용하였다. 비교 결과는 뒤에서 확인 할 수 있다.

## Architecture

## Training

## Result
