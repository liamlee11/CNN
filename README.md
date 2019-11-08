# Implementation of CNN Classifier

## Introduction

지난 프로젝트에서는, k-NN classifier 와 SVM 을 이용한 linear classifier 를 파이썬 코드로 구현하여 CIFAR-10 dataset 에 적용하고 그 결과를 분석하였다. 결과 k-NN classifier 는 image 와 같은 고차원 데이터를 분류하는 분류기로는 적절하지 않았고, SVM classifier 의 경우는 image 의 회전과 확대/축소와 같은 변화에 항상성(invariance)을 가지지 못하여 계산된 거리와 실제 인지적 거리가 차이가 있었다.

이번 프로젝트에서는 이미지 정보 처리에 있어서 가장 유리하다고 알려진 Convolutional neural network(CNN) 을 기반으로 하는 다양한 CNN classifier 들을 바탕으로 좀 더 최적화된 형태로 변형해가는 과정을 여러 주목받은 리서치들과 함께 따라가 볼 것이다. 먼저, CNN 이 처음 소개된 논문 [1] 에서 시작하여, 동 저자의 LeNet5 [2], ILSVRC 의 AlexNet [3], Zeiler 의 ZFNet 및 Visualizing [4], GoogLeNet [5], VGGNet [6], ResNet [7], DenseNet [8] 등의 주요 아키텍처들을 중심으로 네트워크의 구조가 어떻게 변화해왔고, 변화해가고 있는지를 알아볼 것이다.

또한, Nvidia 의 GPU-accelerated 라이브러리인 CUDA Deep Neural Network library(cuDNN) 를 적용하여 제한이 있었던 환경을 개선하였고, 오픈 소스 신경망 라이브러리인 Keras 를 활용하여 좀 더 직관적이고 간결한 코드로 구현하였다. </br></br>

## Devices and Programs

![table1](/table1.PNG)
<br/>
Batch-size 의 경우 tensorflow-gpu 사용시 사용된 GPU 의 메모리에 크게 영향을 받는다. 위 디바이스 환경에서 batch-size 가 128인 경우 OOM(Out of Memory) Error 가 간헐적으로 발생하여 모든 코드에서 batch-size 를 64로 설정하였다. 나중에 또 언급할 것이지만 batch-size 는 학습속도에 크게 영향을 주는 파라미터이므로 자신의 디바이스 환경을 잘 알고 허용되는 범위에서 설정해주어야 한다.
