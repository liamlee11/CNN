# Implementation of CNN Classifier

## Introduction

지난 프로젝트에서는, k-NN classifier 와 SVM 을 이용한 linear classifier 를 파이썬 코드로 구현하여 CIFAR-10 dataset 에 적용하고 그 결과를 분석하였다. 결과 k-NN classifier 는 image 와 같은 고차원 데이터를 분류하는 분류기로는 적절하지 않았고, SVM classifier 의 경우는 image 의 회전과 확대/축소와 같은 변화에 항상성(invariance)을 가지지 못하여 계산된 거리와 실제 인지적 거리가 차이가 있었다.

이번 프로젝트에서는 이미지 정보 처리에 있어서 가장 유리하다고 알려진 Convolutional neural network(CNN) 을 기반으로 하는 다양한 CNN classifier 들을 바탕으로 좀 더 최적화된 형태로 변형해가는 과정을 여러 주목받은 리서치들과 함께 따라가 볼 것이다. 먼저, CNN 이 처음 소개된 [1] 에서 시작하여, 동 저자의 LeNet5 [2], ILSVRC 의 AlexNet [3], Zeiler 의 ZFNet 및 Visualizing [4], GoogLeNet [5], VGGNet [6], ResNet [7], DenseNet [8] 등의 주요 아키텍처들을 중심으로 네트워크의 구조가 어떻게 변화해왔고, 변화해가고 있는지를 알아볼 것이다.

또한, Nvidia 의 GPU-accelerated 라이브러리인 CUDA Deep Neural Network library(cuDNN) 를 적용하여 제한이 있었던 환경을 개선하였고, 오픈 소스 신경망 라이브러리인 Keras 를 활용하여 좀 더 직관적이고 간결한 코드로 구현하였다. </br></br>

## Devices and Programs

![table1](/table1.PNG)
<br/>
Batch-size 의 경우 tensorflow-gpu 사용시 사용된 GPU 의 메모리에 크게 영향을 받는다. 위 디바이스 환경에서 batch-size 가 128인 경우 OOM(Out of Memory) Error 가 간헐적으로 발생하여 모든 코드에서 batch-size 를 64로 설정하였다. batch-size 는 학습속도에 크게 영향을 주는 파라미터이므로 자신의 디바이스 환경을 잘 알고 허용되는 범위에서 설정해주어야 한다.

![table2](/table2.PNG)
<br/>

모든 코드는 파이썬 3을 기준으로 작성되었고, 파이썬 2 환경에서 실행하면 오류가 발생할 수 있다. 만약 2에서 실행하여야 한다면 __future__ 를 이용하면 된다.

## CIFAR-10

학습에 사용될 데이터는 학습 그 자체만큼 중요한 존재이다. 잘 최적화된 아키텍처라 할지라도 제공된 데이터의 양과 질이 좋지 않다면 학습의 성능이 떨어질 수 있다. 따라서 학습의 방향과 구조를 정하기에 앞서 앞으로 다룰 데이터셋의 성격을 정확하게 파악하는 것이 중요하다.

CIFAR-10 dataset 은 10개의 클래스와 각 클래스별로 각각 5000개의 train data 와 1000개의 test data 를 포함하는 image dataset 이다. 모든 이미지는 32 x 32 x 3 사이즈로 규격화되어 있다. 여기서 생각해볼 수 있는 것은 과연 5000개의 이미지가 하나의 클래스의 특징들을 유효하게 추출하는데 충분한 양인가이다. 이것은 그 학습의 한계를 결정짓는 아주 중요한 요소이다. 만약 부족하다면 두가지 상황이 발생할 수 있는데 첫번째는, 구현한 아키텍처는 크지만 데이터의 양이 그에 따라주지 못하여 overfitting 이 일어나는 경우이고 두번째는, 구현한 아키텍처와 데이터의 양이 적절하게 맞아 떨어지지만 데이터의 양의 한계로 더 이상 추출될 수 있는 특징이 부족하여 최종적으로 학습은 특정 수렴점에서 더 이상 분류 성능이 올라가지 않는 것이다. 또한, 이미지들의 성격이 특정 방향으로 치우쳐져 있는 경우라면 해당 클래스의 특징을 정확하게 추출하는데 어려움이 있을 것이다. CIFAR-10 dataset 은 연구 목적으로 수집된 데이터들이기 때문에 일반적인 경우보다 데이터들의 질과 양이 더 좋긴 하지만 그럼에도 부족하다.

이러한 training data 의 한계를 극복하기 위해 Data augmentation 을 수행할 수 있다. 데이터를 증강시키기 위해 여러가지 방법을 쓸 수 있는데 AlexNet 에서는 256 x 256 size images 로부터 224 x 224 patches 를 무작위로 선택하여(crops) 2048배 늘어난 training set 을 획득하였고, 거기에 추가로 PCA 를 수행하여 RGB 채널을 조작하였다. GoogLeNet 은 이미지의 비율을 일정 범위 내에서 유지하면서 다양한 크기의 patch 들을 이용하고, photometric distortion of Andrew Howard [9] 을 사용하여 학습 데이터를 늘렸다. VGGNet 의 경우에는 두가지 crops 을 사용하였는데 첫번째는 single scale 에서이고, 두번째는 multi scale 에서이다. Single scale 에서는 AlexNet, ZFNet 과 마찬가지로 size 가 256 인 경우와 추가로 384 인 경우로 rescale 해서 patches 를 선택하였다. Multi scale 에서는 256~512 사이의 size 에서 무작위로 선택하여 다양한 크기에 대한 대응이 가능해져 정확도가 올라갔다. 이때 속도를 올리기 위해 384 size 에서 미리 학습 시킨 후 무작위로 선택하여 fine tuning 하는 scale jittering 을 수행하였다.

이렇게 data 를 처리하여 overfitting 과 같은 문제점을 개선하려는 시도는 training 뿐만 아니라 testing 시에도 나타난다. 일반적으로 쓰이는 방법은 test images 에 대해 training 과 마찬가지로 여러장의 crop images 를 생성한 후 testing 시에 평균 혹은 voting 방식으로 클래스를 결정한다.

이번 프로젝트에서는 먼저 data augmentation 과정 없이 학습을 수행한 후 발생하는 문제점들을 확인하고, 다시 data augmentation 과정을 포함하여 학습을 수행하였다. 기본적인 과정은 Keras 의 ImageDataGenerator 모듈을 이용하였고, rotation, shift, flip 등의 기법을 사용하였다. 비교 결과는 뒤에서 확인 할 수 있다.

## Architecture

여러 주목받은 CNN 아키텍처들을 따라 이번 프로젝트의 아키텍처를 만들어 나갔지만 모든 아키텍처들의 특징들을 포함시키려는 시도를 한 것은 아니며, 주로 kernel-size, layers 의 depth, dropout, FC layers 등이 아키텍처에 어떤 영향을 주는지에 중점을 두었다. 여기서 VGGNet 을 많이 참고하였다.

고전적인 CNN 구조인 LeNet5 는 총 3개의 conv layer, 2개의 sub-sampling layer 및 1개의 FC layer 로 구성되어 있다. 해당 논문에서 단계별 영상을 통해 CNN 이 topology shift, noise 에 robust 하다는 것을 밝혔고, 망의 규모가 최종 성능에 미치는 방향성을 보여주었다. 2012년의 AlexNet 은 5개의 conv layer 와 3개의 FC layer 로 구성되어 구조적으로는 LeNet5와 비슷하지만 더 깊은 망을 가지는 아키텍처를 가지게 되었다. AlexNet 에서는 GPU 를 병렬적으로 사용하였고, 여러 kernel-size 를 적용한 conv layer 와 activation function 으로 ReLU 를 사용한 것이 큰 특징이다. 여기까지 봤을 때 다음과 같은 상상을 할 수 있는데, CNN 망을 옆으로 누운 피라미드와 같은 형태로 계속 크기를 늘려간다면 언젠간 100%에 근접하는 성능을 보여주지 않을까 라는 것이다. 하지만 실제로는 아주 다양한 요인들에 의해 힘들다는 것이 실험적으로 밝혀졌다. 먼저 망의 규모가 커지면 자연적으로 파라미터의 수가 늘어난다. 이는 필연적으로 overfitting 을 발생시키게 되고 학습의 속도까지 저하시킨다. 또한, 망이 깊어질수록 gradient 가 옅어져서 vanishing gradient 문제가 발생한다.

어떻게 망의 깊이를 늘려 아키텍처의 잠재력을 올릴 수 있을까 고민한 것이 2014년의 GoogLeNet 과 VGGNet 이다. 또한, 이 프로젝트에서 구현할 아키텍처도 VGGNet 을 기반으로 시작할 것이기 때문에 앞서 설명한 아키텍처들과 어떠한 차이점이 있는지 알아볼 것이다.

먼저 VGGNet 의 구조는 다음과 같다. (CIFAR-10 이 아닌 image net 을 dataset 으로 함.) 

![table3](/table3.PNG)
<br/>
![table4](/table4.PNG)
<br/>
첫번째로 Table 3 의 D 를 기반으로 vgg_a 를 구현하였는데, VGGNet 에서는 이전의 아키텍처와 다르게 모든 kernel-size 를 3 x 3 size 이하로 하고 대신 여러 개의 conv layer 를 stack 하는 방식으로 하나의 블록을 형성하였다. 이것을 Factorization into smaller convolutions 이라고 하는데 이를 통해 파라미터의 수는 줄이고 망이 깊어지는 효과를 얻었다. 5 x 5 size 의 필터가 3 x 3 필터 두개가 stack 되는 형태로 대체되어 비용은 28% 줄이고 feature 를 뽑아내는 비선형성을 늘이는 효과를 얻는 것이다.

![figure1](/figure1.PNG)
<br/>
하지만 VGGNet 은 AlexNet 과 마찬가지로 마지막의 3개의 FC layer 때문에 파라미터의 개수가 비대하게 크다. 따라서 두번째로 테스트 해볼 아키텍처로 FC layer 을 제거한 vgg_b 를 선택하고 vgg_a 와 비교하여 FC layer 를 제거했을 때의 성능을 비교할 것이다. 세번째는 vgg_b 에서 dropout 을 추가한 vgg_c 형태를 추가로 선택하여 앞의 두 형태와 비교할 것이다. 마지막으로, 망의 깊이를 더 늘렸을 때 성능이 더 늘어나는지 확인하기 위해 vgg_d 를 추가하였다. 따라서 위 4개의 아키텍처의 구조를 표로 나타낸 결과는 다음과 같다.

![table5_1](/table5_1.PNG)
<br/>
![table5-2](/table5-2.PNG)
<br/>
vgg_a, b, c, d 의 train_acc 와 val_acc 의 결과는 Result 에서 확인 할 수 있다. vgg_c_nd 는 data augmentation 을 하지 않은 경우를 비교하기 위해 수행한 case 이다.

## Training

Training 에 사용된 methods 와 hyperparameters 는 우선 Keras 각 module 에서 제공되는 default 값을 적용시켰다. 모든 hyperparameter 들을 같은 값으로 fix 시킨 후 먼저 최적의 아키텍처를 선택하였고, hyperparameter tuning 은 아키텍처 선택 후에 csv_logger 을 통해 진행하였다.

Loss function 은 categorical_crossentropy 으로, optimizer 는 Adam [10] 을 선택하였다. 이때, Adam 의 초기 파라미터는 learning rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, decay = 0 으로 하고, 추가로 ReduceLROnPlateau 함수를 통해 학습 속도가 저하되었을 때 learning rate 를 감소시키는 부분을 코드에 추가하였다. 만약 3 epochs 동안 val_loss 의 향상이 없다면 learning rate 가 0.5배로 업데이트된다. 업데이트 후에는 2 epochs 만큼의 cooldown 이 있다. patience 값과 cooldown 값을 선택하는 것은 먼저 lr reduce 를 시행하지 않은 상태에서 어떻게 학습이 진행되는지 확인한 다음 patience 와 cooldown 값을 선택하였다.

Batch size 는 앞서 설명한 것처럼 OOM error 을 피하고 GPU memory 가 허용하는 최대한의 범위에서 설정하였고, 그 값은 64 이다. batch size 를 늘리면 GPU 를 사용하기 때문에 그만큼 학습속도가 빨라진다.

Kernel regularizer 는 L2 방식을 선택하였고, regularization strength 의 default 값인 0.01 을 초기값으로 했으나 학습이 크게 저하되고 update 에 문제가 생겨, 적정값인 1.e-4 로 수정하였다.

Data augmentation 은 width_shift, height_shift 범위를 각각 10% 로 주었고, 수평 flip 도 수행하도록 설정하였다.

모든 conv layer 의 ReLU function 전에 batch normalization 을 수행하였다. Batch normalization 을 수행하지 않으면 학습이 진행되지 않는 현상을 발견할 수 있었기 때문이다.

모든 training 과정은 csv_logger 함수를 이용하여 매 epoch 마다 변화하는 값들을 확인하였다. 이 값들을 통해 다양한 상태에서의 성능을 plot 하여 비교하였다.

## Result

![table6](/table6.PNG)
<br/>
![table7](/table7.PNG)
<br/>

Table 6 는 각 아키텍처에서 1 epoch 시 걸리는 시간을 측정한 것이다. Table 7 은 각 아키텍처에서 수렴하는 val_acc 를 측정한 것이다.

![figure23](/figure23.PNG)
<br/>
Figure 2 는 vgg_a model 에 대해 BN layer 가 있을 때와 없을 때의 학습 상태를 비교한 것이다. 그래프의 결과를 통해 batch normalization 의 필요성을 확인 할 수 있었고, 따라서 vgg_a, b, c, d 모든 models 에 대해 batch normalization 을 포함하였다. Figure 3 은 vgg_c model 에 대해 data augmentation 을 수행한 경우와 수행하지 않은 경우를 비교한 것이다. 수행하지 않은 경우는 수행한 경우보다 수렴이 시작하는 지점이 빨랐지만, overfitting 정도가 훨씬 심하고 val_acc 의 수렴점도 약 1.9% 정도 더 낮았다. 따라서 data augmentation 은 overfitting 을 줄일 수 있을 뿐만 아니라 성능 한계도 늘릴 수 있음을 알 수 있다.

![figure4](/figure4.PNG)
<br/>

Figure 4 는 Table 5 모델들에 대한 epoch 에 따른 train acc 와 val acc 의 결과를 그래프로 나타낸 것이다. 모든 결과에서 learning rate 가 감소하는 지점에서 acc jumping 이 관찰되고, 이는 학습이 진행되면서 lr 을 감소시키는 전략이 유효하게 작용한 것임을 알 수 있다. 학습이 수렴을 향한다는 뜻은 local minima 에 빠졌거나 더 깊은 convex 안으로 들어가지 못하는 것을 뜻하므로 만약 후자라면 learning rate 를 줄여 더 깊은 convex 로 진입하도록 유도할 수 있다.

먼저 vgg 아키텍처 맨 마지막 레이어인 3개의 FC layers 을 포함한 vgg_a model 과 vgg_b model 을 비교해보면(그래프의 빨간색과 파란색) 거의 차이가 없는 양상을 보이고 제거한 쪽의 최종 정확도가 약 0.3% 높은 것을 확인 할 수 있다.

vgg_a 와 vgg_b model 모두 overfitting 현상이 크게 발생됨을 확인 할 수 있는데 dropout 을 추가한 vgg_c model 은 overfitting 현상이 상대적으로 크게 줄었다. 여기서 주목할 점은 vgg_a 와 vgg_b model 은 모두 최종 train acc 가 1에 수렴하여 더 이상 val acc 가 향상될 여지가 없었던 반면 vgg_c model 은 train acc 가 94.6%에 수렴하여 계속 train 을 할 수 있을 거라고 예상 할 수 있다. 하지만 이는 두가지 상황을 생각해볼 수 있는데 학습이 local minima 에 빠졌거나 더 깊은 convex 에 들어가기에는 lr 값이 큰 것일 수 있다. 하지만 epoch 을 진행함에 따라 lr 을 줄여도 val acc 가 개선되지 않는 것으로 보아 local minima 에 빠진 것으로 보인다. 이 때는 lr 을 조금 더 키워서 local minima 의 convex 에서 벗어나야 한다. 또한 lr 을 줄이는 시점이 너무 빨랐던 것일 수도 있으므로 reduce_lr 함수의 patience 와 cooldown 을 늘려보는 시도도 하였다. 혹은 이렇게 학습된 파라미터들을 dropout 을 제거한 모델에 다시 적용시켜 계속 학습을 진행해볼 수도 있을 것이다. 이를 확인하기 위해 vgg_c 에 train 된 파라미터를 vgg_b model 에 적용시켜 계속 학습해 보았는데 최종 val acc 가 약 0.5% 상승하는 대신 overfitting 이 증가하였다. 다음으로 vgg_c model 이 a 와 b 에 비해 수렴 시점이 늦은 것을 확인 할 수 있는데 이는 dropout 에 의한 영향으로 보인다.

마지막으로 layer 의 depth 를 더 깊게 했을 때 성능이 어떻게 변하는지 비교하였다. vgg_d 는 vgg_c 에서 layer 를 3개 더 추가한 model 인데 오히려 overfitting 이 심해지고 수렴하는 정확도도 4.3% 낮았다. 이는 parameter 수가 늘어나 전체 model 의 capacity 가 data 에 비해 커져서 발생한 것이다.  수렴하는 정확도가 낮아진 것은 망이 깊어져서 vanishing gradient 문제가 심화되었기 때문이다.

![figure5](/figure5.PNG)
<br/>
Figure 5 은 vgg_c model 에 대해 batch size 를 64를 한 경우와 512를 한 경우를 비교한 것이다. 결과를 통해 학습 속도를 높이기 위해 batch size 를 늘린 것이 오히려 overfitting 발생하고 더 낮은 값에서 수렴하는 역효과를 낳은 것을 확인 할 수 있다. 따라서 batch size 를 적절하게 낮추어(최종적으로 batch size = 32 으로 설정하였음) overfitting 현상을 억제 할 수 있다.(batch normalization 에 의한 regularizer 효과로 보인다.)

위 결과들을 종합해 보면 FC layer 는 CIFAR-10 dataset 에 대해서는 제거해도 큰 변함이 없었다. 오히려 성능이 소폭 상승하였고 아키텍처를 간결화 시키고 파라미터 수를 줄이는 효과가 있었다. 이는 Network in network paper by Lin et al [11] 에서 망을 깊게 하여 모델의 추상화를 높이기 위하여 mlpconv layer 를 쌓는 구조를 만들고 마지막 레이어에 FC layer 를 쓰는 대신 global average pooling 층을 적용하여 FC layer 가 가지고 있던 overfitting 문제와 dropout 에 의존하고 있는 문제를 해결한 것과 함께 생각해볼 수 있다. GoogLeNet 의 inception module 또한 layer 수준에서 해석하여 얻은 결과이다. 다음으로 dropout 기법과 같은 regularization 과정이 overfitting 에 정말 큰 영향을 미치고, learning rate process 도 성능 향상에 큰 도움이 된다. Regularization 의 효과가 아키텍처에 너무 크게 들어가면 training 이 더 진행될 잠재력을 막을 수 있는데, 이 프로젝트에서는 vgg_c 의 dropout 의 값을 학습이 진행됨에 따라 조금씩 감소시키는 것으로 어느정도 성능을 향상시켰다.(100 epoch 을 지난 weight 값을 얻은 후 다시 dropout 값이 수정된 vgg_c 에 넣고 학습을 진행하였다.) Layer 의 depth 는 model 의 capacity 를 높일 수 있지만 overfitting 과 vanishing gradient 문제를 야기함을 확인하였다. ResNet 에서는 conv layer 을 통해 획득한 feature 와 filter 의 intensity 가 낮기 때문에 이것을 제외한 나머지 부분(residual)을 통해 학습을 하여 vanishing gradient 문제를 억제하는 구조를 고안하였다. Wide Residual Networks [12], Aggregated Residual Transformations for Deep Neural Networks [13] 에서는 residual block 의 cardinality 를 확장한 model 을 적용하였다. Identity Mappings in Deep Residual Networks [14] 은 residual block 의 shortcut 정보를 훼손시키지 않게 activation function 의 순서를 addition 앞으로 변형시킨 full pre-activation 과 같은 구조가 더 좋은 성능을 남겼다. 마지막으로 DenseNet 은 bypass 를 summation 하지 않고 다른 모든 지점에 forward 시켜 parameter 의 redundancy 를 줄였다.

![table8](/table8.PNG)
<br/>

## 참고문헌

[1] Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, and L. D. Jackel. Backpropagation applied to handwritten zip code recognition. Neural Comput., 1(4):541–551, Dec. 1989.<br/>
[2] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.<br/>
[3] A. Krizhevsky, I. Sutskever, and G. Hinton. Imagenet classification with deep convolutional neural networks. In Advances in Neural Information Processing Systems 25, pages 1106–1114, 2012. <br/>
[4] M. D. Zeiler and R. Fergus. Visualizing and understanding convolutional neural networks. In ECCV, 2014.<br/>
[5] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich. Going deeper with convolutions. In CVPR, 2015.<br/>
[6] K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. In ICLR, 2015.<br/>
[7] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. CoRR, abs/1512.03385, 2015.<br/>
[8] Gao Huang, Zhuang Liu and Laurens van der Maaten. Densely Connected Convolutional Networks. arXiv:1608.06993v5 [cs.CV] 28 Jan 2018 <br/>
[9] A. G. Howard. Some improvements on deep convolutional neural network based image classification. CoRR, abs/1312.5402, 2013.<br/>
[10] Diederik Kingma and Jimmy Ba. Adam: A Method for Stochastic Optimization. arXiv:1412.6980 [cs.LG]<br/>
[11] M. Lin, Q. Chen, and S. Yan. Network in network. CoRR, abs/1312.4400, 2013.<br/>
[12] S. Zagoruyko and N. Komodakis. Wide residual networks. In BMVC, 2016.<br/>
[13] Saining Xie, Ross Girshick, Piotr Doll´ar, Zhuowen Tu and Kaiming He. Aggregated Residual Transformations for Deep Neural Networks. arXiv:1611.05431v2 [cs.CV] 11 Apr 2017<br/>
[14] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.<br/>
