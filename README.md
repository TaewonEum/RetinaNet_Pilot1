# RetinaNet

# 2023.03~05사이에 업로드 예정

실제로 촬영한 벌의 사진에서 식별가능한 응애를 직접 라벨링작업을 한 후 yolov8 pytorch 좌표 버전으로 export한 이미지로 RetinaNet Test 적용 진행

벌의 질병인 응애 Sample사진의 Detection Test

# device 설정 및 Library import

![image](https://user-images.githubusercontent.com/104436260/225211900-c4023a04-1e10-4e01-bd85-4b8426924089.png)

현재 코드에서는 dataset이 있는 경로를 os.chdir로 아예 고정시켜버림

# GPU 가동 확인

![image](https://user-images.githubusercontent.com/104436260/222882217-2b3ecb8b-327b-44c8-a781-e0f443a35b4e.png)

GPU(CUDA) 사용 확인

# Data 경로 설정

![image](https://user-images.githubusercontent.com/104436260/222883565-78bc3318-717f-419d-86c0-5f4de02ac43e.png)

# Data 갯수 확인

![image](https://user-images.githubusercontent.com/104436260/222883797-e5de9ffa-21ba-4005-9fb1-bbf3fa9ce760.png)

Train, Valid, Test 8:1:1 비율로 Split하여 저장

# Class명 담고 있는 리스트 생성 및 파일명 리스트화

![image](https://user-images.githubusercontent.com/104436260/224184604-cf6bb88b-bde2-4b77-a680-682e77b4a832.png)


# Image Data shape확인, Label Data 확인

![image](https://user-images.githubusercontent.com/104436260/224185506-05e205cb-ec65-410d-81be-75bd3db64fa1.png)

Label 값은: Class, x, y, Width, Height 순으로 이루어짐

RetinaNet을 PyTorch에서 사용하기 위해서는 바운딩 박스 좌표값이 [x_min, y_min, x_max, y_max] 형태의 tensor로 입력되어야 함.

만약 여러 개의 바운딩 박스를 가지고 있는 경우에는 이들을 리스트 또는 torch.Tensor의 2D 배열로 만들어서 구성해야함.

# 기존 좌표값 변환 및 바운딩 박스 확인

![image](https://user-images.githubusercontent.com/104436260/222886691-03ab6bae-0f88-4023-a075-ab3d4a8d4fa9.png)

![image](https://user-images.githubusercontent.com/104436260/222886858-5cef7348-b8a2-45f9-80f8-3592b0434e28.png)

img_width와 img_height를 통해 xmin, ymin, xmax, ymax값 구하여 바운딩박스 좌표 확인

# 새로운 라벨링 리스트 만들기

기존의 라벨링 파일의 값을 변환시켜 RetinaNet input에 맞게 데이터 변형

아래는 xmin, ymin, xmax, ymax 값으로 변환 후 리스트에 저장해주는 코드임

![image](https://user-images.githubusercontent.com/104436260/224187941-fca5c2ea-d209-4055-a2a8-164aaa04f30c.png)

4개의 좌표값을 한 리스트에 저장->이미지 별 객체 갯수만큼 각각 리스트화 해줌

아래는 라벨값을 변환해주는 코드임

![image](https://user-images.githubusercontent.com/104436260/224188742-ec939093-994a-4047-9b3c-9deb1d46303b.png)

라벨링 파일에서 좌표값을 제외한 class값만 추출하여 리스트화 함

최종적으로 아래의 코드를 통해 좌표값과 라벨값을 합쳐주는 딕셔너리를 만들어주는 코드 생성

![image](https://user-images.githubusercontent.com/104436260/224189658-a64dce8d-f29d-455c-8729-7ee4333960fd.png)

![image](https://user-images.githubusercontent.com/104436260/224190852-64b2515f-9c5d-403e-bea8-fff4077e9e51.png)

Pytorch 공식 사이트에서 나온 RetinaNet input 데이터 형태로 데이터를 변형함

![image](https://user-images.githubusercontent.com/104436260/225215045-45db453a-546f-4ede-9835-7f721a966ae5.png)

새로 만든 데이터 셋으로도 객체의 바운딩박스 위치에는 이상이 없음을 확인함

# 커스텀 데이터셋 만들기

Dataset 상속받아 __init__, __len__, __getitem__ 메소드를 사용하여 커스텀 데이터셋을 구성함

![image](https://user-images.githubusercontent.com/104436260/224193375-878d33ba-3a04-4f3f-b615-5c86b97bd8b0.png)

transform 함수까지 작성->이미지 리사이즈, 이미지 텐서변환만 진행함 augmentation은 진행하지 않음

![image](https://user-images.githubusercontent.com/104436260/224196008-cd6a30a7-7ec6-42de-ab70-78978887f89f.png)


Dataset은 각각의 이미지 텐서변환 데이터와 {[좌표값], [라벨값]}으로 이루어짐

![image](https://user-images.githubusercontent.com/104436260/224194834-d3f5a7e9-b879-4be9-a3ee-fda387294faf.png)

Pytorch 공식 튜토리얼상의 설명내용과 같이 Input data 구성 완료

# 커스텀 데이터셋 이미지 및 좌표값 확인

![image](https://user-images.githubusercontent.com/104436260/224194968-3090cb0b-fb3c-4e87-ac85-c0606eeba967.png)

텐서로 변환한 후에도 객체에 대한 좌표값 확인

# DataLoader 정의

PyTorch는 데이터를 다루기 위해 Dataset과 DataLoader라는 두 가지 개념을 제공함.

Dataset은 데이터셋을 추상화한 클래스로, PyTorch에서 사용하는 모든 데이터셋은 Dataset 클래스를 상속하여 만들어집니다. 이 클래스는 데이터셋에 대한 추상화 인터페이스를 제공하며, 각 데이터 포인트를 인덱스로 접근할 수 있도록 구현

DataLoader는 Dataset을 더욱 효율적으로 다루기 위한 유틸리티 클래스입니다. DataLoader는 배치 크기, 셔플 여부, 데이터 로딩을 병렬로 처리할지 여부 등 다양한 인자를 설정할 수 있습니다. DataLoader는 Dataset을 받아서 데이터를 배치 단위로 묶어주는 역할

따라서, Dataset 클래스는 데이터셋을 추상화하여 데이터 포인트를 하나씩 다루는 인터페이스를 제공하고, DataLoader 클래스는 Dataset을 받아서 배치를 만들고 데이터를 더욱 효율적으로 다루기 위한 인터페이스를 제공

![image](https://user-images.githubusercontent.com/104436260/224198276-7bf2b05b-b8e7-401a-9cc5-b562a4f9b027.png)

collate_fn이란 Dataset을 batch단위로 묶을 때 사용함 이미지 마다 들어있는 객체의 수가 다르기 때문에 데이터는 일반적으로 다른 사이즈를 가지고 있음. 데이터를 배치로 묶을 때는 이를 동일한 크기로 맞춰주어야 함 이를 위해 collate_fn은 각각의 데이터를 처리하여 배치 단위로 묶을 수 있도록 해줌

여기선 배치 단위로 각각의 이미지와 이미지의 타겟값을 Tuple형태로 묶어줌

# Train, Test 전체 객체개수 확인

![image](https://user-images.githubusercontent.com/104436260/224199161-941ff38e-5e94-4778-ac15-e8a46c0f73ab.png)

RetinaNet Define

![image](https://user-images.githubusercontent.com/104436260/224201037-41be063b-a717-48a5-975c-b665a21dcd3e.png)

현재 사업의 샘플 데이터의 Detection여부를 알아보기 위함이기 때문에 전이학습 진행, backbone 가중치는 사용함.

# parameter Settings

![image](https://user-images.githubusercontent.com/104436260/224203418-0dd20548-1d85-44ea-befa-05bdd300cf2b.png)

# 학습결과 확인 epoch=1으로 설정(결과 확인용)

![image](https://user-images.githubusercontent.com/104436260/224205277-df6ae0cb-55f0-4483-8943-75fcd4743d84.png)

# RetinaNet 모델 학습을 수행 Process

먼저, GPU를 사용하기 위하여 image tensor데이터와 targets의 box좌표, label값을 모두 GPU메모리로 이동시켜야 함.

RetinaNet Model에 이미지와 타겟값을 input하면 output결과로 score가 나오게 됨.

먼저 loss_dict를 print문을 통해 출력해보면

결과는 batch당 classification loss score값과, bbox_regression score가 출력됨

두개 모두 loss값 즉 실제 바운딩박스 값과 예측 바운딩박스 값의 차이라고 보면되는 데 두개의 loss값은 차이가 존재함

# Loss값

먼저 Loss값이란 모델의 예측과 실제 값 사이의 차이를 나타내는 함수임.

RetinaNet에서는 하나의 loss함수로 모든 예측값을 처리하는것이 아니라 두개의 loss함수를 사용하여 예측값을 처리함

이는 RetinaNet이 클래스 불균형성 문제를 해결하기 위한 방법임

먼저 classification loss는 객체 검출 모델에서는 각 bounding box에 대해 해당 객체가 있는지 없는지 예측하는 이진분류를 수행함. 즉 이 loss는 객체가 있는 위치에 대한 예측이 정확한지를 판단하는 데 사용됨.

두번째는 regression loss로 bounding box의 위치와 크기를 예측하는 값임. 이 loss는 bounding box의 위치와 크기 예측이 정확한지를 판단하는데 사용됨

# RetinaNet의 Focal Loss
