# RetinaNet 정리

RetinaNet은 Facebook AI Research에서 발표한 Object Detection 모델임.

Focal loss라는 새로운 방식의 Loss 계산을 도입하여 class imbalance 문제를 해결하고 높은 정확도를 보이는 것이 특징.

기존의 object detection 모델은 classification과 bounding box regression 두가지 문제를 동시에 해결하는 Two-stage 방식과 One-stage 방식으로 나뉘어짐

이 두 가지 방식 모두 클래스 불균형 문제로 인해 높은 정확도를 갖지 못하는 문제가 있음

RetinaNet은 이 문제를 Focal Loss를 도입하여 해결하려함.

Focal loss는 클래스 불균형 문제를 해결하고 정확도를 향상시키기 위해 사용됨


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

# RetinaNet Define

![image](https://user-images.githubusercontent.com/104436260/224201037-41be063b-a717-48a5-975c-b665a21dcd3e.png)

현재 사업의 샘플 데이터의 Detection여부를 알아보기 위함이기 때문에 전이학습 진행, backbone 가중치는 사용함.

# 파라미터 설정 및 모델 학습 진행

![image](https://user-images.githubusercontent.com/104436260/225219057-76b38fd7-8769-4121-beda-56d6c7938eb5.png)

optimizer.zero_grad= PyTorch에서 기울기(gradient)를 계산할 때, 기존의 기울기를 0으로 초기화합니다. 모델의 매개변수에 대한 기울기를 업데이트하기 전에, 기존의 기울기를 초기화

losses.backward()=델이 예측한 값과 정답 간의 차이(손실)를 계산한 후, 역전파(backward)를 통해 모델의 매개변수에 대한 기울기를 계산. 이 기울기는 loss.backward() 함수를 호출하는 시점에 각 매개변수에 대한 기울기가 계산

optimizer.step()=계산된 기울기를 이용하여 모델의 매개변수를 업데이트함. 즉, 이전의 매개변수 값에 학습률(learning rate)을 곱한 후에, 기울기를 더해주는 방식으로 매개변수를 업데이트.

# 모델 평가 진행

![image](https://user-images.githubusercontent.com/104436260/225220422-ea69c83e-a831-41d8-8b44-bc32d4cfe49e.png)

학습한 모델을 저장

![image](https://user-images.githubusercontent.com/104436260/225220510-bfb2f088-8602-453c-a231-c941699a543e.png)

모델 테스트 진행

![image](https://user-images.githubusercontent.com/104436260/225220601-d361584c-7de9-4286-bcc6-ed49f1e3e20c.png)

output 시각화 함수 작성

![image](https://user-images.githubusercontent.com/104436260/225220676-6a21acd4-41d0-4815-9dc3-e11410a67b7e.png)

![image](https://user-images.githubusercontent.com/104436260/225220864-378d4f01-b0ae-4258-966c-268c90515ae3.png)

output 결과 시각화

![image](https://user-images.githubusercontent.com/104436260/225220967-63ffad57-fa33-474d-87ca-47c5c7d4b287.png)

데이터 셋도 적고 사진에서 차지하는 응애의 비율이 매우적기 때문에 학습이 거의 안된듯 함.





