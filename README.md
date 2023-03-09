# RetinaNet

# 2023.03~05사이에 업로드 예정

실제로 촬영한 벌의 사진에서 식별가능한 응애를 직접 라벨링작업을 한 후 yolov8 pytorch 좌표 버전으로 export한 이미지로 RetinaNet Test 적용 진행

벌의 질병인 응애 Sample사진의 Detection Test

# device 설정 및 Library import

![image](https://user-images.githubusercontent.com/104436260/222882063-3e6b8be3-557d-4196-8c8d-5bffde780d85.png)

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

![image](https://user-images.githubusercontent.com/104436260/224185977-df98f00f-ec75-41aa-abfc-bc3330b43cec.png)






