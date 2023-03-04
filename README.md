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

# Label Data 확인

![image](https://user-images.githubusercontent.com/104436260/222884579-1ccd424f-a5ea-40f4-b3f4-4c4340896761.png)



