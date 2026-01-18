import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 1. 모델 및 레이블 정의
def run_inference(image_path, model_path='my_fashion_model.keras'):
    # 저장된 모델 불러오기
    model = tf.keras.models.load_model(model_path)

    # 클래스 이름 정의 (학습 때와 동일한 순서)
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # 2. 이미지 불러오기 및 전처리
    # - 모델이 배운 것과 똑같이 만들어야 합니다.
    img = Image.open(image_path).convert('L') # 흑백(Grayscale) 변환
    img = img.resize((28, 28))                # 28x28 크기 조정

    img_array = np.array(img)

    # 중요: Fashion MNIST는 배경이 검은색(0), 물체가 흰색(255)입니다.
    # 만약 넣으려는 사진이 흰 배경에 검은 옷이라면 색상을 반전시켜야 할 수도 있습니다.
    # img_array = 255 - img_array

    img_array = img_array / 255.0             # 정규화 (0~1)
    img_array = img_array.reshape(1, 28, 28, 1) # 모델 입력 차원 (Batch, H, W, C) 맞추기

    # 3. 예측 실행
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0]) # 확률로 변환

    result_index = np.argmax(predictions[0])
    result_name = class_names[result_index]
    confidence = 100 * np.max(predictions[0])

    print(f"예측 결과: {result_name} ({confidence:.2f}%)")

    # 결과 시각화
    plt.imshow(img, cmap='gray')
    plt.title(f"{result_name} ({confidence:.2f}%)")
    plt.axis('off')
    plt.show()

# 사용 예시
run_inference('images/test_bag_02.jpeg')