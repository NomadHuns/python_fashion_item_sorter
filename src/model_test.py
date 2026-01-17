import tensorflow as tf
import matplotlib.pyplot as plt

# 1. 데이터셋 로드
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 2. 데이터 정규화 (0~255 사이의 픽셀 값을 0~1 사이로 변환)
train_images = train_images / 255.0
test_images = test_images / 255.0

# 데이터 확인용 레이블 리스트
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model = tf.keras.Sequential([
    # 이미지의 특징을 추출하는 부분 (Feature Extraction)
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # 추출된 특징을 1차원으로 펼쳐 분류하는 부분 (Classification)
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax') # 10개 카테고리 분류
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 3. 모델 학습 (epochs는 반복 횟수입니다)
history = model.fit(train_images, train_labels, epochs=10, validation_split=0.2)

# 4. 테스트 데이터로 성능 평가
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\n테스트 정확도: {test_acc:.4f}')

plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()