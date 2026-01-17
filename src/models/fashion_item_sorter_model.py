import tensorflow as tf
import matplotlib.pyplot as plt

def make_model():
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

    return model

def fit_model(model, train_images, train_labels, epochs):
    return model.fit(train_images, train_labels, epochs=epochs, validation_split=0.2)

def evaluate_model(model, test_images, test_labels, history):
    # 한글 폰트 설정 (Mac 전용)
    plt.rc('font', family='AppleGothic')
    plt.rcParams['axes.unicode_minus'] = False

    model.evaluate(test_images, test_labels, verbose=2)

    plt.plot(history.history['accuracy'], label='훈련 정확도')
    plt.plot(history.history['val_accuracy'], label = '검증 정확도')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.title('모델 정확도 추이')
    plt.show()