import tensorflow as tf
import matplotlib.pyplot as plt

def make_model():
    model = tf.keras.Sequential([
        # 1. 첫 번째 특징 추출 블록
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # 2. 두 번째 특징 추출 블록
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # 3. 추가된 세 번째 특징 추출 블록 (New!)
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        # 이미지 크기가 이미 작아졌으므로 여기서는 MaxPooling을 생략하거나 신중히 결정합니다.

        # 4. 분류 부분
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.summary()

    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    return model

def fit_model(model, train_images, train_labels, epochs):
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)

    return model.fit(train_images, train_labels, epochs=epochs, validation_split=0.2, callbacks=[early_stop])

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