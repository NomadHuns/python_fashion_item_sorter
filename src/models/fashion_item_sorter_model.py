import tensorflow as tf
import matplotlib.pyplot as plt

def make_model():
    # 1. 데이터 증강 레이어 정의
    data_augmentation = tf.keras.Sequential([
        # 좌우 반전 (옷의 경우 좌우가 바뀌어도 같은 아이템입니다)
        tf.keras.layers.RandomFlip("horizontal"),
        # 이미지 회전 (최대 10% 내외로 살짝 비틀기)
        tf.keras.layers.RandomRotation(0.1),
        # 이미지 확대/축소 (최대 10% 내외)
        tf.keras.layers.RandomZoom(0.1),
    ])

    model = tf.keras.Sequential([
        # 데이터 증강 레이어를 처음에 배치
        data_augmentation,

        # 이미지의 특징을 추출하는 부분 (Feature Extraction)
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # 추출된 특징을 1차원으로 펼쳐 분류하는 부분 (Classification)
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2), # 20%의 뉴런을 무작위로 쉬게 함
        tf.keras.layers.Dense(10, activation='softmax') # 10개 카테고리 분류
    ])

    model.summary()

    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    return model

def fit_model(model, train_images, train_labels, epochs):
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        start_from_epoch=10, # 초기 10회는 무슨 일이 있어도 멈추지 않고 학습함
        min_delta=0.0001, # 아주 미세한 향상만 있어도 '참기' 횟수를 초기화함
        restore_best_weights=True, # 멈춘 뒤 가장 성적이 좋았던 때로 복구
        verbose=1 # 언제 멈췄는지 알려준다
    )

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