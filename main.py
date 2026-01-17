from src.datasets.fashion_dataset import fashion_dataset
from src.models.fashion_item_sorter_model import make_model, fit_model, evaluate_model
from src.normalizator.normalizator import image_normalize

# 1. 데이터셋 로드
(train_images, train_labels), (test_images, test_labels) = fashion_dataset()

# 2. 데이터 정규화 (0~255 사이의 픽셀 값을 0~1 사이로 변환)
train_images = image_normalize(train_images)
test_images = image_normalize(test_images)

# 데이터 확인용 레이블 리스트
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model = make_model()

# 3. 모델 학습 (epochs는 반복 횟수입니다)
history = fit_model(model, train_images, train_labels, 20)

# 4. 테스트 데이터로 성능 평가
evaluate_model(model, test_images, test_labels, history)