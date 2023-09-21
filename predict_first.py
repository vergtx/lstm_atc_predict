import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import joblib

# 데이터 설정을 위한 클래스 (학습 코드에서 사용한 것과 동일)
class DataConfig:
    def __init__(self):
        self.features = ['workcode', 'state', 'block1', 'bay1', 'row1', 'tier1', 'crane']
        self.target1 = 'bay2'
        self.target2 = 'row2'
        self.target3 = 'tier2'

# 객체 생성
config = DataConfig()

# 예측을 위한 새로운 데이터 불러오기
file_path_on_6th = 'tos_data_ver6_on_6th.csv'  # 수정 필요
df_on_6th = pd.read_csv(file_path_on_6th)

# 저장된 모델 불러오기
model = load_model('best_model.h5')

# LabelEncoder 불러오기
label_encoders = {}
for column in config.features:
    if df_on_6th[column].dtype == 'object':
        label_encoders[column] = joblib.load(f"{column}_label_encoder.pkl")  # 저장된 LabelEncoder 불러오기
        df_on_6th[column] = label_encoders[column].transform(df_on_6th[column])  # 변환 적용

# 데이터 시퀀스화
sequence_lengths = 50  # 학습 코드와 동일하게 설정
stride = sequence_lengths // 5  # 학습 코드와 동일하게 설정
X_on_6th = df_on_6th[config.features].values
X_sequences_on_6th = []
for i in range(0, len(df_on_6th) - sequence_lengths + 1):
    X_sequences_on_6th.append(X_on_6th[i:i + sequence_lengths])
X_sequences_on_6th = np.array(X_sequences_on_6th)

# 예측 수행
predictions = model.predict(X_sequences_on_6th)

# 예측 결과를 DataFrame에 추가하는 부분
pred_bay = np.argmax(predictions[0], axis=2).flatten() +1
pred_row = np.argmax(predictions[1], axis=2).flatten()
pred_tier = np.argmax(predictions[2], axis=2).flatten()

# 새로운 DataFrame을 생성하여 문제를 해결
new_df_on_6th = df_on_6th.copy()

min_length = min(len(pred_bay), len(new_df_on_6th))

# loc를 사용하여 명시적으로 인덱스를 설정
new_df_on_6th.loc[:min_length-1, 'pred_bay'] = pred_bay[:min_length] 
new_df_on_6th.loc[:min_length-1, 'pred_row'] = pred_row[:min_length]
new_df_on_6th.loc[:min_length-1, 'pred_tier'] = pred_tier[:min_length]

# 새로운 CSV 파일로 저장
new_df_on_6th.to_csv('tos_data_with_predictions.csv', index=False)