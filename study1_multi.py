# 필요한 라이브러리 임포트
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import joblib
# 데이터 설정을 위한 클래스
class DataConfig:
    def __init__(self):
        self.features = ['workcode', 'state', 'block1', 'bay1', 'row1', 'tier1', 'crane']
        self.target1 = 'bay2'
        self.target2 = 'row2'
        self.target3 = 'tier2'

# 객체 생성
config = DataConfig()

# 1. CSV 파일로부터 데이터 로드
file_path = 'tos_data_ver6_until_5th.csv'
df = pd.read_csv(file_path)

# 2. 특성과 타겟 설정 (객체 지향적 접근)
features = config.features
targets = [config.target1, config.target2, config.target3]

# 3. 숫자가 아닌 데이터에 대한 Label Encoding
label_encoders = {}
for column in features:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
        joblib.dump(le, f"{column}_label_encoder.pkl")  # 추가된 부분

# 원-핫 인코딩을 위한 OneHotEncoder 생성
onehot_encoder = OneHotEncoder(sparse_output=False)  

# 타겟 변수를 원-핫 인코딩 (수정된 부분)
Y1_onehot = onehot_encoder.fit_transform(df[[config.target1]].values)
Y2_onehot = onehot_encoder.fit_transform(df[[config.target2]].values)
Y3_onehot = onehot_encoder.fit_transform(df[[config.target3]].values)

# 원-핫 인코더 저장
joblib.dump(onehot_encoder, "onehot_encoder.pkl")

# 4. 데이터 시퀀스화
epoch = 500
sequence_lengths = 50
stride = sequence_lengths // 3
X = df[features].values
X_sequences = []
Y1_sequences = []  # 수정된 부분
Y2_sequences = []  # 수정된 부분
Y3_sequences = []  # 수정된 부분
for i in range(0, len(df) - sequence_lengths, stride):
    X_sequences.append(X[i:i + sequence_lengths])
    Y1_sequences.append(Y1_onehot[i:i + sequence_lengths])  # 수정된 부분
    Y2_sequences.append(Y2_onehot[i:i + sequence_lengths])  # 수정된 부분
    Y3_sequences.append(Y3_onehot[i:i + sequence_lengths])  # 수정된 부분
X_sequences = np.array(X_sequences)
Y1_sequences = np.array(Y1_sequences)  # 수정된 부분
Y2_sequences = np.array(Y2_sequences)  # 수정된 부분
Y3_sequences = np.array(Y3_sequences)  # 수정된 부분

# 5. 모델 구성
input_shape = (sequence_lengths, len(features))
input_layer = Input(shape=input_shape, name='input')
lstm_layer = LSTM(128, return_sequences=True, name='lstm')(input_layer)

# 각 타겟을 위한 Dense 레이어
output1 = Dense(Y1_onehot.shape[1], activation='softmax', name=config.target1)(lstm_layer)  # 수정된 부분
output2 = Dense(Y2_onehot.shape[1], activation='softmax', name=config.target2)(lstm_layer)  # 수정된 부분
output3 = Dense(Y3_onehot.shape[1], activation='softmax', name=config.target3)(lstm_layer)  # 수정된 부분

model = Model(inputs=input_layer, outputs=[output1, output2, output3])

# 6. 모델 컴파일
model.compile(optimizer=Adam(),
              loss={
                  config.target1: 'categorical_crossentropy',
                  config.target2: 'categorical_crossentropy',
                  config.target3: 'categorical_crossentropy'
              },
              metrics=['accuracy'])

# 7. 콜백과 체크포인트 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

# 8. 모델 학습 (수정된 부분)
history = model.fit(
    X_sequences,
    [Y1_sequences, Y2_sequences, Y3_sequences],  # 수정된 부분
    epochs=epoch,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, model_checkpoint]
)

# 9. 학습 결과 저장 (수정된 부분)
history_df = pd.DataFrame(history.history)
history_df.to_csv('training_history.csv', index=False)

# 각 타겟에 대한 정확도를 그래프로 표현 (수정된 부분)
plt.figure(figsize=(12, 8))
plt.plot(history.history[config.target1 + '_accuracy'], label=config.target1 + ' Train Accuracy')
plt.plot(history.history['val_' + config.target1 + '_accuracy'], label=config.target1 + ' Val Accuracy')
plt.plot(history.history[config.target2 + '_accuracy'], label=config.target2 + ' Train Accuracy')
plt.plot(history.history['val_' + config.target2 + '_accuracy'], label=config.target2 + ' Val Accuracy')
plt.plot(history.history[config.target3 + '_accuracy'], label=config.target3 + ' Train Accuracy')
plt.plot(history.history['val_' + config.target3 + '_accuracy'], label=config.target3 + ' Val Accuracy')

plt.title('Model Accuracy for Each Target')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.savefig('accuracy_plot.png')


# 다중 출력 모델: 하나의 입력 특성 세트를 사용하여 여러 타겟을 동시에 예측하는 모델
# LSTM과 Dense 레이어: LSTM 레이어를 통과한 출력을 여러 Dense 레이어에 연결하는 방법
# 활성화 함수: 각 Dense 레이어에서 'softmax' 활성화 함수를 사용?
# 컴파일 설정: 다중 출력 모델을 컴파일할 때, 각 출력 레이어에 다른 손실 함수를 어떻게 설정