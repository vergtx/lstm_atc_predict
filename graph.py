import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 데이터 불러오기
original_data_path = 'tos_data_ver6_on_6th.csv'
predict_data_path = 'tos_data_ver6_on_6th_needed_predict_with_predictions.csv'

original_data_df = pd.read_csv(original_data_path)
predict_data_df = pd.read_csv(predict_data_path)

# 최대 1800개의 레코드를 비교
original_data_subset = original_data_df.head(1800)
predict_data_subset = predict_data_df.head(1800)

# 일치율 계산을 위한 함수
def calculate_match_percentage(original, predict, column_name, subset_size):
    return (original.head(subset_size)[column_name] == predict.head(subset_size)[column_name]).sum() / subset_size * 100

# 다양한 서브셋 크기에 대한 일치율 계산
subset_sizes = [50, 100, 900, 1800]
bay_match_percentages = []
row_match_percentages = []
tier_match_percentages = []

for size in subset_sizes:
    bay_match_percentages.append(calculate_match_percentage(original_data_subset, predict_data_subset, 'bay2', size))
    row_match_percentages.append(calculate_match_percentage(original_data_subset, predict_data_subset, 'row2', size))
    tier_match_percentages.append(calculate_match_percentage(original_data_subset, predict_data_subset, 'tier2', size))

# x축의 위치와 레이블 정의
labels = [f"First {size} Records" for size in subset_sizes]
x = np.arange(len(labels))

# 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(x, bay_match_percentages, marker='o', label='Bay Match (%)', color='red')
plt.plot(x, row_match_percentages, marker='s', label='Row Match (%)', color='blue')
plt.plot(x, tier_match_percentages, marker='d', label='Tier Match (%)', color='green')

plt.xticks(x, labels)
plt.xlabel('Data Subset')
plt.ylabel('Match Percentage (%)')
plt.title('Comparison of Match Percentages for Different Data Subsets')
plt.legend()
plt.grid(True)

# 그래프를 파일로 저장
plt.savefig('comparison_graph_with_colors.png')

# 결과를 표로 저장
comparison_df = pd.DataFrame({
    'Data Subset': labels,
    'Bay Match (%)': bay_match_percentages,
    'Row Match (%)': row_match_percentages,
    'Tier Match (%)': tier_match_percentages
})

# CSV 파일로 저장
comparison_df.to_csv('comparison_table.csv', index=False)
