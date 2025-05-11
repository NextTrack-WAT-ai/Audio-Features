import pandas as pd
import glob
import logging
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Assuming your CSVs are in a folder
csv_files = glob.glob('./comparisons/*.csv')
# print("here")
logging.info(f"Found {len(csv_files)} CSV files in ./comparisons/")
dfs = [pd.read_csv(file) for file in csv_files]
full_df = pd.concat(dfs, ignore_index=True)

# Include key_error_score rows and prepare for scaling
key_df = full_df[full_df['Feature'] == 'key_error_score'][['Feature', 'Your Algorithm']].copy()
key_df = key_df.rename(columns={'Your Algorithm': 'Difference'})  # Treat as its own "error" metric

# Filter only real difference-based features
diff_df = full_df[full_df['Feature'] != 'key_error_score'][['Feature', 'Difference']]
full_diff_df = pd.concat([diff_df, key_df], ignore_index=True)




scaler = MinMaxScaler()
full_diff_df['Scaled_Difference'] = scaler.fit_transform(full_diff_df[['Difference']])

summary = (
    full_diff_df.groupby('Feature')['Scaled_Difference']
    .agg(['mean','median', 'std', 'max', 'min'])
    .sort_values('mean', ascending=False)
)

summary.to_csv('./a_summary.csv', index=True)
logging.info(summary)

