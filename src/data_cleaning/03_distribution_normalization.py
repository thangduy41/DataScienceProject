import pandas as pd
import numpy as np

def normalize_distribution(df: pd.DataFrame) -> pd.DataFrame:
    # Áp dụng log1p cho các cột lệch phân phối
    for col in ['Price', 'Area', 'Bedrooms', 'Bathrooms', 'Floors', 'AccessWidth', 'FacadeWidth']:
        df[col] = np.log1p(df[col])
    return df

if __name__ == '__main__':
    # Đọc dữ liệu
    df = pd.read_csv('data/interim/02_feature_engineering.csv')

    # Chuẩn hóa phân phối (log-transform)
    df = normalize_distribution(df)

    # Ghi đè lên file mới
    df.to_csv('data/interim/03_distribution_normalization.csv', index=False)
