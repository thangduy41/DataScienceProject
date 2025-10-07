import pandas as pd
import numpy as np
from typing import List, Union, Dict
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from datetime import datetime
from utils.load_data import load_csv

"""
Phương pháp xử lý ngoại lệ:

1. IQR (Interquartile Range):
   - Áp dụng cho: Price, Area
   - Lý do: Phân phối lệch, có nhiều giá trị cực đoan
   - Tham số: iqr_multiplier = 2.5 (tăng ngưỡng để giữ lại nhiều dữ liệu hơn)

2. Percentile:
   - Áp dụng cho: Bedrooms, Bathrooms, Floors
   - Lý do: Biến rời rạc, cần kiểm soát chính xác tỷ lệ dữ liệu
   - Tham số: percentile_low = 0.01, percentile_high = 0.99

3. Z-score:
   - Áp dụng cho: AccessWidth, FacadeWidth
   - Lý do: Phân phối gần chuẩn, cần xét đến giá trị trung bình
   - Tham số: z_score_threshold = 3.5 (tăng ngưỡng để giữ lại nhiều dữ liệu hơn)
"""

def process_outliers(
    df: pd.DataFrame,
    column_methods: Dict[str, str],
    iqr_multiplier: float = 1.5,
    percentile_low: float = 0.01,
    percentile_high: float = 0.99,
    z_score_threshold: float = 3,
    plot: bool = False,
    save_path: str = None
) -> pd.DataFrame:
    """
    Xử lý ngoại lệ cho các cột số trong DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame cần xử lý
    column_methods : Dict[str, str]
        Dictionary chứa phương pháp xử lý cho từng cột
    iqr_multiplier : float
        Hệ số nhân cho phương pháp IQR
    percentile_low : float
        Ngưỡng dưới cho phương pháp percentile
    percentile_high : float
        Ngưỡng trên cho phương pháp percentile
    z_score_threshold : float
        Ngưỡng cho phương pháp z-score
    plot : bool
        Có vẽ đồ thị phân tích không
    save_path : str
        Đường dẫn để lưu kết quả
        
    Returns:
    --------
    pd.DataFrame, Dict
        DataFrame đã được xử lý ngoại lệ và dictionary chứa các ngưỡng
    """
    df_clean = df.copy()
    thresholds = {}
    initial_count = len(df_clean)
    
    print(f"\nSố bản ghi ban đầu: {initial_count}")
    
    for col, method in column_methods.items():
        if col not in df_clean.columns:
            print(f"Cột {col} không tồn tại trong DataFrame")
            continue
            
        if not pd.api.types.is_numeric_dtype(df_clean[col]):
            print(f"Cột {col} không phải kiểu số")
            continue
            
        print(f"\nXử lý ngoại lệ cho cột {col} (Phương pháp: {method.upper()}):")
        print(f"Số bản ghi hiện tại: {len(df_clean)}")
        
        # Tính ngưỡng theo phương pháp được chọn
        if method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            low_bound = Q1 - iqr_multiplier * IQR
            high_bound = Q3 + iqr_multiplier * IQR
            
        elif method == 'percentile':
            low_bound = df_clean[col].quantile(percentile_low)
            high_bound = df_clean[col].quantile(percentile_high)
            
        elif method == 'zscore':
            mean = df_clean[col].mean()
            std = df_clean[col].std()
            low_bound = mean - z_score_threshold * std
            high_bound = mean + z_score_threshold * std
            
        else:
            raise ValueError("Phương pháp không hợp lệ. Chọn 'iqr', 'percentile' hoặc 'zscore'")
        
        # Lưu ngưỡng
        thresholds[col] = {'low': low_bound, 'high': high_bound}
        
        # Lọc dữ liệu
        mask = (df_clean[col] >= low_bound) & (df_clean[col] <= high_bound)
        df_clean = df_clean[mask]
        
        print(f"Ngưỡng dưới: {low_bound:.2f}")
        print(f"Ngưỡng trên: {high_bound:.2f}")
        print(f"Số bản ghi sau khi lọc: {len(df_clean)}")
        print(f"Số bản ghi bị loại: {len(df) - len(df_clean)}")
        
        # Vẽ đồ thị nếu được yêu cầu
        if plot:
            plt.figure(figsize=(12, 4))
            
            # Histogram
            plt.subplot(1, 2, 1)
            sns.histplot(df[col].dropna(), bins=50, kde=True)
            plt.axvline(low_bound, color='r', linestyle='--', label='Ngưỡng dưới')
            plt.axvline(high_bound, color='r', linestyle='--', label='Ngưỡng trên')
            plt.title(f'Phân bố của {col}')
            plt.legend()
            
            # Boxplot
            plt.subplot(1, 2, 2)
            sns.boxplot(x=df[col])
            plt.title(f'Boxplot của {col}')
            
            plt.tight_layout()
            plt.show()
    
    # Lưu kết quả nếu có đường dẫn
    if save_path:
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Lưu DataFrame
        df_clean.to_csv(save_path, index=False)
    
    return df_clean, thresholds

def main():
    # Cấu hình
    config = {
        'plot': True,     # True/False
        'save': True,     # True/False
        'input': 'data/interim/04_processed_missing.csv',
        'output_dir': 'data/interim',
        'report_dir': 'reports'
    }
    
    # Định nghĩa phương pháp xử lý cho từng cột
    column_methods = {
        'Price': 'zscore',
        'Area': 'zscore',
        'Bedrooms': 'iqr',
        'Bathrooms': 'iqr',
        'Floors': 'zscore',
        'AccessWidth': 'iqr',
        'FacadeWidth': 'iqr'
    }
    
    # Đọc dữ liệu sử dụng hàm load_csv
    df = load_csv(config['input'])
    
    # Tạo đường dẫn lưu file
    save_path = None
    report_path = None
    if config['save']:
        # Lưu file CSV vào thư mục data
        save_path = os.path.join(config['output_dir'], '05_processed_outliers.csv')
        # Lưu file báo cáo vào thư mục reports
        report_path = os.path.join(config['report_dir'], '05_outlier_analysis.txt')
    
    # Xử lý ngoại lệ
    df_clean, thresholds = process_outliers(
        df,
        column_methods,
        plot=config['plot'],
        save_path=save_path
    )
    
    # Lưu báo cáo nếu có đường dẫn
    if report_path:
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"BÁO CÁO XỬ LÝ NGOẠI LỆ\n")
            f.write(f"========================\n\n")
            f.write(f"Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Số bản ghi ban đầu: {len(df)}\n")
            f.write(f"Số bản ghi sau xử lý: {len(df_clean)}\n")
            f.write(f"Số bản ghi bị loại: {len(df) - len(df_clean)}\n\n")
            
            f.write("Phương pháp xử lý cho từng cột:\n")
            for col, method in column_methods.items():
                f.write(f"{col}: {method.upper()}\n")
            
            f.write("\nCác ngưỡng đã sử dụng:\n")
            for col, bounds in thresholds.items():
                f.write(f"{col}: {bounds['low']:.2f} - {bounds['high']:.2f}\n")
    
    print("\nĐã xử lý ngoại lệ xong!")
    if save_path:
        print(f"\nDữ liệu đã được lưu tại: {save_path}")
    if report_path:
        print(f"Báo cáo đã được lưu tại: {report_path}")

if __name__ == "__main__":
    main() 