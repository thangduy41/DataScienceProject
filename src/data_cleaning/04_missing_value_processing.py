import pandas as pd
import numpy as np
from typing import List, Dict, Union
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from utils.load_data import load_csv

"""
Phương pháp xử lý giá trị thiếu:

1. Giá trị số (Bedrooms, Bathrooms, Floors, AccessWidth, FacadeWidth):
   - Mean/Median: Điền bằng giá trị trung bình/trung vị
   - KNN: Điền dựa trên k giá trị gần nhất
   - Interpolation: Nội suy tuyến tính

2. Giá trị phân loại (không có cột phân loại nào có giá trị thiếu):
   - Mode: Điền bằng giá trị xuất hiện nhiều nhất
   - KNN: Điền dựa trên k giá trị gần nhất
   - Custom: Điền bằng giá trị mặc định
"""

# Định nghĩa phương pháp xử lý cho từng cột
column_methods = {
    # Cột số
    'Bedrooms': 'mode',
    'Bathrooms': 'mode',
    'Floors': 'mode',
    'AccessWidth': 'median',
    'FacadeWidth': 'median'
}

def process_missing_values(
    df: pd.DataFrame,
    column_methods: Dict[str, str],
    plot: bool = False,
    save_path: str = None
) -> pd.DataFrame:
    """
    Xử lý giá trị thiếu trong DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame cần xử lý
    column_methods : Dict[str, str]
        Dictionary chứa phương pháp xử lý cho từng cột
    plot : bool
        Có vẽ đồ thị phân tích không
    save_path : str
        Đường dẫn để lưu kết quả
        
    Returns:
    --------
    pd.DataFrame
        DataFrame đã được xử lý
    """
    df_clean = df.copy()
    
    for col, method in column_methods.items():
        if col not in df_clean.columns:
            print(f"Cột {col} không tồn tại trong DataFrame")
            continue
        
        missing_count = df_clean[col].isnull().sum()
        if missing_count == 0:
            continue
        
        print(f"\nXử lý giá trị thiếu cho cột {col}:")
        print(f"Số giá trị thiếu: {missing_count}")
        
        if method == 'mean':
            fill_value = df_clean[col].mean()
        elif method == 'median':
            fill_value = df_clean[col].median()
        elif method == 'mode':
            fill_value = df_clean[col].mode()[0]
        
        df_clean[col] = df_clean[col].fillna(fill_value)
        
        if plot:
            plt.figure(figsize=(12, 4))
            
            # Histogram trước và sau khi điền
            plt.subplot(1, 2, 1)
            sns.histplot(df[col].dropna(), bins=50, kde=True, label='Trước khi điền')
            sns.histplot(df_clean[col], bins=50, kde=True, label='Sau khi điền')
            plt.title(f'Phân bố của {col}')
            plt.legend()
            
            # Boxplot trước và sau khi điền
            plt.subplot(1, 2, 2)
            sns.boxplot(data=pd.DataFrame({
                'Trước khi điền': df[col].dropna(),
                'Sau khi điền': df_clean[col]
            }))
            plt.title(f'Boxplot của {col}')
            
            plt.tight_layout()
            plt.show()
    
    # Lưu kết quả nếu có đường dẫn
    if save_path:
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Lưu DataFrame
        df_clean.to_csv(save_path, index=False)
    
    return df_clean

def main():
    # Cấu hình
    config = {
        'plot': True,                # True/False
        'save': True,                # True/False
        'input': 'data/interim/03_distribution_normalization.csv',
        'output_dir': 'data/interim',
        'report_dir': 'reports'
    }
    
    # Đọc dữ liệu sử dụng hàm load_csv
    df = load_csv(config['input'])
    
    # Tạo đường dẫn lưu file
    save_path = None
    report_path = None
    if config['save']:
        # Lưu file CSV vào thư mục data
        save_path = os.path.join(config['output_dir'], f'04_processed_missing.csv')
        # Lưu file báo cáo vào thư mục reports
        report_path = os.path.join(config['report_dir'], f'04_missing_value_analysis.txt')
    
    # Xử lý giá trị thiếu
    process_missing_values(
        df,
        column_methods=column_methods,
        plot=config['plot'],
        save_path=save_path
    )
    
    # Lưu báo cáo nếu có đường dẫn
    if report_path:
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"BÁO CÁO XỬ LÝ GIÁ TRỊ THIẾU\n")
            f.write(f"==========================\n\n")
            f.write(f"Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("1. Cấu hình xử lý:\n")
            for col, method in column_methods.items():
                f.write(f"   - Phương pháp cho cột {col}: {method}\n")
    
    print("\nĐã xử lý giá trị thiếu xong!")
    if save_path:
        print(f"\nDữ liệu đã được lưu tại: {save_path}")
    if report_path:
        print(f"Báo cáo đã được lưu tại: {report_path}")

if __name__ == "__main__":
    main() 