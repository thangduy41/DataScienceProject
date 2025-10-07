import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import os
from utils.load_data import load_csv
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureProcessor:
    """Lớp cơ sở cho xử lý đặc trưng"""
    def __init__(self, input_path: str, output_dir: str, report_dir: str):
        self.input_path = input_path
        self.output_dir = output_dir
        self.report_dir = report_dir
        
        # Danh sách các đặc trưng sẽ sử dụng
        self.selected_features = [
            'Price', 'Area', 'Bedrooms', 'Bathrooms', 'Floors',
            'AccessWidth', 'FacadeWidth', 'LegalStatus', 'Furnishing',
            'Distribute', 'GDP_USD'
        ]
        
        # Danh sách các biến phân loại
        self.categorical_columns = ['LegalStatus', 'Furnishing']
        
        # Danh sách các biến số
        self.numerical_columns = [
            'Price', 'Area', 'Bedrooms', 'Bathrooms', 'Floors',
            'AccessWidth', 'FacadeWidth', 'Distribute', 'GDP_USD'
        ]

    def process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Phương thức xử lý đặc trưng cơ bản"""
        raise NotImplementedError("Subclasses must implement process_features")

    def save_results(self, df: pd.DataFrame, algorithm_name: str):
        """Lưu kết quả và tạo báo cáo"""
        # Tạo tên file dựa trên thuật toán
        output_path = os.path.join(self.output_dir, f'06_{algorithm_name}_features.csv')
        report_path = os.path.join(self.report_dir, f'06_{algorithm_name}_feature_processing.txt')
        
        # Lưu dữ liệu
        df.to_csv(output_path, index=False)
        logger.info(f"Đã lưu kết quả tại: {output_path}")
        
        # Tạo báo cáo
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        self._create_report(df, report_path, algorithm_name)

    def _create_report(self, df: pd.DataFrame, report_path: str, algorithm_name: str):
        """Tạo báo cáo chi tiết"""
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"BÁO CÁO XỬ LÝ ĐẶC TRƯNG CHO {algorithm_name.upper()}\n")
            f.write(f"========================================\n\n")
            f.write(f"Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("1. Thống kê:\n")
            f.write(f"   - Số lượng biến: {len(df.columns)}\n")
            f.write(f"   - Số lượng mẫu: {len(df)}\n\n")
            
            f.write("2. Danh sách các biến:\n")
            for col in df.columns:
                f.write(f"   - {col}\n")

class LinearRegressionProcessor(FeatureProcessor):
    """Xử lý đặc trưng cho Linear Regression"""
    def __init__(self, input_path: str, output_dir: str, report_dir: str):
        super().__init__(input_path, output_dir, report_dir)
        self.encoding_config = {
            'LegalStatus': 'one_hot',
            'Furnishing': 'one_hot'
        }
        self.scaling_config = {
            'Price': 'standard',
            'Area': 'standard',
            'Bedrooms': 'standard',
            'Bathrooms': 'standard',
            'Floors': 'standard',
            'AccessWidth': 'standard',
            'FacadeWidth': 'standard',
            'Distribute': 'standard',
            'GDP_USD': 'standard'
        }

    def process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Xử lý đặc trưng cho Linear Regression"""
        df_filtered = df[self.selected_features].copy()
        
        # Chuẩn hóa biến số
        for col in self.numerical_columns:
            if col in df_filtered.columns:
                scaler = StandardScaler()
                df_filtered[col] = scaler.fit_transform(df_filtered[[col]])
        
        # Mã hóa biến phân loại
        for col in self.categorical_columns:
            if col in df_filtered.columns:
                dummies = pd.get_dummies(df_filtered[col], prefix=col)
                df_filtered = pd.concat([df_filtered.drop(col, axis=1), dummies], axis=1)
        
        return df_filtered

class KNNProcessor(FeatureProcessor):
    """Xử lý đặc trưng cho KNN"""
    def __init__(self, input_path: str, output_dir: str, report_dir: str):
        super().__init__(input_path, output_dir, report_dir)
        self.encoding_config = {
            'LegalStatus': 'one_hot',
            'Furnishing': 'one_hot'
        }
        self.scaling_config = {
            'Price': 'minmax',
            'Area': 'minmax',
            'Bedrooms': 'minmax',
            'Bathrooms': 'minmax',
            'Floors': 'minmax',
            'AccessWidth': 'minmax',
            'FacadeWidth': 'minmax',
            'Distribute': 'minmax',
            'GDP_USD': 'minmax'
        }

    def process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Xử lý đặc trưng cho KNN"""
        df_filtered = df[self.selected_features].copy()
        
        # Chuẩn hóa biến số
        for col in self.numerical_columns:
            if col in df_filtered.columns:
                scaler = MinMaxScaler()
                df_filtered[col] = scaler.fit_transform(df_filtered[[col]])
        
        # Mã hóa biến phân loại
        for col in self.categorical_columns:
            if col in df_filtered.columns:
                dummies = pd.get_dummies(df_filtered[col], prefix=col)
                df_filtered = pd.concat([df_filtered.drop(col, axis=1), dummies], axis=1)
        
        return df_filtered

class XGBoostProcessor(FeatureProcessor):
    """Xử lý đặc trưng cho XGBoost"""
    def __init__(self, input_path: str, output_dir: str, report_dir: str):
        super().__init__(input_path, output_dir, report_dir)
        self.encoding_config = {
            'LegalStatus': 'label',  # XGBoost thường hoạt động tốt với label encoding
            'Furnishing': 'label'
        }
        self.scaling_config = {
            'Price': 'robust',
            'Area': 'robust',
            'Bedrooms': 'robust',
            'Bathrooms': 'robust',
            'Floors': 'robust',
            'AccessWidth': 'robust',
            'FacadeWidth': 'robust',
            'Distribute': 'robust',
            'GDP_USD': 'robust'
        }

    def process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Xử lý đặc trưng cho XGBoost"""
        df_filtered = df[self.selected_features].copy()
        
        # Chuẩn hóa biến số
        for col in self.numerical_columns:
            if col in df_filtered.columns:
                scaler = RobustScaler()
                df_filtered[col] = scaler.fit_transform(df_filtered[[col]])
        
        # Mã hóa biến phân loại
        for col in self.categorical_columns:
            if col in df_filtered.columns:
                df_filtered[col] = df_filtered[col].astype('category').cat.codes
        
        return df_filtered

def main():
    # Cấu hình
    config = {
        'input': 'data/interim/05_processed_outliers.csv',
        'output_dir': 'data/interim',
        'report_dir': 'reports'
    }
    
    try:
        # Đọc dữ liệu
        logger.info("Đang đọc dữ liệu...")
        df = load_csv(config['input'])
        
        # Xử lý đặc trưng cho từng thuật toán
        processors = {
            'linear_regression': LinearRegressionProcessor,
            'knn': KNNProcessor,
            'xgboost': XGBoostProcessor
        }
        
        for algorithm_name, processor_class in processors.items():
            logger.info(f"Đang xử lý đặc trưng cho {algorithm_name}...")
            processor = processor_class(config['input'], config['output_dir'], config['report_dir'])
            df_processed = processor.process_features(df)
            processor.save_results(df_processed, algorithm_name)
            logger.info(f"Hoàn thành xử lý đặc trưng cho {algorithm_name}!")
        
    except Exception as e:
        logger.error(f"Có lỗi xảy ra: {str(e)}")
        raise

if __name__ == "__main__":
    main() 