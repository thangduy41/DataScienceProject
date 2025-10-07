import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import os
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from utils.load_data import load_csv

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelDataPreparator:
    """Lớp cơ sở cho chuẩn bị dữ liệu mô hình"""
    def __init__(self, input_path: str, output_dir: str, model_dir: str, report_dir: str):
        self.input_path = input_path
        self.output_dir = output_dir
        self.model_dir = model_dir
        self.report_dir = report_dir
        
        # Danh sách các đặc trưng sẽ sử dụng
        self.selected_features = [
            'Price', 'Area', 'Bedrooms', 'Bathrooms', 'Floors',
            'AccessWidth', 'FacadeWidth', 'LegalStatus', 'Furnishing',
            'Distribute', 'GDP_USD'
        ]
        
        # Danh sách các biến số
        self.numerical_columns = [
            'Area', 'Bedrooms', 'Bathrooms', 'Floors',
            'AccessWidth', 'FacadeWidth', 'Distribute', 'GDP_USD'
        ]

    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """Phương thức chuẩn bị dữ liệu cơ bản"""
        raise NotImplementedError("Subclasses must implement prepare_data")

    def save_results(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                    y_train: pd.Series, y_test: pd.Series, 
                    scalers: Dict, algorithm_name: str):
        """Lưu kết quả và tạo báo cáo"""
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Lưu dữ liệu
        X_train.to_csv(os.path.join(self.output_dir, f'X_train_{algorithm_name}.csv'), index=False)
        X_test.to_csv(os.path.join(self.output_dir, f'X_test_{algorithm_name}.csv'), index=False)
        y_train.to_csv(os.path.join(self.output_dir, f'y_train_{algorithm_name}.csv'), index=False)
        y_test.to_csv(os.path.join(self.output_dir, f'y_test_{algorithm_name}.csv'), index=False)
        
        # Lưu scalers
        joblib.dump(scalers, os.path.join(self.model_dir, f'scalers_{algorithm_name}.joblib'))
        
        # Tạo báo cáo
        self._create_report(X_train, X_test, y_train, y_test, algorithm_name)

    def _create_report(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                      y_train: pd.Series, y_test: pd.Series, algorithm_name: str):
        """Tạo báo cáo chi tiết"""
        report_path = os.path.join(self.report_dir, f'07_data_preparation_{algorithm_name}.txt')
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"BÁO CÁO CHUẨN BỊ DỮ LIỆU CHO {algorithm_name.upper()}\n")
            f.write(f"========================================\n\n")
            f.write(f"Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("1. Thống kê dữ liệu:\n")
            f.write(f"   - Tổng số mẫu: {len(X_train) + len(X_test)}\n")
            f.write(f"   - Số mẫu train: {len(X_train)}\n")
            f.write(f"   - Số mẫu test: {len(X_test)}\n\n")
            
            f.write("2. Thông tin về các biến:\n")
            f.write(f"   - Số lượng features: {X_train.shape[1]}\n")
            f.write(f"   - Danh sách features:\n")
            for col in X_train.columns:
                f.write(f"      - {col}\n")
            
            f.write("\n3. Thống kê giá nhà:\n")
            f.write(f"   - Giá trung bình (train): {y_train.mean():,.0f} VND\n")
            f.write(f"   - Giá thấp nhất (train): {y_train.min():,.0f} VND\n")
            f.write(f"   - Giá cao nhất (train): {y_train.max():,.0f} VND\n")
            f.write(f"   - Giá trung bình (test): {y_test.mean():,.0f} VND\n")
            f.write(f"   - Giá thấp nhất (test): {y_test.min():,.0f} VND\n")
            f.write(f"   - Giá cao nhất (test): {y_test.max():,.0f} VND\n")

class LinearRegressionPreparator(ModelDataPreparator):
    """Chuẩn bị dữ liệu cho Linear Regression"""
    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        # Tách features và target
        X = df.drop('Price', axis=1)
        y = df['Price']
        
        # Chia giá thành các nhóm
        price_bins = pd.qcut(y, q=10, labels=False)
        
        # Chuẩn hóa các cột số
        scalers = {}
        for col in self.numerical_columns:
            if col in X.columns:
                scaler = StandardScaler()
                X[col] = scaler.fit_transform(X[[col]])
                scalers[col] = scaler
        
        # Chia dữ liệu thành train và test dựa trên nhóm giá
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=price_bins
        )
        
        return X_train, X_test, y_train, y_test, scalers

class KNNPreparator(ModelDataPreparator):
    """Chuẩn bị dữ liệu cho KNN"""
    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        # Tách features và target
        X = df.drop('Price', axis=1)
        y = df['Price']
        
        # Chia giá thành các nhóm
        price_bins = pd.qcut(y, q=10, labels=False)
        
        # Chuẩn hóa các cột số
        scalers = {}
        for col in self.numerical_columns:
            if col in X.columns:
                scaler = MinMaxScaler()
                X[col] = scaler.fit_transform(X[[col]])
                scalers[col] = scaler
        
        # Chia dữ liệu thành train và test dựa trên nhóm giá
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=price_bins
        )
        
        return X_train, X_test, y_train, y_test, scalers

class XGBoostPreparator(ModelDataPreparator):
    """Chuẩn bị dữ liệu cho XGBoost"""
    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        # Tách features và target
        X = df.drop('Price', axis=1)
        y = df['Price']
        
        # Chia giá thành các nhóm
        price_bins = pd.qcut(y, q=10, labels=False)
        
        # Chuẩn hóa các cột số
        scalers = {}
        for col in self.numerical_columns:
            if col in X.columns:
                scaler = RobustScaler()
                X[col] = scaler.fit_transform(X[[col]])
                scalers[col] = scaler
        
        # Chia dữ liệu thành train và test dựa trên nhóm giá
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=price_bins
        )
        
        return X_train, X_test, y_train, y_test, scalers

def main():
    # Cấu hình
    config = {
        'input_paths': {
            'linear_regression': 'data/interim/06_linear_regression_features.csv',
            'knn': 'data/interim/06_knn_features.csv',
            'xgboost': 'data/interim/06_xgboost_features.csv'
        },
        'output_dir': 'data/processed',
        'model_dir': 'models',
        'report_dir': 'reports'
    }
    
    try:
        # Chuẩn bị dữ liệu cho từng thuật toán
        preparators = {
            'linear_regression': LinearRegressionPreparator,
            'knn': KNNPreparator,
            'xgboost': XGBoostPreparator
        }
        
        for algorithm_name, preparator_class in preparators.items():
            logger.info(f"Đang chuẩn bị dữ liệu cho {algorithm_name}...")
            
            # Đọc dữ liệu từ file tương ứng
            logger.info(f"Đang đọc dữ liệu từ {config['input_paths'][algorithm_name]}...")
            df = load_csv(config['input_paths'][algorithm_name])
            
            preparator = preparator_class(
                config['input_paths'][algorithm_name],
                config['output_dir'],
                config['model_dir'],
                config['report_dir']
            )
            X_train, X_test, y_train, y_test, scalers = preparator.prepare_data(df)
            preparator.save_results(X_train, X_test, y_train, y_test, scalers, algorithm_name)
            logger.info(f"Hoàn thành chuẩn bị dữ liệu cho {algorithm_name}!")
        
    except Exception as e:
        logger.error(f"Có lỗi xảy ra: {str(e)}")
        raise

if __name__ == "__main__":
    main() 