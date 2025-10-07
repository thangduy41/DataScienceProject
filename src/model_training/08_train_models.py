import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from utils.load_data import load_csv

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_and_evaluate_model(X_train: pd.DataFrame, X_test: pd.DataFrame, 
                           y_train: pd.Series, y_test: pd.Series,
                           model_name: str) -> Dict:
    """
    Huấn luyện và đánh giá một mô hình cụ thể.
    
    Parameters:
    -----------
    X_train, X_test : pd.DataFrame
        Features cho tập train và test
    y_train, y_test : pd.Series
        Target cho tập train và test
    model_name : str
        Tên của mô hình cần huấn luyện
        
    Returns:
    --------
    Dict
        Dictionary chứa mô hình và kết quả đánh giá
    """
    # Định nghĩa mô hình dựa trên tên
    if model_name == 'linear_regression':
        model = LinearRegression()
    elif model_name == 'knn':
        model = KNeighborsRegressor(n_neighbors=5, weights='distance')
    elif model_name == 'xgboost':
        model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            gamma=0,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=42,
            n_jobs=-1,
            tree_method='hist',
            objective='reg:squarederror'
        )
    else:
        raise ValueError(f"Không hỗ trợ mô hình {model_name}")
    
    logger.info(f"Đang huấn luyện mô hình {model_name}...")
    
    # Huấn luyện mô hình
    model.fit(X_train, y_train)
    
    # Dự đoán
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Tính toán các metrics
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test)
    }
    
    result = {
        'model': model,
        'metrics': metrics
    }
    
    logger.info(f"Hoàn thành huấn luyện mô hình {model_name}")
    logger.info(f"Test RMSE: {metrics['test_rmse']:,.0f} VND")
    logger.info(f"Test R2: {metrics['test_r2']:.3f}")
    
    return result

def main():
    # Cấu hình
    config = {
        'input_dir': 'data/processed',
        'model_dir': 'models',
        'report_dir': 'reports'
    }
    
    # Danh sách các thuật toán
    algorithms = ['linear_regression', 'knn', 'xgboost']
    
    try:
        results = {}
        
        # Huấn luyện và đánh giá từng mô hình
        for algorithm in algorithms:
            logger.info(f"\nĐang xử lý thuật toán {algorithm}...")
            
            # Đọc dữ liệu
            X_train = load_csv(os.path.join(config['input_dir'], f'X_train_{algorithm}.csv'))
            y_train = load_csv(os.path.join(config['input_dir'], f'y_train_{algorithm}.csv'))['Price']
            X_test = load_csv(os.path.join(config['input_dir'], f'X_test_{algorithm}.csv'))
            y_test = load_csv(os.path.join(config['input_dir'], f'y_test_{algorithm}.csv'))['Price']
            
            # Huấn luyện và đánh giá mô hình
            result = train_and_evaluate_model(X_train, X_test, y_train, y_test, algorithm)
            results[algorithm] = result
            
            # Lưu mô hình
            model_path = os.path.join(config['model_dir'], f'{algorithm}_model.joblib')
            joblib.dump(result['model'], model_path)
            logger.info(f"Đã lưu mô hình tại {model_path}")
        
        # Tạo báo cáo
        report_path = os.path.join(config['report_dir'], '08_model_evaluation.txt')
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"BÁO CÁO ĐÁNH GIÁ MÔ HÌNH\n")
            f.write(f"=====================\n\n")
            f.write(f"Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for algorithm, result in results.items():
                f.write(f"Mô hình: {algorithm.upper()}\n")
                f.write(f"-------------------\n")
                metrics = result['metrics']
                f.write(f"RMSE (Train): {metrics['train_rmse']:,.0f} VND\n")
                f.write(f"RMSE (Test): {metrics['test_rmse']:,.0f} VND\n")
                f.write(f"R2 Score (Train): {metrics['train_r2']:.3f}\n")
                f.write(f"R2 Score (Test): {metrics['test_r2']:.3f}\n")
                f.write(f"MAE (Train): {metrics['train_mae']:,.0f} VND\n")
                f.write(f"MAE (Test): {metrics['test_mae']:,.0f} VND\n")
                f.write(f"MAPE (Test): {metrics['test_mae'] / y_test.mean() * 100:.2f}%\n\n")
            
            # Xác định mô hình tốt nhất dựa trên R2 score
            best_model = max(results.items(), key=lambda x: x[1]['metrics']['test_r2'])[0]
            f.write(f"\nMô hình tốt nhất: {best_model.upper()}\n")
            f.write(f"R2 Score: {results[best_model]['metrics']['test_r2']:.3f}\n")
        
        logger.info("Hoàn thành huấn luyện và đánh giá mô hình!")
        
    except Exception as e:
        logger.error(f"Có lỗi xảy ra: {str(e)}")
        raise

if __name__ == "__main__":
    main() 