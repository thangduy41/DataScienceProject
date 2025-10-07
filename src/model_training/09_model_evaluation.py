import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from utils.load_data import load_csv

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def evaluate_model(X_test: pd.DataFrame, y_test: pd.Series, model, model_name: str) -> Dict:
    """
    Đánh giá chi tiết một mô hình.
    
    Parameters:
    -----------
    X_test : pd.DataFrame
        Features cho tập test
    y_test : pd.Series
        Target cho tập test
    model : object
        Mô hình cần đánh giá
    model_name : str
        Tên của mô hình
        
    Returns:
    --------
    Dict
        Dictionary chứa các metrics và thông tin đánh giá
    """
    # Dự đoán
    y_pred = model.predict(X_test)
    
    # Tính toán các metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mae / y_test.mean() * 100
    
    # Tính toán residual
    residuals = y_test - y_pred
    
    # Tính toán các thống kê về residual
    residual_stats = {
        'mean': residuals.mean(),
        'std': residuals.std(),
        'min': residuals.min(),
        'max': residuals.max(),
        'skew': residuals.skew(),
        'kurtosis': residuals.kurtosis()
    }
    
    # Tính toán các phân vị của residual
    residual_percentiles = {
        '1%': np.percentile(residuals, 1),
        '5%': np.percentile(residuals, 5),
        '25%': np.percentile(residuals, 25),
        '50%': np.percentile(residuals, 50),
        '75%': np.percentile(residuals, 75),
        '95%': np.percentile(residuals, 95),
        '99%': np.percentile(residuals, 99)
    }
    
    return {
        'metrics': {
            'rmse': rmse,
            'r2': r2,
            'mae': mae,
            'mape': mape
        },
        'residual_stats': residual_stats,
        'residual_percentiles': residual_percentiles,
        'predictions': y_pred,
        'residuals': residuals
    }

def plot_evaluation_results(y_test: pd.Series, evaluation_results: Dict, model_name: str, output_dir: str):
    """
    Tạo các biểu đồ đánh giá mô hình.
    
    Parameters:
    -----------
    y_test : pd.Series
        Giá trị thực tế
    evaluation_results : Dict
        Kết quả đánh giá mô hình
    model_name : str
        Tên của mô hình
    output_dir : str
        Thư mục lưu biểu đồ
    """
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Biểu đồ scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, evaluation_results['predictions'], alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Giá thực tế')
    plt.ylabel('Giá dự đoán')
    plt.title(f'Scatter Plot - {model_name.upper()}')
    plt.savefig(os.path.join(output_dir, f'{model_name}_scatter.png'))
    plt.close()
    
    # 2. Biểu đồ residual
    plt.figure(figsize=(10, 6))
    plt.scatter(evaluation_results['predictions'], evaluation_results['residuals'], alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Giá dự đoán')
    plt.ylabel('Residual')
    plt.title(f'Residual Plot - {model_name.upper()}')
    plt.savefig(os.path.join(output_dir, f'{model_name}_residual.png'))
    plt.close()
    
    # 3. Biểu đồ phân phối residual
    plt.figure(figsize=(10, 6))
    sns.histplot(evaluation_results['residuals'], kde=True)
    plt.xlabel('Residual')
    plt.ylabel('Tần suất')
    plt.title(f'Phân phối Residual - {model_name.upper()}')
    plt.savefig(os.path.join(output_dir, f'{model_name}_residual_dist.png'))
    plt.close()

def main():
    # Cấu hình
    config = {
        'input_dir': 'data/processed',
        'model_dir': 'models',
        'report_dir': 'reports',
        'plot_dir': 'reports/plots'
    }
    
    # Danh sách các thuật toán
    algorithms = ['linear_regression', 'knn', 'xgboost']
    
    try:
        results = {}
        
        # Đánh giá từng mô hình
        for algorithm in algorithms:
            logger.info(f"\nĐang đánh giá mô hình {algorithm}...")
            
            # Đọc dữ liệu test
            X_test = load_csv(os.path.join(config['input_dir'], f'X_test_{algorithm}.csv'))
            y_test = load_csv(os.path.join(config['input_dir'], f'y_test_{algorithm}.csv'))['Price']
            
            # Đọc mô hình
            model_path = os.path.join(config['model_dir'], f'{algorithm}_model.joblib')
            model = joblib.load(model_path)
            
            # Đánh giá mô hình
            evaluation = evaluate_model(X_test, y_test, model, algorithm)
            results[algorithm] = evaluation
            
            # Tạo biểu đồ
            plot_evaluation_results(y_test, evaluation, algorithm, config['plot_dir'])
        
        # Tạo báo cáo
        report_path = os.path.join(config['report_dir'], '09_model_evaluation.txt')
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"BÁO CÁO ĐÁNH GIÁ CHI TIẾT MÔ HÌNH\n")
            f.write(f"==============================\n\n")
            f.write(f"Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for algorithm, evaluation in results.items():
                f.write(f"Mô hình: {algorithm.upper()}\n")
                f.write(f"-------------------\n")
                
                # Metrics
                metrics = evaluation['metrics']
                f.write("1. Metrics:\n")
                f.write(f"   - RMSE: {metrics['rmse']:,.0f} VND\n")
                f.write(f"   - R2 Score: {metrics['r2']:.3f}\n")
                f.write(f"   - MAE: {metrics['mae']:,.0f} VND\n")
                f.write(f"   - MAPE: {metrics['mape']:.2f}%\n\n")
                
                # Residual statistics
                residual_stats = evaluation['residual_stats']
                f.write("2. Thống kê Residual:\n")
                f.write(f"   - Mean: {residual_stats['mean']:,.0f} VND\n")
                f.write(f"   - Std: {residual_stats['std']:,.0f} VND\n")
                f.write(f"   - Min: {residual_stats['min']:,.0f} VND\n")
                f.write(f"   - Max: {residual_stats['max']:,.0f} VND\n")
                f.write(f"   - Skewness: {residual_stats['skew']:.3f}\n")
                f.write(f"   - Kurtosis: {residual_stats['kurtosis']:.3f}\n\n")
                
                # Residual percentiles
                residual_percentiles = evaluation['residual_percentiles']
                f.write("3. Phân vị Residual:\n")
                for percentile, value in residual_percentiles.items():
                    f.write(f"   - {percentile}: {value:,.0f} VND\n")
                f.write("\n")
            
            # Xác định mô hình tốt nhất
            best_model = max(results.items(), key=lambda x: x[1]['metrics']['r2'])[0]
            f.write(f"\nMô hình tốt nhất: {best_model.upper()}\n")
            f.write(f"R2 Score: {results[best_model]['metrics']['r2']:.3f}\n")
        
        logger.info("Hoàn thành đánh giá mô hình!")
        
    except Exception as e:
        logger.error(f"Có lỗi xảy ra: {str(e)}")
        raise

if __name__ == "__main__":
    main() 