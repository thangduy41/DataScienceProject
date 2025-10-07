import pandas as pd
import numpy as np
import re
from utils.load_data import load_csv
import logging
from datetime import datetime
import os

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def normalize_number(value):
    """Chuẩn hóa giá trị số từ chuỗi"""
    if pd.isnull(value):
        return np.nan
    value = str(value)
    value = re.sub(r'[^0-9,\.]', '', value)
    value = value.replace(',', '.')
    try:
        return float(value)
    except ValueError:
        return np.nan

def convert_price(value, area=None):
    """Chuyển đổi và chuẩn hóa mức giá"""
    value = str(value).lower().replace('.', '').replace(',', '.')
    if any(unit in value for unit in ['thỏa thuận', 'nghìn', 'tỷ/m²', 'triệu/tháng']):
        return np.nan
    if 'triệu/m²' in value:
        try:
            price_per_m2 = float(value.replace('triệu/m²', '').strip())
            return round(price_per_m2 * area / 1000, 2)  # Chia cho 1000 để ra tỷ đồng
        except ValueError:
            return np.nan
    if 'triệu' in value:
        try:
            return float(value.replace('triệu', '').strip()) / 1000
        except ValueError:
            return np.nan
    if 'tỷ' in value:
        try:
            return float(value.replace('tỷ', '').strip())
        except ValueError:
            return np.nan
    return np.nan

column_rename_map = {
    'Diện tích': 'Area',
    'Số phòng ngủ': 'Bedrooms',
    'Số phòng tắm, vệ sinh': 'Bathrooms',
    'Số tầng': 'Floors',
    'Đường vào': 'AccessWidth',
    'Mặt tiền': 'FacadeWidth',
    'Mức giá': 'Price',
    'Địa điểm': 'Location',
    'Pháp lý': 'LegalStatus',
    'Nội thất': 'Furnishing'
}

def preprocess_data(df):
    """
    Tiền xử lý dữ liệu bất động sản.
    
    Args:
        df (pd.DataFrame): DataFrame chứa dữ liệu thô
        
    Returns:
        pd.DataFrame: DataFrame đã được xử lý
    """
    initial_rows = len(df)
    
    # 1. Loại bỏ các cột không cần thiết
    columns_to_drop = ['Tiêu đề', 'URL', 'Ngày đăng', 'Hướng nhà', 'Hướng ban công']
    df = df.drop(columns=columns_to_drop)

    # 2. Loại bỏ các dòng có nhiều giá trị thiếu
    df = df.dropna(thresh=len(df.columns) - 3)

    # 3. Đổi tên cột sang tiếng Anh
    df = df.rename(columns=column_rename_map)

    # 4. Chuẩn hóa các cột số
    numeric_columns = ['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'AccessWidth', 'FacadeWidth']
    for col in numeric_columns: df[col] = df[col].apply(normalize_number)

         
    # 5. Chuẩn hóa mức giá
    df['Price'] = df.apply(lambda row: convert_price(row['Price'], row['Area']), axis=1)
    df = df.dropna(subset=['Price'])

    # 6. Tách địa điểm
    location_split = df['Location'].str.split(',', expand=True, n=1)
    df = df.drop(columns=['Location'])
    df = pd.concat([df, location_split.rename(columns={0:'District', 1:'Province'})], axis=1)

    # 7. Chuẩn hóa các cột phân loại
    df['LegalStatus'] = df['LegalStatus'].apply(lambda x: 'yes' if 'sổ' in str(x).lower() and 'chờ' not in str(x).lower() else 'no')
    df['Furnishing'] = df['Furnishing'].apply(lambda x: 'no' if pd.isna(x) or str(x).strip() == '' or 'no' in str(x).lower() else 'yes')

    # 8. Chuyển đổi kiểu dữ liệu
    df = df.convert_dtypes()

    final_rows = len(df)
    logger.info(f"Số dòng dữ liệu: {initial_rows} -> {final_rows} (giảm {initial_rows - final_rows} dòng)")

    return df

def generate_preprocessing_report(df_before: pd.DataFrame, df_after: pd.DataFrame, report_dir: str):
    """
    Tạo báo cáo chi tiết về quá trình tiền xử lý dữ liệu.
    
    Parameters:
    -----------
    df_before : pd.DataFrame
        DataFrame trước khi xử lý
    df_after : pd.DataFrame
        DataFrame sau khi xử lý
    report_dir : str
        Thư mục lưu báo cáo
    """
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, '01_preprocessing_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"BÁO CÁO TIỀN XỬ LÝ DỮ LIỆU\n")
        f.write(f"=====================\n\n")
        f.write(f"Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 1. Thông tin tổng quan
        f.write("1. THÔNG TIN TỔNG QUAN\n")
        f.write("-------------------\n")
        f.write(f"Số dòng trước xử lý: {len(df_before)}\n")
        f.write(f"Số dòng sau xử lý: {len(df_after)}\n")
        f.write(f"Số dòng bị loại bỏ: {len(df_before) - len(df_after)}\n")
        f.write(f"Tỷ lệ giữ lại: {(len(df_after) / len(df_before) * 100):.2f}%\n\n")
        
        # 2. Thông tin về các cột
        f.write("2. THÔNG TIN VỀ CÁC CỘT\n")
        f.write("-------------------\n")
        f.write("Các cột được giữ lại:\n")
        for col in df_after.columns:
            f.write(f"- {col}\n")
        f.write("\nCác cột bị loại bỏ:\n")
        for col in df_before.columns:
            if col not in df_after.columns:
                f.write(f"- {col}\n")
        f.write("\n")
        
        # 3. Thống kê về dữ liệu thiếu
        f.write("3. THỐNG KÊ DỮ LIỆU THIẾU\n")
        f.write("-------------------\n")
        missing_stats = df_after.isnull().sum()
        missing_stats = missing_stats[missing_stats > 0]
        if len(missing_stats) > 0:
            for col, count in missing_stats.items():
                f.write(f"- {col}: {count} giá trị thiếu ({count/len(df_after)*100:.2f}%)\n")
        else:
            f.write("Không có dữ liệu thiếu sau khi xử lý\n")
        f.write("\n")
        
        # 4. Thống kê về các biến số
        f.write("4. THỐNG KÊ CÁC BIẾN SỐ\n")
        f.write("-------------------\n")
        numeric_cols = df_after.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            stats = df_after[col].describe()
            f.write(f"\n{col}:\n")
            f.write(f"- Số lượng: {stats['count']:.0f}\n")
            f.write(f"- Giá trị trung bình: {stats['mean']:.2f}\n")
            f.write(f"- Độ lệch chuẩn: {stats['std']:.2f}\n")
            f.write(f"- Giá trị nhỏ nhất: {stats['min']:.2f}\n")
            f.write(f"- Giá trị lớn nhất: {stats['max']:.2f}\n")
        f.write("\n")
        
        # 5. Thống kê về các biến phân loại
        f.write("5. THỐNG KÊ CÁC BIẾN PHÂN LOẠI\n")
        f.write("-------------------\n")
        categorical_cols = df_after.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            value_counts = df_after[col].value_counts()
            f.write(f"\n{col}:\n")
            for value, count in value_counts.items():
                f.write(f"- {value}: {count} ({count/len(df_after)*100:.2f}%)\n")
        
        # 6. Các vấn đề đã phát hiện và xử lý
        f.write("\n6. CÁC VẤN ĐỀ ĐÃ PHÁT HIỆN VÀ XỬ LÝ\n")
        f.write("-------------------\n")
        f.write("- Loại bỏ các cột không cần thiết\n")
        f.write("- Chuẩn hóa các giá trị số từ chuỗi\n")
        f.write("- Chuyển đổi và chuẩn hóa mức giá về đơn vị tỷ đồng\n")
        f.write("- Tách địa điểm thành District và Province\n")
        f.write("- Chuẩn hóa trạng thái pháp lý và nội thất\n")
        f.write("- Loại bỏ các dòng có hơn 3 giá trị thiếu\n")

def main():

    # Cấu hình đường dẫn
    config = {
        'input_dir': 'data/raw',
        'output_dir': 'data/interim',
        'report_dir': 'reports'
    }

    try:
        # Đọc dữ liệu
        logger.info("Đang đọc dữ liệu...")
        df = load_csv(os.path.join(config['input_dir'], 'data.csv'))
        
        # Lưu DataFrame gốc để so sánh
        df_before = df.copy()
        df_before = df.rename(columns=column_rename_map)
        # Tiền xử lý dữ liệu
        logger.info("Đang tiền xử lý dữ liệu...")
        df_cleaned = preprocess_data(df)
        
        # Tạo báo cáo
        logger.info("Đang tạo báo cáo...")
        generate_preprocessing_report(df_before, df_cleaned, config['report_dir'])
        
        # Lưu dữ liệu đã xử lý
        os.makedirs(config['output_dir'], exist_ok=True)
        output_path = os.path.join(config['output_dir'], '01_preprocess.csv')
        df_cleaned.to_csv(output_path, index=False)
        
        logger.info(f"✅ Dữ liệu đã được xử lý và lưu vào '{output_path}'")
        logger.info(f"✅ Báo cáo đã được tạo trong thư mục '{config['report_dir']}'")
        
    except Exception as e:
        logger.error(f"Có lỗi xảy ra: {str(e)}")
        raise

if __name__ == "__main__":
    main()
