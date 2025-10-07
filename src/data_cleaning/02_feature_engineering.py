import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import os
from utils.load_data import load_csv, load_json

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_wiki_data(wiki_data: Dict) -> bool:
    """
    Kiểm tra tính hợp lệ của dữ liệu wiki.
    
    Parameters:
    -----------
    wiki_data : Dict
        Dữ liệu wiki cần kiểm tra
        
    Returns:
    --------
    bool
        True nếu dữ liệu hợp lệ, False nếu không
    """
    if not isinstance(wiki_data, dict):
        logger.error("Dữ liệu wiki phải là dictionary")
        return False
        
    required_fields = ['number_people', 'area', 'distribute', 'communes']
    
    for province, districts in wiki_data.items():
        for district, info in districts.items():
            if not all(field in info for field in required_fields):
                logger.error(f"Thiếu trường bắt buộc trong dữ liệu của {province} - {district}")
                return False
                
            # Chuyển đổi giá trị số trước khi kiểm tra
            try:
                number_people = convert_number_string(info['number_people'])
                area = convert_number_string(info['area'])
            except Exception as e:
                logger.error(f"Không thể chuyển đổi số ở {province} - {district}: {e}")
                return False
            # Chấp nhận giá trị dân số bằng 0 cho các khu vực đặc biệt
            if number_people < 0:
                logger.error(f"Số dân không hợp lệ trong dữ liệu của {province} - {district}")
                return False
            if area <= 0:
                logger.error(f"Diện tích không hợp lệ trong dữ liệu của {province} - {district}")
                return False
    return True

def convert_number_string(value: str) -> float:
    """
    Chuyển đổi chuỗi số từ định dạng wiki sang float.
    Ví dụ: "56.370" -> 56370.0, "1.225,2" -> 1225.2
    
    Parameters:
    -----------
    value : str
        Chuỗi số cần chuyển đổi
        
    Returns:
    --------
    float
        Giá trị số đã chuyển đổi
    """
    if not isinstance(value, str):
        return float(value)
        
    # Loại bỏ khoảng trắng
    value = value.strip()
    
    # Nếu có dấu phẩy làm dấu phân cách thập phân
    if ',' in value:
        # Thay thế dấu chấm làm dấu phân cách hàng nghìn
        value = value.replace('.', '')
        # Thay thế dấu phẩy làm dấu chấm
        value = value.replace(',', '.')
    else:
        # Nếu chỉ có dấu chấm, loại bỏ nó
        value = value.replace('.', '')
    
    return float(value)

def load_gdp_data(file_path: str) -> pd.DataFrame:
    """
    Đọc và xử lý dữ liệu GDP của các tỉnh/thành phố.
    
    Parameters:
    -----------
    file_path : str
        Đường dẫn đến file GDP data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame chứa dữ liệu GDP đã được xử lý
    """
    df_gdp = pd.read_csv(file_path)
    # Chuẩn hóa tên tỉnh/thành phố
    df_gdp['Province'] = df_gdp['Tên tỉnh, thành phố'].str.strip().str.lower()
    # Chọn và đổi tên các cột cần thiết
    df_gdp = df_gdp[['Province', 'Tổng GRDP (tỉ USD)']]
    df_gdp.columns = ['Province',  'GDP_USD']
    return df_gdp

def flatten_wiki_data(wiki_data: Dict) -> pd.DataFrame:
    """
    Chuyển đổi dữ liệu wiki từ dạng nested dictionary sang DataFrame.
    
    Parameters:
    -----------
    wiki_data : Dict
        Dữ liệu wiki dạng nested dictionary
        
    Returns:
    --------
    pd.DataFrame
        DataFrame chứa dữ liệu wiki đã được làm phẳng
    """
    rows = []
    for province, districts in wiki_data.items():
        for district, info in districts.items():
            # Chuyển đổi các giá trị số
            area = convert_number_string(info.get('area', '0'))
            population = convert_number_string(info.get('number_people', '0'))
            communes = convert_number_string(info.get('communes', '0'))
            
            rows.append({
                'Province': province.strip().lower(),
                'District': district.strip().lower(),
                'Population': population,
                'Area': area,
                'CommuneCount': communes,
                'Distribute': info.get('distribute', None)
            })
    return pd.DataFrame(rows)

def get_district_corrections() -> Dict[str, str]:
    """
    Trả về dictionary ánh xạ các tên quận/huyện đặc biệt.
    
    Returns:
    --------
    Dict[str, str]
        Dictionary ánh xạ tên quận/huyện cũ sang tên mới
    """
    return {
        "yên dũng": "bắc giang",      # Đã sáp nhập vào thành phố Bắc Giang
        "đông sơn": "thanh hóa",      # Đã sáp nhập vào thành phố Thanh Hóa
        "quận 9": "thủ đức",
        "quận 2": "thủ đức",
        "long điền": "long đất",
        "huế": "phú xuân"
    }

def merge_datasets(
    df_main: pd.DataFrame,
    wiki_data: Dict,
    gdp_data: pd.DataFrame,
    save_path: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge dữ liệu chính với dữ liệu wiki và GDP.
    
    Parameters:
    -----------
    df_main : pd.DataFrame
        DataFrame chính cần merge
    wiki_data : Dict
        Dữ liệu wiki cần merge
    gdp_data : pd.DataFrame
        Dữ liệu GDP cần merge
    save_path : Optional[str]
        Đường dẫn lưu kết quả
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        DataFrame đã merge và DataFrame chứa các bản ghi không merge được
    """
    # Kiểm tra dữ liệu wiki
    if not validate_wiki_data(wiki_data):
        raise ValueError("Dữ liệu wiki không hợp lệ")
    
    # Chuyển đổi dữ liệu wiki
    df_wiki = flatten_wiki_data(wiki_data)
    
    # Đổi tên cột Area thành DistrictArea trong df_wiki
    df_wiki.rename(columns={'Area': 'DistrictArea'}, inplace=True)
    
    # Chuẩn hóa tên
    df_main['Province'] = df_main['Province'].str.strip().str.lower()
    df_main['District'] = df_main['District'].str.strip().str.lower()
    
    # Ánh xạ các tên District đặc biệt
    district_corrections = get_district_corrections()
    df_main['District'] = df_main['District'].replace(district_corrections)
    
    # Merge với wiki data
    df_merged = pd.merge(df_main, df_wiki, on=['Province', 'District'], how='left')
    
    # Merge với GDP data
    df_merged = pd.merge(df_merged, gdp_data, on='Province', how='left')
    
    # Tách các bản ghi không merge được
    not_merged = df_merged[df_merged['Population'].isna()]
    
    # Log kết quả
    if not not_merged.empty:
        logger.warning(f"Có {len(not_merged)} bản ghi không merge được")
        logger.warning("Các quận/huyện không tìm thấy trong dữ liệu wiki:")
        for _, row in not_merged[['Province', 'District']].drop_duplicates().iterrows():
            logger.warning(f"- {row['Province']} - {row['District']}")
    else:
        logger.info("Tất cả các bản ghi đã được merge thành công")
    
    # Lưu kết quả nếu có đường dẫn
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df_merged.to_csv(save_path, index=False)
        logger.info(f"Đã lưu kết quả tại: {save_path}")
    
    return df_merged, not_merged

def normalize_distribution(df: pd.DataFrame) -> pd.DataFrame:
    # Log-transform trực tiếp cho Price, Area
    df['Price'] = np.log1p(df['Price'])
    df['Area'] = np.log1p(df['Area'])

    # Log1p trực tiếp cho Bedrooms, Bathrooms, Floors
    for col in ['Bedrooms', 'Bathrooms', 'Floors']:
        df[col] = np.log1p(df[col])

    return df

def main():
    # Cấu hình
    config = {
        'input': 'data/interim/01_preprocess.csv',
        'wiki_data': 'data/raw/wiki.json',
        'gdp_data': 'data/raw/gdp_provinces.csv',
        'output_dir': 'data/interim',
        'report_dir': 'reports'
    }
    
    try:
        # Đọc dữ liệu
        logger.info("Đang đọc dữ liệu...")
        df = load_csv(config['input'])
        wiki_data = load_json(config['wiki_data'])
        gdp_data = load_gdp_data(config['gdp_data'])
        
        # Merge dữ liệu
        logger.info("Đang merge dữ liệu...")
        save_path = os.path.join(config['output_dir'], '02_feature_engineering.csv')
        df_merged, not_merged = merge_datasets(df, wiki_data, gdp_data, save_path)
        
        # Lưu báo cáo
        report_path = os.path.join(config['report_dir'], '02_merge_analysis.txt')
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"BÁO CÁO MERGE DỮ LIỆU\n")
            f.write(f"===================\n\n")
            f.write(f"Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("1. Thống kê:\n")
            f.write(f"   - Tổng số bản ghi: {len(df)}\n")
            f.write(f"   - Số bản ghi merge thành công: {len(df_merged) - len(not_merged)}\n")
            f.write(f"   - Số bản ghi không merge được: {len(not_merged)}\n\n")
            
            f.write("2. Các quận/huyện không tìm thấy trong dữ liệu wiki:\n")
            for _, row in not_merged[['Province', 'District']].drop_duplicates().iterrows():
                f.write(f"   - {row['Province']} - {row['District']}\n")
        
        logger.info("Hoàn thành xử lý dữ liệu!")
        
        # Chuẩn hóa phân phối, ghi đè lên cột gốc
        df_merged = normalize_distribution(df_merged)

        # Lưu lại
        df_merged.to_csv('data/interim/03_distribution_normalization.csv', index=False)
        
    except Exception as e:
        logger.error(f"Có lỗi xảy ra: {str(e)}")
        raise

if __name__ == "__main__":
    main()
