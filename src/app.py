import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime
from utils.load_data import load_csv
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Cấu hình trang
st.set_page_config(
    page_title="Dự đoán giá nhà",
    page_icon="🏠",
    layout="wide"
)

# Cấu hình debug
DEBUG = False  # Set to True to show debug information

# Tiêu đề
st.title("🏠 Dự đoán giá nhà đất")
st.markdown("---")

# Hàm debug print
def debug_print(message: str, data=None):
    """In thông tin debug nếu DEBUG=True"""
    if DEBUG:
        st.write(message)
        if data is not None:
            st.write(data)

# Hàm chuyển đổi giá trị LegalStatus
def convert_legal_status_to_label(status: str) -> int:
    """
    Chuyển đổi giá trị LegalStatus từ text sang số theo label encoding
    """
    status_map = {
        "Có": 1,
        "Không": 0
    }
    return status_map.get(status, 0)

# Hàm chuyển đổi giá trị Furnishing
def convert_furnishing_to_label(furnishing: str) -> int:
    """
    Chuyển đổi giá trị Furnishing từ text sang số theo label encoding
    """
    furnishing_map = {
        "Có": 1,
        "Không": 0
    }
    return furnishing_map.get(furnishing, 0)

# Hàm chuẩn hóa phân phối
def normalize_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Áp dụng log1p cho các cột lệch phân phối
    """
    # Áp dụng log1p cho các cột lệch phân phối
    for col in ['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'AccessWidth', 'FacadeWidth']:
        df[col] = np.log1p(df[col])
    return df

# Hàm xử lý đặc trưng cho Linear Regression
def process_linear_regression_features(df: pd.DataFrame, scalers: dict) -> pd.DataFrame:
    """Xử lý đặc trưng cho Linear Regression"""
    df_filtered = df.copy()
    debug_print("Linear Regression - Input data:", df_filtered)
    
    # Chuẩn hóa biến số
    numerical_columns = ['Area', 'Bedrooms', 'Bathrooms', 'Floors', 
                        'AccessWidth', 'FacadeWidth', 'Distribute', 'GDP_USD']
    for col in numerical_columns:
        if col in df_filtered.columns and col in scalers:
            df_filtered[col] = scalers[col].transform(df_filtered[[col]])
            debug_print(f"Linear Regression - After scaling {col}:", df_filtered[col].values)
    
    # Mã hóa biến phân loại
    categorical_columns = ['LegalStatus', 'Furnishing']
    for col in categorical_columns:
        if col in df_filtered.columns:
            # Chuyển đổi giá trị số thành 'yes'/'no'
            df_filtered[col] = df_filtered[col].map({1: 'yes', 0: 'no'})
            dummies = pd.get_dummies(df_filtered[col], prefix=col)
            df_filtered = pd.concat([df_filtered.drop(col, axis=1), dummies], axis=1)
            debug_print(f"Linear Regression - After encoding {col}:", df_filtered)
    
    # Đảm bảo thứ tự cột
    expected_columns = [
        'Area', 'Bedrooms', 'Bathrooms', 'Floors', 'AccessWidth', 
        'FacadeWidth', 'Distribute', 'GDP_USD',
        'LegalStatus_no', 'LegalStatus_yes',
        'Furnishing_no', 'Furnishing_yes'
    ]
    
    # Đảm bảo tất cả các cột cần thiết đều có mặt
    for col in expected_columns:
        if col not in df_filtered.columns:
            df_filtered[col] = 0
    
    df_filtered = df_filtered[expected_columns]
    debug_print("Linear Regression - Final processed data:", df_filtered)
    return df_filtered

# Hàm xử lý đặc trưng cho KNN
def process_knn_features(df: pd.DataFrame, scalers: dict) -> pd.DataFrame:
    """Xử lý đặc trưng cho KNN"""
    df_filtered = df.copy()
    debug_print("KNN - Input data:", df_filtered)
    
    # Chuẩn hóa biến số
    numerical_columns = ['Area', 'Bedrooms', 'Bathrooms', 'Floors', 
                        'AccessWidth', 'FacadeWidth', 'Distribute', 'GDP_USD']
    for col in numerical_columns:
        if col in df_filtered.columns and col in scalers:
            df_filtered[col] = scalers[col].transform(df_filtered[[col]])
            debug_print(f"KNN - After scaling {col}:", df_filtered[col].values)
    
    # Mã hóa biến phân loại
    categorical_columns = ['LegalStatus', 'Furnishing']
    for col in categorical_columns:
        if col in df_filtered.columns:
            # Chuyển đổi giá trị số thành 'yes'/'no'
            df_filtered[col] = df_filtered[col].map({1: 'yes', 0: 'no'})
            dummies = pd.get_dummies(df_filtered[col], prefix=col)
            df_filtered = pd.concat([df_filtered.drop(col, axis=1), dummies], axis=1)
            debug_print(f"KNN - After encoding {col}:", df_filtered)
    
    # Đảm bảo thứ tự cột
    expected_columns = [
        'Area', 'Bedrooms', 'Bathrooms', 'Floors', 'AccessWidth', 
        'FacadeWidth', 'Distribute', 'GDP_USD',
        'LegalStatus_no', 'LegalStatus_yes',
        'Furnishing_no', 'Furnishing_yes'
    ]
    
    # Đảm bảo tất cả các cột cần thiết đều có mặt
    for col in expected_columns:
        if col not in df_filtered.columns:
            df_filtered[col] = 0
    
    df_filtered = df_filtered[expected_columns]
    debug_print("KNN - Final processed data:", df_filtered)
    return df_filtered

# Hàm xử lý đặc trưng cho XGBoost
def process_xgboost_features(df: pd.DataFrame, scalers: dict) -> pd.DataFrame:
    """Xử lý đặc trưng cho XGBoost"""
    df_filtered = df.copy()
    debug_print("XGBoost - Input data:", df_filtered)
    
    # Chuẩn hóa biến số
    numerical_columns = ['Area', 'Bedrooms', 'Bathrooms', 'Floors', 
                        'AccessWidth', 'FacadeWidth', 'Distribute', 'GDP_USD']
    for col in numerical_columns:
        if col in df_filtered.columns and col in scalers:
            df_filtered[col] = scalers[col].transform(df_filtered[[col]])
            debug_print(f"XGBoost - After scaling {col}:", df_filtered[col].values)
    
    # Mã hóa biến phân loại
    categorical_columns = ['LegalStatus', 'Furnishing']
    for col in categorical_columns:
        if col in df_filtered.columns:
            df_filtered[col] = df_filtered[col].astype('category').cat.codes
            debug_print(f"XGBoost - After encoding {col}:", df_filtered[col].values)
    
    # Đảm bảo thứ tự cột
    expected_columns = [
        'Area', 'Bedrooms', 'Bathrooms', 'Floors', 'AccessWidth', 
        'FacadeWidth', 'LegalStatus', 'Furnishing', 'Distribute', 'GDP_USD'
    ]
    
    df_filtered = df_filtered[expected_columns]
    debug_print("XGBoost - Final processed data:", df_filtered)
    return df_filtered

# Load dữ liệu địa lý
@st.cache_resource
def load_location_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    wiki_path = os.path.join(base_dir, 'data', 'raw', 'wiki.json')
    gdp_path = os.path.join(base_dir, 'data', 'raw', 'gdp_provinces.csv')
    
    if not os.path.exists(wiki_path):
        st.error(f"Không tìm thấy file wiki.json tại: {wiki_path}")
        return {}
    
    # Load dữ liệu wiki
    with open(wiki_path, 'r', encoding='utf-8') as f:
        location_data = json.load(f)
    
    # Load và merge GDP data
    if os.path.exists(gdp_path):
        try:
            # Load GDP data
            gdp_data = pd.read_csv(gdp_path)
            debug_print("GDP Data loaded:", gdp_data.head())  # Debug info
            
            # Đổi tên cột cho phù hợp
            gdp_data = gdp_data.rename(columns={
                'Tên tỉnh, thành phố': 'Province',
                'Tổng GRDP (tỉ USD)': 'GDP_USD'
            })
            
            # Chuyển tên tỉnh/thành phố trong GDP data về chữ thường
            gdp_data['Province'] = gdp_data['Province'].str.lower()
            
            # Debug info
            debug_print("GDP Data after renaming:", gdp_data.head())
            debug_print("Available provinces in GDP data:", gdp_data['Province'].unique())
            debug_print("Available provinces in location data:", list(location_data.keys()))
            
            # Merge GDP data với location data
            for province in location_data:
                province_gdp = gdp_data[gdp_data['Province'] == province.lower()]
                if not province_gdp.empty:
                    gdp_value = float(province_gdp['GDP_USD'].values[0])
                    debug_print(f"Found GDP for {province}: {gdp_value}")  # Debug info
                    for district in location_data[province]:
                        location_data[province][district]['gdp_usd'] = gdp_value
                else:
                    st.warning(f"Không tìm thấy GDP cho tỉnh/thành phố: {province}")
        except Exception as e:
            st.error(f"Lỗi khi xử lý GDP data: {str(e)}")
            st.error("Chi tiết lỗi:")
            st.error(str(e.__class__.__name__))
            st.error(str(e))
    else:
        st.error(f"Không tìm thấy file gdp_provinces.csv tại: {gdp_path}")
    
    return location_data

# Load model và scaler
@st.cache_resource
def load_models_and_scalers():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    
    if not os.path.exists(models_dir):
        st.error(f"Không tìm thấy thư mục models tại: {models_dir}")
        return {}, {}
    
    # Dictionary để lưu các model
    models = {}
    
    # Thông tin về các model
    model_info = {
        'linear_regression_model.joblib': {
            'name': 'Linear Regression',
            'scaler_file': 'scalers_linear_regression.joblib',
            'description': 'Linear Regression (R2: 0.518)'
        },
        'knn_model.joblib': {
            'name': 'KNN',
            'scaler_file': 'scalers_knn.joblib',
            'description': 'KNN (R2: 0.612)'
        },
        'xgboost_model.joblib': {
            'name': 'XGBoost',
            'scaler_file': 'scalers_xgboost.joblib',
            'description': 'XGBoost (R2: 0.781)'
        }
    }
    
    # Load tất cả các model và scaler
    for model_file, info in model_info.items():
        model_path = os.path.join(models_dir, model_file)
        scaler_path = os.path.join(models_dir, info['scaler_file'])
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                model = joblib.load(model_path)
                scalers = joblib.load(scaler_path)
                models[info['name']] = {
                    'model': model,
                    'scalers': scalers,
                    'description': info['description']
                }
            except Exception as e:
                st.warning(f"Không thể tải model hoặc scaler cho {info['name']}: {str(e)}")
    
    return models

try:
    models = load_models_and_scalers()
    location_data = load_location_data()
    if not models:
        st.error("Không tìm thấy model nào để dự đoán.")
        st.stop()
except Exception as e:
    st.error("Không thể tải mô hình hoặc dữ liệu địa lý. Vui lòng kiểm tra lại đường dẫn và file.")
    st.stop()

# Chọn model
st.subheader("Chọn mô hình dự đoán")
model_names = list(models.keys())
selected_model_name = st.selectbox(
    "Mô hình",
    model_names,
    index=0,
    help="Chọn mô hình để dự đoán giá nhà"
)

# Hiển thị thông tin về model được chọn
if selected_model_name:
    model_info = models[selected_model_name]
    st.info(f"""
    **Thông tin mô hình:**
    - {model_info['description']}
 
    """)

# Chọn địa chỉ
st.subheader("Địa chỉ")
if not location_data:
    st.error("Không có dữ liệu địa lý. Vui lòng kiểm tra lại file wiki.json")
    st.stop()

provinces = list(location_data.keys())
selected_province = st.selectbox(
    "Tỉnh/Thành phố",
    provinces,
    help="Chọn tỉnh/thành phố để xem danh sách quận/huyện và thông tin chi tiết"
)

districts = list(location_data[selected_province].keys())
selected_district = st.selectbox(
    "Quận/Huyện",
    districts,
    help="Thông tin sẽ được cập nhật tự động khi chọn quận/huyện"
)

# Lấy thông tin về quận/huyện được chọn
district_info = location_data[selected_province][selected_district]
commune_density = float(district_info['distribute'].replace('.', ''))
commune_count = int(district_info['communes'])

# Hiển thị thông tin về quận/huyện
st.info(f"""
**Thông tin {selected_district.title()}:**
- Mật độ dân số: {commune_density:,.0f} người/km²
- Số phường/xã: {commune_count}
- Diện tích: {district_info['area']} km²
- Dân số: {district_info['number_people']} người
""")

# Tạo form nhập liệu
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Thông tin cơ bản")
        area = st.number_input("Diện tích (m²)", min_value=0.0, value=100.0)
        bedrooms = st.number_input("Số phòng ngủ", min_value=0, value=2)
        bathrooms = st.number_input("Số phòng tắm", min_value=0, value=1)
        floors = st.number_input("Số tầng", min_value=0, value=1)
        
    with col2:
        st.subheader("Thông tin bổ sung")
        access_width = st.number_input("Đường vào (m)", min_value=0.0, value=4.0)
        facade_width = st.number_input("Mặt tiền (m)", min_value=0.0, value=5.0)
        
    st.subheader("Trạng thái")
    col3, col4 = st.columns(2)
    
    with col3:
        legal_status = st.radio("Pháp lý", ["Có", "Không"])
        
    with col4:
        furnishing = st.radio("Nội thất", ["Có", "Không"])
    
    submit_button = st.form_submit_button("Dự đoán giá")

# Xử lý dự đoán
if submit_button:
    try:
        # Debug info trước khi tạo input data
        debug_print("Selected province:", selected_province)
        debug_print("Selected district:", selected_district)
        debug_print("District info before processing:", location_data[selected_province][selected_district])
        
        # Tạo DataFrame từ input
        gdp_value = location_data[selected_province][selected_district].get('gdp_usd', 0)
        debug_print("GDP value to be used:", gdp_value)
        
        input_data = pd.DataFrame({
            'Area': [area],
            'Bedrooms': [bedrooms],
            'Bathrooms': [bathrooms],
            'Floors': [floors],
            'AccessWidth': [access_width],
            'FacadeWidth': [facade_width],
            'LegalStatus': [convert_legal_status_to_label(legal_status)],
            'Furnishing': [convert_furnishing_to_label(furnishing)],
            'Distribute': [commune_density],
            'GDP_USD': [gdp_value]
        })
        
        # Debug info
        debug_print("Input data before processing:", input_data)
        
        # Lấy model và scaler tương ứng
        model_info = models[selected_model_name]
        model = model_info['model']
        scalers = model_info['scalers']
        
        # Áp dụng log1p transformation cho các cột số
        input_data = normalize_distribution(input_data)
        debug_print("After log1p transformation:", input_data)
        
        # Xử lý đặc trưng theo từng thuật toán
        if selected_model_name == 'Linear Regression':
            input_data = process_linear_regression_features(input_data, scalers)
        elif selected_model_name == 'KNN':
            input_data = process_knn_features(input_data, scalers)
        elif selected_model_name == 'XGBoost':
            input_data = process_xgboost_features(input_data, scalers)
        
        # Debug info
        debug_print("Final processed data:", input_data)
        
        # Dự đoán
        prediction = model.predict(input_data)[0]
        
        # Chuyển đổi giá trị dự đoán từ log scale về giá trị thực
        prediction = np.expm1(prediction)
        
        # Hiển thị kết quả
        st.markdown("---")
        st.subheader("Kết quả dự đoán")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Giá dự đoán", f"{prediction:,.2f} tỷ đồng")
        
        with col2:
            st.metric("Giá/m²", f"{(prediction * 1000 / area):,.0f} triệu đồng/m²")
        
        # with col3:
        #     st.metric("Thời gian dự đoán", datetime.now().strftime("%H:%M:%S"))
        
        # Hiển thị thông tin chi tiết
        st.markdown("### Thông tin chi tiết")
        st.markdown(f"""
        - Mô hình sử dụng: {selected_model_name}
        - Địa chỉ: {selected_district.title()}, {selected_province.title()}
        - Diện tích: {area:,.1f} m²
        - Số phòng ngủ: {bedrooms}
        - Số phòng tắm: {bathrooms}
        - Số tầng: {floors}
        - Đường vào: {access_width:,.1f} m
        - Mặt tiền: {facade_width:,.1f} m
        - Mật độ dân số: {commune_density:,.0f} người/km²
        - GDP (USD): {gdp_value:,.2f} tỷ USD
        - Pháp lý: {legal_status}
        - Nội thất: {furnishing}
        """)
        
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi dự đoán: {str(e)}")
        st.error("Chi tiết lỗi:")
        st.error(str(e.__class__.__name__))
        st.error(str(e))

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Được phát triển bởi HM&KPDL Team</p>
    <p>© 2024 - Phiên bản 1.0</p>
</div>
""", unsafe_allow_html=True) 