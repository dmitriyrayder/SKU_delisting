import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    st.warning("⚠️ Prophet не установлен. Прогнозы будут недоступны.")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from io import BytesIO

st.set_page_config(page_title="Анализ товаров", layout="wide")

# Инициализация session_state
if 'run_analysis' not in st.session_state:
    st.session_state.run_analysis = False
if 'loaded_data' not in st.session_state:
    st.session_state.loaded_data = None
if 'data_source_type' not in st.session_state:
    st.session_state.data_source_type = None

st.title("🔍 Анализ товаров: определение кандидатов на снятие")

# === НАСТРОЙКИ ===
with st.sidebar:
    st.header("⚙️ Настройки")
    TOP_N = st.slider("Количество топ-артикулов для Prophet", 10, 50, 20)

    st.subheader("🎯 Критерии снятия")
    zero_weeks_threshold = st.slider("Недель подряд без продаж", 8, 20, 12)
    min_total_sales = st.slider("Минимальный объем продаж", 1, 50, 5)
    max_store_ratio = st.slider("Макс. доля магазинов без продаж (%)", 70, 95, 85, 5) / 100

    st.subheader("🤖 Модель ML")
    use_balanced_model = st.checkbox("Использовать балансировку классов", value=True)
    final_threshold = st.slider("Финальный порог для снятия (%)", 50, 90, 70, 5) / 100

    st.divider()

    # Кнопка очистки кеша
    if st.button("🔄 Очистить кеш данных"):
        st.session_state.loaded_data = None
        st.cache_data.clear()
        st.success("Кеш очищен!")
        st.rerun()

# === ЗАГРУЗКА ДАННЫХ ===
st.header("📁 Загрузка данных")
st.info("💡 Формат: дата, артикул, количество, магазин, название")

# Выбор источника данных
data_source = st.radio(
    "Выберите источник данных:",
    ["Google Sheets", "Локальный файл"],
    horizontal=True
)

uploaded_file = None
sheets_url = None

if data_source == "Локальный файл":
    uploaded_file = st.file_uploader("Выберите Excel файл", type=['xlsx', 'xls'])
else:
    sheets_url = st.text_input(
        "Ссылка на Google Sheets:",
        value="https://docs.google.com/spreadsheets/d/1lJLON5N_EKQ5ICv0Pprp5DamP1tNAhBIph4uEoWC04Q/edit?gid=64159818#gid=64159818",
        help="Таблица должна иметь публичный доступ"
    )

# === КЕШИРОВАННЫЕ ФУНКЦИИ ЗАГРУЗКИ ===
@st.cache_data(show_spinner="🔄 Загрузка данных из Google Sheets...")
def _fetch_google_sheets_data(sheets_url):
    """Кешированная загрузка сырых данных из Google Sheets"""
    import re

    # Извлекаем spreadsheet ID
    spreadsheet_match = re.search(r'/spreadsheets/d/([a-zA-Z0-9-_]+)', sheets_url)
    if not spreadsheet_match:
        raise ValueError("Неверный формат ссылки на Google Sheets")

    spreadsheet_id = spreadsheet_match.group(1)

    # Извлекаем GID (ID листа)
    gid_match = re.search(r'[#&]gid=([0-9]+)', sheets_url)
    gid = gid_match.group(1) if gid_match else '0'

    # Формируем URL для экспорта в Excel формате
    export_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=xlsx&gid={gid}"

    # Загружаем данные
    df = pd.read_excel(export_url, nrows=100000)
    return df

@st.cache_data(show_spinner="🔄 Загрузка файла...")
def _load_excel_file(file_bytes, sheet_name):
    """Кешированная загрузка Excel файла"""
    from io import BytesIO
    df = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet_name, nrows=100000)
    return df

def load_and_process_data(uploaded_file):
    if uploaded_file is None:
        st.info("👆 Загрузите Excel файл для начала работы")
        return None, False

    try:
        file_size = len(uploaded_file.read())
        uploaded_file.seek(0)

        if file_size > 50 * 1024 * 1024:
            st.error("❌ Файл слишком большой. Максимум: 50MB")
            return None, False

        # Определяем листы
        file_bytes = uploaded_file.read()
        uploaded_file.seek(0)
        excel_file = pd.ExcelFile(uploaded_file)
        selected_sheet = st.selectbox("Выберите лист:", excel_file.sheet_names) if len(excel_file.sheet_names) > 1 else excel_file.sheet_names[0]

        # Используем кешированную загрузку
        df = _load_excel_file(file_bytes, selected_sheet)
        if len(df) == 100000:
            st.warning("⚠️ Файл обрезан до 100,000 строк")
        
        st.success(f"✅ Загружено {len(df)} строк")
        
        # Сопоставление колонок
        available_cols = list(df.columns)
        col1, col2 = st.columns(2)
        
        with col1:
            date_col = st.selectbox("Дата:", available_cols, index=next((i for i, col in enumerate(available_cols) if any(word in col.lower() for word in ['дат', 'date'])), 0))
            art_col = st.selectbox("Артикул:", available_cols, index=next((i for i, col in enumerate(available_cols) if any(word in col.lower() for word in ['арт', 'art'])), 0))
            qty_col = st.selectbox("Количество:", available_cols, index=next((i for i, col in enumerate(available_cols) if any(word in col.lower() for word in ['кол', 'qty', 'количество'])), 0))
        
        with col2:
            magazin_col = st.selectbox("Магазин:", available_cols, index=next((i for i, col in enumerate(available_cols) if any(word in col.lower() for word in ['маг', 'magazin', 'магазин'])), 0))
            name_col = st.selectbox("Название:", available_cols, index=next((i for i, col in enumerate(available_cols) if any(word in col.lower() for word in ['назв', 'name', 'название'])), 0))
            segment_col = st.selectbox("Сегмент (опционально):", ['Без сегментации'] + available_cols)
        
        # Переименование колонок
        column_mapping = {date_col: 'Data', art_col: 'Art', qty_col: 'Qty', magazin_col: 'Magazin', name_col: 'Name'}
        if segment_col != 'Без сегментации':
            column_mapping[segment_col] = 'Segment'
        
        df = df.rename(columns=column_mapping)
        
        # Проверка обязательных колонок
        required_cols = ['Data', 'Art', 'Qty', 'Magazin', 'Name']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"❌ Отсутствуют колонки: {missing_cols}")
            return None, False
        
        # Фильтрация по сегменту
        if 'Segment' in df.columns:
            st.subheader("🎯 Выбор сегмента")
            unique_segments = sorted(df['Segment'].dropna().unique())
            selected_segment = st.selectbox("Сегмент:", ['Все сегменты'] + list(unique_segments))
            
            if selected_segment != 'Все сегменты':
                df = df[df['Segment'] == selected_segment].copy()
                st.success(f"✅ Выбран сегмент: {selected_segment}")
        
        with st.expander("📊 Предварительный просмотр"):
            st.dataframe(df.head())
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Записей", len(df))
            with col2: st.metric("Артикулов", df['Art'].nunique())
            with col3:
                try:
                    date_min = pd.to_datetime(df['Data'], errors='coerce').min()
                    date_max = pd.to_datetime(df['Data'], errors='coerce').max()
                    st.metric("Период", f"{date_min.strftime('%Y-%m-%d')} - {date_max.strftime('%Y-%m-%d')}")
                except:
                    st.metric("Период", "Ошибка дат")
        
        return df, True
        
    except Exception as e:
        st.error(f"❌ Ошибка загрузки: {str(e)}")
        return None, False

def load_from_google_sheets(sheets_url):
    """Загрузка данных из публичной Google Sheets таблицы"""
    if not sheets_url or sheets_url.strip() == "":
        st.info("👆 Введите ссылку на Google Sheets")
        return None, False

    try:
        # Используем кешированную загрузку данных
        df = _fetch_google_sheets_data(sheets_url)

        if len(df) == 100000:
            st.warning("⚠️ Файл обрезан до 100,000 строк")

        st.success(f"✅ Загружено {len(df)} строк из Google Sheets")

        # Сопоставление колонок (идентично load_and_process_data)
        available_cols = list(df.columns)
        col1, col2 = st.columns(2)

        with col1:
            date_col = st.selectbox("Дата:", available_cols, index=next((i for i, col in enumerate(available_cols) if any(word in col.lower() for word in ['дат', 'date'])), 0), key="gs_date")
            art_col = st.selectbox("Артикул:", available_cols, index=next((i for i, col in enumerate(available_cols) if any(word in col.lower() for word in ['арт', 'art'])), 0), key="gs_art")
            qty_col = st.selectbox("Количество:", available_cols, index=next((i for i, col in enumerate(available_cols) if any(word in col.lower() for word in ['кол', 'qty', 'количество'])), 0), key="gs_qty")

        with col2:
            magazin_col = st.selectbox("Магазин:", available_cols, index=next((i for i, col in enumerate(available_cols) if any(word in col.lower() for word in ['маг', 'magazin', 'магазин'])), 0), key="gs_magazin")
            name_col = st.selectbox("Название:", available_cols, index=next((i for i, col in enumerate(available_cols) if any(word in col.lower() for word in ['назв', 'name', 'название'])), 0), key="gs_name")
            segment_col = st.selectbox("Сегмент (опционально):", ['Без сегментации'] + available_cols, key="gs_segment")

        # Переименование колонок
        column_mapping = {date_col: 'Data', art_col: 'Art', qty_col: 'Qty', magazin_col: 'Magazin', name_col: 'Name'}
        if segment_col != 'Без сегментации':
            column_mapping[segment_col] = 'Segment'

        df = df.rename(columns=column_mapping)

        # Проверка обязательных колонок
        required_cols = ['Data', 'Art', 'Qty', 'Magazin', 'Name']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"❌ Отсутствуют колонки: {missing_cols}")
            return None, False

        # Фильтрация по сегменту
        if 'Segment' in df.columns:
            st.subheader("🎯 Выбор сегмента")
            unique_segments = sorted(df['Segment'].dropna().unique())
            selected_segment = st.selectbox("Сегмент:", ['Все сегменты'] + list(unique_segments), key="gs_segment_filter")

            if selected_segment != 'Все сегменты':
                df = df[df['Segment'] == selected_segment].copy()
                st.success(f"✅ Выбран сегмент: {selected_segment}")

        with st.expander("📊 Предварительный просмотр"):
            st.dataframe(df.head())
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Записей", len(df))
            with col2: st.metric("Артикулов", df['Art'].nunique())
            with col3:
                try:
                    date_min = pd.to_datetime(df['Data'], errors='coerce').min()
                    date_max = pd.to_datetime(df['Data'], errors='coerce').max()
                    st.metric("Период", f"{date_min.strftime('%Y-%m-%d')} - {date_max.strftime('%Y-%m-%d')}")
                except:
                    st.metric("Период", "Ошибка дат")

        return df, True

    except Exception as e:
        st.error(f"❌ Ошибка загрузки из Google Sheets: {str(e)}")
        st.info("💡 Убедитесь, что таблица имеет публичный доступ")
        return None, False

# Загрузка данных в зависимости от выбранного источника с использованием session_state
# Проверяем, изменился ли источник данных
if st.session_state.data_source_type != data_source:
    st.session_state.loaded_data = None  # Сбрасываем кеш при смене источника
    st.session_state.data_source_type = data_source

# Если данные уже загружены и источник не изменился, используем кешированные
if st.session_state.loaded_data is not None:
    df, data_loaded = st.session_state.loaded_data
    if data_loaded:
        st.info("ℹ️ Используются ранее загруженные данные")
else:
    # Загружаем новые данные
    if data_source == "Локальный файл":
        df, data_loaded = load_and_process_data(uploaded_file)
    else:
        df, data_loaded = load_from_google_sheets(sheets_url)

    # Сохраняем в session_state
    if data_loaded:
        st.session_state.loaded_data = (df, data_loaded)

if data_loaded:
    st.header("🚀 Запуск анализа")
    if st.button("▶️ НАЧАТЬ АНАЛИЗ", type="primary", use_container_width=True):
        st.session_state.run_analysis = True
    
    if not st.session_state.get('run_analysis', False):
        st.info("👆 Нажмите кнопку для запуска анализа")
        st.stop()
else:
    st.stop()

# === ОСНОВНАЯ ОБРАБОТКА ===
def process_data(df):
    with st.spinner("🔄 Обработка данных..."):
        # Очистка данных
        df['Data'] = pd.to_datetime(df['Data'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Data'])
        df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce').fillna(0)
        df = df[df['Qty'] >= 0]
        
        if len(df) == 0:
            st.error("❌ Нет валидных данных")
            st.stop()
        
        df['year_week'] = df['Data'].dt.strftime('%Y-%U')
        
        # Ограничение артикулов
        all_arts = df['Art'].unique()
        if len(all_arts) > 5000:
            st.warning("⚠️ Обрабатываем топ-5000 артикулов по продажам")
            top_arts = df.groupby('Art')['Qty'].sum().nlargest(5000).index
            all_arts = top_arts
            df = df[df['Art'].isin(all_arts)]
        
        # Агрегация по неделям
        weekly = df.groupby(['Art', 'year_week'])['Qty'].sum().reset_index()
        unique_weeks = sorted(df['year_week'].unique())
        all_weeks = pd.MultiIndex.from_product([all_arts, unique_weeks], names=['Art', 'year_week'])
        weekly = weekly.set_index(['Art', 'year_week']).reindex(all_weeks, fill_value=0).reset_index()
        
        return df, weekly, all_arts, unique_weeks

def calculate_abc_xyz_analysis(df):
    # ABC анализ
    abc_analysis = df.groupby('Art').agg({
        'Qty': ['sum', 'mean', 'std'],
        'Data': ['min', 'max']
    }).reset_index()
    
    abc_analysis.columns = ['Art', 'total_qty', 'avg_qty', 'std_qty', 'first_sale', 'last_sale']
    abc_analysis['days_in_catalog'] = (abc_analysis['last_sale'] - abc_analysis['first_sale']).dt.days + 1
    
    # ABC категории (исправлено: сортировка перед кумулятивным расчетом)
    abc_analysis = abc_analysis.sort_values('total_qty', ascending=False).reset_index(drop=True)
    abc_analysis['cum_qty'] = abc_analysis['total_qty'].cumsum()
    total_sum = abc_analysis['total_qty'].sum()
    abc_analysis['cum_qty_pct'] = abc_analysis['cum_qty'] / total_sum if total_sum > 0 else 0
    
    def get_abc_category(cum_pct):
        if cum_pct <= 0.8: return 'A'
        elif cum_pct <= 0.95: return 'B'
        else: return 'C'
    
    abc_analysis['abc_category'] = abc_analysis['cum_qty_pct'].apply(get_abc_category)
    
    # XYZ анализ (исправлено: обработка нулевых значений)
    abc_analysis['coefficient_variation'] = np.where(
        abc_analysis['avg_qty'] > 0,
        abc_analysis['std_qty'] / abc_analysis['avg_qty'],
        999  # Большое значение для товаров без продаж
    )
    
    def get_xyz_category(cv):
        if cv <= 0.1: return 'X'  # Стабильный спрос
        elif cv <= 0.25: return 'Y'  # Умеренно изменчивый
        else: return 'Z'  # Нестабильный спрос
    
    abc_analysis['xyz_category'] = abc_analysis['coefficient_variation'].apply(get_xyz_category)
    
    return abc_analysis

def calculate_features(weekly, df):
    def compute_features(group):
        sorted_group = group.sort_values('year_week')
        qty_series = sorted_group['Qty'].values
        
        if len(qty_series) == 0:
            return pd.Series({
                'ma_3': 0, 
                'ma_6': 0, 
                'consecutive_zeros': 0,
                'zero_weeks_12': 0, 
                'trend': 0
            })
        
        # Скользящие средние
        qty_series_pd = pd.Series(qty_series)
        ma_3 = qty_series_pd.rolling(3, min_periods=1).mean().iloc[-1]
        ma_6 = qty_series_pd.rolling(6, min_periods=1).mean().iloc[-1]
        
        # Последовательные нули с конца
        consecutive_zeros = 0
        for val in reversed(qty_series):
            if val == 0: 
                consecutive_zeros += 1
            else: 
                break
        
        # Нули за последние 12 недель
        zero_weeks_12 = int(np.sum(qty_series[-12:] == 0)) if len(qty_series) >= 12 else int(np.sum(qty_series == 0))
        
        # Тренд
        trend = 0
        if len(qty_series) >= 4:
            try:
                x = np.arange(len(qty_series))
                coeffs = np.polyfit(x, qty_series, 1)
                trend = float(coeffs[0])
            except:
                trend = 0
        
        return pd.Series({
            'ma_3': float(ma_3), 
            'ma_6': float(ma_6), 
            'consecutive_zeros': int(consecutive_zeros),
            'zero_weeks_12': int(zero_weeks_12), 
            'trend': float(trend)
        })
    
    # Применяем функцию и получаем DataFrame с Art в индексе
    features = weekly.groupby('Art').apply(compute_features, include_groups=False).reset_index()
    
    # Расчет доли магазинов без продаж
    total_stores = df['Magazin'].nunique()
    
    if total_stores == 0:
        st.error("❌ Не найдено магазинов в данных")
        st.stop()
    
    # Магазины с продажами для каждого артикула
    stores_with_sales = df[df['Qty'] > 0].groupby('Art')['Magazin'].nunique().reset_index()
    stores_with_sales.columns = ['Art', 'stores_with_sales']
    stores_with_sales['no_store_ratio'] = 1 - (stores_with_sales['stores_with_sales'] / total_stores)
    
    features = features.merge(stores_with_sales[['Art', 'no_store_ratio']], on='Art', how='left')
    features['no_store_ratio'] = features['no_store_ratio'].fillna(1.0)
    
    return features

def create_ml_model(features, abc_analysis):
    # Создание меток для обучения (ИСПРАВЛЕННАЯ ЛОГИКА)
    def create_labels(row):
        score = 0
        
        # Категория C - агрессивные критерии
        if row['abc_category'] == 'C':
            if row['consecutive_zeros'] >= zero_weeks_threshold: 
                score += 3
            elif row['zero_weeks_12'] >= zero_weeks_threshold // 2: 
                score += 2
            
            if row['no_store_ratio'] > max_store_ratio: 
                score += 2
            
            if row['total_qty'] < min_total_sales: 
                score += 2
            
            if row['trend'] < -0.1: 
                score += 1
        
        # Категория B - умеренные критерии (ИСПРАВЛЕНО)
        elif row['abc_category'] == 'B':
            if row['consecutive_zeros'] >= zero_weeks_threshold * 2:  # 24 недели
                score += 3
            elif row['consecutive_zeros'] >= zero_weeks_threshold:  # 12 недель
                score += 2
            
            if row['no_store_ratio'] > max_store_ratio:  # 85%
                score += 2
            
            if row['total_qty'] < min_total_sales * 2:  # 10 единиц
                score += 1
            
            if row['trend'] < -0.1:
                score += 1
        
        # Категория A - только критичные случаи
        elif row['abc_category'] == 'A':
            if row['consecutive_zeros'] >= zero_weeks_threshold * 3:  # 36 недель
                score += 2
            if row['no_store_ratio'] > 0.95:  # 95%
                score += 1
        
        # Критичные случаи для ЛЮБОЙ категории
        if row['consecutive_zeros'] >= zero_weeks_threshold * 2 and row['no_store_ratio'] > max_store_ratio:
            score += 2  # Усиление для комбинации факторов
        
        return 1 if score >= 4 else 0
    
    # Объединение данных
    final_features = features.merge(
        abc_analysis[['Art', 'total_qty', 'abc_category', 'last_sale']], 
        on='Art', 
        how='left'
    )
    final_features['label'] = final_features.apply(create_labels, axis=1)
    
    # Обучение модели
    feature_cols = ['ma_3', 'ma_6', 'consecutive_zeros', 'zero_weeks_12', 'trend', 'no_store_ratio', 'total_qty']
    X = final_features[feature_cols].fillna(0)
    y = final_features['label']
    
    st.write(f"**Распределение:** Снять: {y.sum()}, Оставить: {len(y) - y.sum()}")
    
    # Проверка возможности обучения
    if len(y.unique()) > 1 and y.sum() >= 2:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                stratify=y, 
                random_state=42, 
                test_size=0.3
            )
            
            clf = RandomForestClassifier(
                n_estimators=30, 
                random_state=42, 
                class_weight='balanced' if use_balanced_model else None,
                max_depth=8, 
                min_samples_split=5, 
                n_jobs=1
            )
            
            clf.fit(X_train, y_train)
            final_features['prob_dying'] = clf.predict_proba(X)[:, 1] * 100
            test_score = clf.score(X_test, y_test)
            
        except Exception as e:
            st.warning(f"⚠️ Ошибка ML: {e}. Используем простую логику.")
            final_features['prob_dying'] = final_features['label'].astype(float) * 100
            test_score = 0.0
    else:
        st.warning("⚠️ Недостаточно данных для ML. Используем простую логику.")
        final_features['prob_dying'] = final_features['label'].astype(float) * 100
        test_score = 0.0
    
    return final_features, test_score

def create_prophet_forecasts(df, abc_analysis):
    if not PROPHET_AVAILABLE:
        return pd.DataFrame()
    
    try:
        with st.spinner("📈 Прогнозы Prophet..."):
            top_arts = abc_analysis.nlargest(TOP_N, 'total_qty')['Art']
            forecasts = []
            
            for art in top_arts:
                try:
                    sales = df[df['Art'] == art].groupby('Data')['Qty'].sum().reset_index()
                    if len(sales) < 8: 
                        continue
                    
                    sales.columns = ['ds', 'y']
                    
                    model = Prophet(
                        daily_seasonality=False, 
                        weekly_seasonality=False, 
                        yearly_seasonality=False,
                        changepoint_prior_scale=0.05
                    )
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model.fit(sales)
                        future = model.make_future_dataframe(periods=30)
                        forecast = model.predict(future)
                    
                    median_30 = max(0, forecast.tail(30)['yhat'].median())
                    forecasts.append({'Art': art, 'forecast_30_median': float(median_30)})
                    
                except Exception as e:
                    continue
            
            return pd.DataFrame(forecasts)
            
    except Exception as e:
        st.warning(f"⚠️ Ошибка Prophet: {e}")
        return pd.DataFrame()

def get_recommendations(row):
    # Формирование причин
    reasons = []
    
    if row['abc_category'] == 'C': 
        reasons.append("Категория C")
    elif row['abc_category'] == 'B':
        reasons.append("Категория B")
    
    if row['consecutive_zeros'] >= zero_weeks_threshold * 2:
        reasons.append(f"Без продаж {int(row['consecutive_zeros'])} недель (критично!)")
    elif row['consecutive_zeros'] >= zero_weeks_threshold: 
        reasons.append(f"Без продаж {int(row['consecutive_zeros'])} недель")
    
    if row['zero_weeks_12'] >= zero_weeks_threshold // 2: 
        reasons.append(f"Из 12 недель {int(row['zero_weeks_12'])} без продаж")
    
    if row['no_store_ratio'] > max_store_ratio: 
        stores_with_sales_pct = (1 - row['no_store_ratio']) * 100
        reasons.append(f"Продажи в {stores_with_sales_pct:.0f}% магазинов")
    
    if row['total_qty'] < min_total_sales: 
        reasons.append(f"Малый объем ({row['total_qty']:.1f})")
    elif row['total_qty'] < min_total_sales * 2:
        reasons.append(f"Низкий объем ({row['total_qty']:.1f})")
    
    if row['trend'] < -0.1: 
        reasons.append("Негативный тренд")
    
    # Добавляем дату последней продажи
    if pd.notnull(row.get('last_sale')):
        last_sale_str = row['last_sale'].strftime('%Y-%m-%d')
        reasons.append(f"Последняя продажа: {last_sale_str}")
    
    reason = "; ".join(reasons) if reasons else "Стабильные продажи"
    
    # КРИТИЧНЫЕ СЛУЧАИ - переопределение независимо от ML
    # 1. Экстремально долгое отсутствие продаж
    if row['consecutive_zeros'] >= zero_weeks_threshold * 3:  # 36 недель
        return reason, "🚫 Снять"
    
    # 2. Категория C с превышением всех порогов
    if (row['abc_category'] == 'C' and 
        row['consecutive_zeros'] >= zero_weeks_threshold and 
        row['total_qty'] < min_total_sales and
        row['no_store_ratio'] > max_store_ratio):
        return reason, "🚫 Снять"
    
    # 3. Категория B с критическими показателями
    if (row['abc_category'] == 'B' and 
        row['consecutive_zeros'] >= zero_weeks_threshold * 2 and 
        row['no_store_ratio'] > max_store_ratio):
        return reason, "🚫 Снять"
    
    # 4. Долгое отсутствие + низкое распространение для B
    if (row['abc_category'] == 'B' and
        row['consecutive_zeros'] >= zero_weeks_threshold * 1.5 and
        row['no_store_ratio'] > 0.85 and
        row['total_qty'] < min_total_sales * 2):
        return reason, "⚠️ Наблюдать"
    
    # Стандартная логика на основе ML
    prob_threshold_pct = final_threshold * 100
    
    if row['prob_dying'] > prob_threshold_pct:
        return reason, "🚫 Снять"
    elif row['prob_dying'] > prob_threshold_pct * 0.7:
        return reason, "⚠️ Наблюдать"
    
    # Дополнительные проверки для "Наблюдать"
    if (row['consecutive_zeros'] >= zero_weeks_threshold and 
        row['no_store_ratio'] > 0.75):
        return reason, "⚠️ Наблюдать"
    
    return reason, "✅ Оставить"

# Выполнение анализа
df, weekly, all_arts, unique_weeks = process_data(df)
abc_analysis = calculate_abc_xyz_analysis(df)
features = calculate_features(weekly, df)
final_features, test_score = create_ml_model(features, abc_analysis)
forecast_df = create_prophet_forecasts(df, abc_analysis)

# Финальная таблица
final = final_features.merge(abc_analysis[['Art', 'xyz_category', 'last_sale']], on='Art', how='left')
if not forecast_df.empty:
    final = final.merge(forecast_df, on='Art', how='left')
final = final.merge(df[['Art', 'Name']].drop_duplicates(), on='Art', how='left')

# Получение рекомендаций
recommendations = final.apply(get_recommendations, axis=1)
final['Причина'] = [rec[0] for rec in recommendations]
final['Рекомендация'] = [rec[1] for rec in recommendations]

# === РЕЗУЛЬТАТЫ ===
st.header("📊 Результаты анализа")

total_products = len(final)
candidates_remove = len(final[final['Рекомендация'] == "🚫 Снять"])
candidates_watch = len(final[final['Рекомендация'] == "⚠️ Наблюдать"])

col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("Всего товаров", total_products)
with col2: st.metric("К снятию", candidates_remove, f"{candidates_remove/total_products*100:.1f}%")
with col3: st.metric("Наблюдать", candidates_watch, f"{candidates_watch/total_products*100:.1f}%")
with col4: st.metric("Точность модели", f"{test_score:.2f}" if test_score > 0 else "N/A")

# ABC/XYZ распределение
st.subheader("📈 ABC/XYZ анализ")
abc_dist = final['abc_category'].value_counts()
xyz_dist = final['xyz_category'].value_counts()

col1, col2 = st.columns(2)
with col1:
    st.write("**ABC категории:**")
    st.write(f"A: {abc_dist.get('A', 0)}, B: {abc_dist.get('B', 0)}, C: {abc_dist.get('C', 0)}")
with col2:
    st.write("**XYZ категории:**")
    st.write(f"X: {xyz_dist.get('X', 0)}, Y: {xyz_dist.get('Y', 0)}, Z: {xyz_dist.get('Z', 0)}")

# === ФИЛЬТРЫ И ТАБЛИЦА ===
st.subheader("🔍 Фильтры")
col1, col2, col3 = st.columns(3)

with col1:
    filter_recommendation = st.selectbox("Рекомендация:", ["Все", "🚫 Снять", "⚠️ Наблюдать", "✅ Оставить"])
    filter_abc = st.selectbox("ABC:", ["Все", "A", "B", "C"])
with col2:
    min_prob = st.slider("Мин. вероятность (%)", 0, 100, 0)
    filter_xyz = st.selectbox("XYZ:", ["Все", "X", "Y", "Z"])
with col3:
    min_zero_weeks = st.slider("Мин. недель без продаж", 0, 20, 0)
    search_art = st.text_input("Поиск артикула/названия")

# Применение фильтров
filtered_df = final.copy()
if filter_recommendation != "Все":
    filtered_df = filtered_df[filtered_df['Рекомендация'] == filter_recommendation]
if filter_abc != "Все":
    filtered_df = filtered_df[filtered_df['abc_category'] == filter_abc]
if filter_xyz != "Все":
    filtered_df = filtered_df[filtered_df['xyz_category'] == filter_xyz]

filtered_df = filtered_df[
    (filtered_df['prob_dying'] >= min_prob) &
    (filtered_df['consecutive_zeros'] >= min_zero_weeks)
]

if search_art:
    mask = (filtered_df['Art'].astype(str).str.contains(search_art, case=False, na=False) |
            filtered_df['Name'].astype(str).str.contains(search_art, case=False, na=False))
    filtered_df = filtered_df[mask]

# Таблица результатов
st.subheader(f"📋 Результаты ({len(filtered_df)} товаров)")

display_columns = ['Art', 'Name', 'abc_category', 'xyz_category', 'total_qty', 'consecutive_zeros', 'no_store_ratio', 'prob_dying', 'Причина', 'Рекомендация']
if 'forecast_30_median' in filtered_df.columns:
    display_columns.insert(-2, 'forecast_30_median')

display_df = filtered_df[display_columns].copy()
display_df['no_store_ratio'] = (display_df['no_store_ratio'] * 100).round(1)
display_df['prob_dying'] = display_df['prob_dying'].round(1)

column_names = ['Артикул', 'Название', 'ABC', 'XYZ', 'Объем', 'Недель_без_продаж', 'Магазины_без_продаж_%', 'Вероятность_снятия_%']
if 'forecast_30_median' in display_df.columns:
    column_names.append('Прогноз_30дн')
column_names.extend(['Причина', 'Рекомендация'])

display_df.columns = column_names
st.dataframe(display_df, use_container_width=True)

# === ЭКСПОРТ ===
st.subheader("💾 Экспорт")
if st.button("📥 Подготовить Excel"):
    try:
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            output_cols = ['Art', 'Name', 'abc_category', 'xyz_category', 'total_qty', 'consecutive_zeros', 'no_store_ratio', 'prob_dying', 'Причина', 'Рекомендация']
            if 'forecast_30_median' in final.columns:
                output_cols.insert(-2, 'forecast_30_median')
            
            final[output_cols].to_excel(writer, sheet_name='Результаты', index=False)
            
            stats = pd.DataFrame({
                'Метрика': ['Всего', 'Снять', 'Наблюдать', 'Оставить', 'Порог_ML_%'],
                'Значение': [total_products, candidates_remove, candidates_watch, 
                           total_products - candidates_remove - candidates_watch, final_threshold*100]
            })
            stats.to_excel(writer, sheet_name='Статистика', index=False)
        
        st.download_button("📥 Скачать Excel", buffer.getvalue(), "analysis_results.xlsx", 
                          "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.success("✅ Готово!")
    except Exception as e:
        st.error(f"❌ Ошибка: {str(e)}")

with st.expander("ℹ️ Информация"):
    st.write(f"**Статус:** Prophet {'✅' if PROPHET_AVAILABLE else '❌'}, Обработано: {len(final)}")
    if not PROPHET_AVAILABLE:
        st.warning("⚠️ Установите Prophet: pip install prophet")
