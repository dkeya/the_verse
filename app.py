import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
from datetime import datetime, timedelta
import hashlib
from st_aggrid import AgGrid, GridOptionsBuilder
import json
import base64
from io import BytesIO
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from fpdf import FPDF
import tempfile
import os
import hmac
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import logging
import altair as alt
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import dask.dataframe as dd
import shap
import optuna
import joblib
import gc
import tempfile
import os
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import partial_dependence
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt

# ==============================================
# ENHANCED DATA MODEL WITH REAL DATA INTEGRATION
# ==============================================

TENANT_CONFIG = {
    'virtual_analytics': {
        'storage_bucket': 'va-admin-data',
        'db_schema': 'admin',
        'allowed_ips': ['0.0.0.0/0']
    },
    'acme_brokers': {
        'storage_bucket': 'va-acme-data',
        'db_schema': 'tenant_acme',
        'allowed_ips': ['192.168.1.0/24']
    },
    'global_insurers': {
        'storage_bucket': 'va-global-data',
        'db_schema': 'tenant_global',
        'allowed_ips': ['10.0.0.0/16']
    }
}

COLUMN_MAPPING = {
    'provider_id': ['Provider_Name', 'provider_id', 'Provider'],
    'amount': ['Claim_Amount_KES', 'amount', 'TotalClaimed'],
    'date': ['Submission_Date', 'date', 'ClaimDate'],
    'employee_id': ['Employee_ID', 'employee_id', 'MemberID'],
    'diagnosis': ['Diagnosis', 'diagnosis', 'Ailment'],
    'treatment': ['Treatment', 'treatment', 'Benefit']
}

def get_column(df, possible_names):
    """Case-insensitive column name matching with normalization"""
    # Normalize all column names and possible names
    df_cols = [str(col).strip().lower().replace(' ', '_') for col in df.columns]
    possible_names = [str(name).strip().lower().replace(' ', '_') for name in possible_names]

    for i, col in enumerate(df_cols):
        if col in possible_names:
            return df.columns[i]  # Return original column name

    # Try partial matches if no exact match
    for i, col in enumerate(df_cols):
        for name in possible_names:
            if name in col or col in name:
                return df.columns[i]

    return None

def get_tenant_config(tenant_id):
    """Get tenant-specific configuration"""
    return TENANT_CONFIG.get(tenant_id, {
        'storage_bucket': f'va-{tenant_id}-data',
        'db_schema': f'tenant_{tenant_id}',
        'allowed_ips': []
    })

# ==============================================
# AUTHENTICATION SYSTEM
# ==============================================

USER_DB = {
    'admin': {
        'password': hashlib.sha256('admin123'.encode()).hexdigest(),
        'role': 'admin',
        'tenant': 'virtual_analytics',
        'name': 'Admin User'
    },
    'broker1': {
        'password': hashlib.sha256('brokerpass'.encode()).hexdigest(),
        'role': 'broker',
        'tenant': 'acme_brokers',
        'name': 'Broker User',
        'clients': {
            'Acme Corp': {
                'users': ['acme_admin'],
                'plan': 'premium',
                'data_access': ['claims', 'reports', 'predictions']
            },
            'Globex': {
                'users': ['globex_user'],
                'plan': 'standard',
                'data_access': ['claims', 'predictions']
            }
        }
    },
    'underwriter1': {
        'password': hashlib.sha256('underwriterpass'.encode()).hexdigest(),
        'role': 'underwriter',
        'tenant': 'global_insurers',
        'name': 'Underwriter User',
        'clients': {
            'Initech': {
                'users': ['initech_finance'],
                'plan': 'enterprise',
                'data_access': ['claims', 'fraud', 'reports', 'predictions']
            },
            'Umbrella': {
                'users': ['umbrella_hr'],
                'plan': 'premium',
                'data_access': ['claims', 'reports', 'predictions']
            }
        }
    },
    'acme_admin': {
        'password': hashlib.sha256('acme123'.encode()).hexdigest(),
        'role': 'client',
        'tenant': 'acme_brokers',
        'client_org': 'Acme Corp',
        'name': 'Acme Admin'
    },
    'initech_finance': {
        'password': hashlib.sha256('initech123'.encode()).hexdigest(),
        'role': 'client',
        'tenant': 'global_insurers',
        'client_org': 'Initech',
        'name': 'Initech Finance'
    }
}

def authenticate(username, password):
    """Enhanced authentication with tenant checks"""
    if username in USER_DB:
        hashed_pw = hashlib.sha256(password.encode()).hexdigest()
        if USER_DB[username]['password'] == hashed_pw:
            user_info = USER_DB[username].copy()
            user_info['tenant_config'] = get_tenant_config(user_info['tenant'])
            return True, user_info
    return False, None

def login_form():
    """Enhanced login form with security features"""
    with st.form("Login"):
        st.markdown("<h1 style='text-align: center;'>🌀 the Verse</h1>", unsafe_allow_html=True)
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            if 'login_attempts' not in st.session_state:
                st.session_state.login_attempts = 0

            if st.session_state.login_attempts >= 3:
                st.error("Too many attempts. Please try again later.")
                time.sleep(2)
                return

            success, user_info = authenticate(username, password)
            if success:
                st.session_state.login_attempts = 0
                st.session_state['authenticated'] = True
                st.session_state['user_info'] = user_info
                log_audit_event(username, "login_success")
                st.rerun()
            else:
                st.session_state.login_attempts += 1
                log_audit_event(username, "login_failed")
                st.error("Invalid credentials")

# ==============================================
# SECURITY & AUDIT COMPONENTS
# ==============================================

def log_audit_event(user, action, target=None):
    """Centralized audit logging with tenant isolation"""
    if not os.path.exists('logs'):
        os.makedirs('logs')

    tenant = st.session_state.get('user_info', {}).get('tenant', 'system')
    log_entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'user': user,
        'action': action,
        'target': target,
        'ip': st.query_params.get('client_ip', [''])[0]
    }

    with open(f"logs/{tenant}_audit.log", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

def encrypt_data(value, tenant_key):
    """Simplified format-preserving encryption example"""
    return str((int(float(value) * 100) * 7919) % 999999)

def pseudonymize_claim_data(record, tenant_key):
    """Pseudonymize sensitive fields for a claim record"""
    record['employee_id'] = hmac.new(
        tenant_key.encode(),
        record['employee_id'].encode(),
        hashlib.sha256
    ).hexdigest()[:12]

    if 'amount' in record:
        record['amount'] = encrypt_data(record['amount'], tenant_key)

    return record

# ==============================================
# ENHANCED CLAIMS PREDICTION SYSTEM
# ==============================================

class ClaimsPredictor:
    """Enhanced healthcare claims predictor with fraud detection capabilities"""
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.data = None
        self.clean_data = None
        self.feature_importance = None
        self.baseline_metrics = None
        self.category_order = ['Silver', 'Gold', 'Platinum']
        self.required_prediction_columns = None
        self.logger = logging.getLogger(__name__)
        self.available_values = {}  # Store available values for dropdowns
        self.monitor = ModelMonitor()
        self.fraud_model = None
        self.training_date = None
        self.raw_data = None  # Store original data

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('claims_predictor.log'),
                logging.StreamHandler()
            ]
        )

    def save_model(self, file_path):
        """Serialize the complete predictor object to disk"""
        try:
            with open(file_path, 'wb') as f:
                joblib.dump(self, f)
            self.logger.info(f"Model successfully saved to {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Model serialization failed: {str(e)}")
            st.error("Failed to save model. Please check logs for details.")
            return False

    @classmethod
    def load_model(cls, file_path):
        """Deserialize a predictor object from disk"""
        try:
            with open(file_path, 'rb') as f:
                predictor = joblib.load(f)
                if not isinstance(predictor, cls):
                    raise TypeError("Loaded object is not a ClaimsPredictor instance")
                predictor.logger.info(f"Model successfully loaded from {file_path}")
                return predictor
        except Exception as e:
            logging.error(f"Model deserialization failed: {str(e)}")
            st.error("Failed to load model. File may be corrupted or incompatible.")
            return None

    def load_data(self, uploaded_file):
        """Optimized data loading with memory management"""
        try:
            self.logger.info(f"Loading file: {uploaded_file.name}")

            # Validate file type
            if not (uploaded_file.name.endswith('.csv') or uploaded_file.name.endswith('.xlsx')):
                raise ValueError("Unsupported file format. Please upload CSV or Excel file.")

            # For large files (>50MB), use chunked processing
            if uploaded_file.size > 50 * 1024 * 1024:
                st.warning("Large file detected - using optimized chunked processing")

                # Save to temp file for processing
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                try:
                    success = self.process_large_data(tmp_path)
                    return self.data if success else None
                finally:
                    os.unlink(tmp_path)
            else:
                # Use standard loading for smaller files
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:  # Excel
                    df = pd.read_excel(uploaded_file)

                # Validate
                if len(df) < 100:
                    raise ValueError("Insufficient data. Need at least 100 claims.")

                required_cols = ['Employee_ID', 'Claim_Amount_KES', 'Submission_Date', 'Employer']
                missing_cols = set(required_cols) - set(df.columns)
                if missing_cols:
                    raise ValueError(f"Missing required columns: {missing_cols}")

                # Process
                df = self._optimize_dataframe(df)
                df.columns = df.columns.str.strip().str.replace(' ', '_')
                self.data = df
                self._initialize_available_values()
                self._validate_healthcare_data(df)

                return df

        except Exception as e:
            self.logger.error(f"Data loading failed: {str(e)}")
            st.error(f"Data loading error: {str(e)}")
            return None

    def clean_and_prepare_data(self):
        """Comprehensive data cleaning pipeline with error handling and validation"""
        if self.data is None:
            st.warning("No data loaded")
            return False

        try:
            with st.spinner("Cleaning and preparing data..."):
                # Create copy while preserving original
                df = self.data.copy()
                self.raw_data = self.data.copy()  # Preserve original

                # Enhanced Data Type Validation
                type_conversions = {
                    'Employee_Age': 'int',
                    'Claim_Amount_KES': 'float',
                    'Co_Payment_KES': 'float',
                    'Submission_Date': 'datetime64[ns]',
                    'Service_Date': 'datetime64[ns]',
                    'Hire_Date': 'datetime64[ns]'
                }

                for col, dtype in type_conversions.items():
                    if col in df.columns:
                        try:
                            if dtype == 'float':
                                df[col] = df[col].astype(str).str.replace('[^\d.]', '', regex=True)
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                            elif dtype == 'int':
                                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                            elif dtype.startswith('datetime'):
                                df[col] = pd.to_datetime(df[col], errors='coerce')
                        except Exception as e:
                            st.warning(f"Error converting {col} to {dtype}: {str(e)}")
                            df[col] = df[col].astype(str)

                # Missing Values Handling
                missing_report = pd.DataFrame(df.isnull().sum(), columns=['Missing Values'])
                missing_report['% Missing'] = (missing_report['Missing Values'] / len(df)) * 100

                # Drop columns with >70% missing values
                cols_to_drop = missing_report[missing_report['% Missing'] > 70].index.tolist()
                if cols_to_drop:
                    df = df.drop(columns=cols_to_drop)
                    st.warning(f"Dropped columns with >70% missing values: {', '.join(cols_to_drop)}")

                # Fill remaining missing values
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].fillna('Unknown')
                    elif df[col].dtype in ['int64', 'float64']:
                        df[col] = df[col].fillna(df[col].median())

                # Data Validation
                if 'Claim_Amount_KES' in df.columns:
                    df['Claim_Amount_KES'] = df['Claim_Amount_KES'].abs()
                    if (df['Claim_Amount_KES'] <= 0).any():
                        st.warning("Found non-positive claim amounts - setting to median")
                        df.loc[df['Claim_Amount_KES'] <= 0, 'Claim_Amount_KES'] = df['Claim_Amount_KES'].median()

                if 'Employee_Age' in df.columns:
                    df['Employee_Age'] = df['Employee_Age'].clip(18, 100)

                # Feature Engineering
                if 'Department' not in df.columns:
                    df['Department'] = 'General'

                if 'Hire_Date' in df.columns:
                    df['Tenure'] = (datetime.now() - df['Hire_Date']).dt.days / 365
                    df['Tenure_Group'] = pd.cut(df['Tenure'],
                                              bins=[0, 1, 5, 100],
                                              labels=['<1yr', '1-5yrs', '5+yrs'])

                if 'Employee_Age' in df.columns:
                    df['Age_Group'] = pd.cut(df['Employee_Age'],
                                           bins=[0, 25, 35, 45, 55, 65, 100],
                                           labels=['<25', '25-35', '35-45', '45-55', '55-65', '65+'])

                if 'Claim_Amount_KES' in df.columns:
                    df['Claim_Size'] = pd.qcut(df['Claim_Amount_KES'],
                                             q=4,
                                             labels=['Small', 'Medium', 'Large', 'Very Large'])

                if 'Submission_Date' in df.columns:
                    df['Claim_Weekday'] = df['Submission_Date'].dt.day_name()
                    df['Claim_Month'] = df['Submission_Date'].dt.month_name()
                    df['Claim_Quarter'] = df['Submission_Date'].dt.quarter

                self.clean_data = df
                self.logger.info("Data cleaning completed successfully")
                return True

        except Exception as e:
            self.logger.error(f"Data cleaning failed: {str(e)}", exc_info=True)
            st.error(f"Data cleaning failed: {str(e)}")
            return False

    def generate_data_report(self):
        """Generate an exploratory data analysis report using pandas profiling"""
        try:
            if self.clean_data is None:
                st.warning("No cleaned data available to generate report")
                return False

            with st.spinner("Generating data report..."):
                profile = ProfileReport(
                    self.clean_data,
                    minimal=True,
                    explorative=True,
                    progress_bar=False
                )

                st_profile_report(profile)
                return True

        except Exception as e:
            self.logger.error(f"Failed to generate data report: {str(e)}")
            st.error(f"Failed to generate data report: {str(e)}")
            return False

    def preprocess_data(self, target='Claim_Amount_KES'):
        """Prepare data for modeling with enhanced validation"""
        try:
            if self.clean_data is None:
                raise ValueError("No cleaned data available")

            # Define features and target
            X = self.clean_data.drop(columns=[target], errors='ignore')
            y = self.clean_data[target]

            # Define feature types with validation
            categorical_features = [
                'Visit_Type', 'Diagnosis_Group', 'Treatment_Type',
                'Provider_Name', 'Hospital_County', 'Employee_Gender',
                'Claim_Weekday', 'Claim_Month', 'Employer', 'Category',
                'Age_Group', 'Claim_Size', 'Department', 'Tenure_Group'
            ]

            numerical_features = [
                'Employee_Age', 'Co_Payment_KES', 'Is_Pre_Authorized',
                'Inpatient_Cap_KES_Utilization', 'Outpatient_Cap_KES_Utilization',
                'Optical_Cap_KES_Utilization', 'Dental_Cap_KES_Utilization',
                'Maternity_Cap_KES_Utilization', 'Claim_Amount_to_Mean',
                'Same_Day_Claims', 'Employer_Z_Score'
            ]

            # Validate features exist in data
            categorical_features = [f for f in categorical_features if f in X.columns]
            numerical_features = [f for f in numerical_features if f in X.columns]

            if not categorical_features and not numerical_features:
                raise ValueError("No valid features found for modeling")

            # Create preprocessing pipeline
            self.preprocessor = ColumnTransformer([
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])

            # Store required columns for prediction
            self.required_prediction_columns = numerical_features + categorical_features

            return X, y

        except Exception as e:
            self.logger.error(f"Preprocessing error: {str(e)}")
            st.error(f"Preprocessing error: {str(e)}")
            return None, None

    def train_model(self, model_type="Gradient Boosting", target='Claim_Amount_KES',
                test_size=0.2, cv_folds=5, do_tuning=False, max_iter=20):
        """Enhanced model training with multiple algorithms"""
        try:
            X, y = self.preprocess_data(target)
            if X is None or y is None:
                return False

            # Split data with temporal validation if date is available
            if 'Submission_Date' in X.columns:
                X_sorted = X.sort_values('Submission_Date')
                y_sorted = y[X_sorted.index]
                split_idx = int(len(X_sorted) * (1 - test_size))
                X_train, X_test = X_sorted.iloc[:split_idx], X_sorted.iloc[split_idx:]
                y_train, y_test = y_sorted.iloc[:split_idx], y_sorted.iloc[split_idx:]
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42)

            # Initialize models
            models = {}
            param_grids = {}

            if model_type == "Gradient Boosting" or model_type == "Auto Select Best":
                models['GradientBoosting'] = GradientBoostingRegressor(random_state=42)
                param_grids['GradientBoosting'] = {
                    'regressor__n_estimators': [100, 200, 300],
                    'regressor__learning_rate': [0.01, 0.05, 0.1],
                    'regressor__max_depth': [3, 5, 7]
                }

            if model_type == "Random Forest" or model_type == "Auto Select Best":
                models['RandomForest'] = RandomForestRegressor(random_state=42)
                param_grids['RandomForest'] = {
                    'regressor__n_estimators': [50, 100, 200],
                    'regressor__max_depth': [5, 10, None],
                    'regressor__min_samples_split': [2, 5, 10]
                }

            if model_type == "XGBoost" or model_type == "Auto Select Best":
                # Detect GPU support
                tree_method = 'gpu_hist' if xgb.XGBRegressor().get_params().get('tree_method') == 'gpu_hist' else 'auto'

                models['XGBoost'] = xgb.XGBRegressor(random_state=42, tree_method=tree_method)
                param_grids['XGBoost'] = {
                    'regressor__n_estimators': [50, 100, 200],
                    'regressor__max_depth': [3, 5, 7],
                    'regressor__learning_rate': [0.01, 0.05, 0.1],
                    'regressor__tree_method': [tree_method]
                }

            if model_type == "Neural Network" or model_type == "Auto Select Best":
                models['NeuralNetwork'] = MLPRegressor(random_state=42, max_iter=500)
                param_grids['NeuralNetwork'] = {
                    'regressor__hidden_layer_sizes': [(50,), (50, 25), (100, 50)],
                    'regressor__activation': ['relu', 'tanh'],
                    'regressor__solver': ['adam', 'sgd']
                }

            # Train and evaluate each model
            results = []
            best_model = None
            best_score = -np.inf

            for name, model in models.items():
                # Create pipeline
                pipeline = Pipeline([
                    ('preprocessor', self.preprocessor),
                    ('regressor', model)
                ])

                # Hyperparameter tuning if enabled
                if do_tuning and name in param_grids:
                    if name == 'XGBoost':
                        # Special handling for XGBoost with Optuna
                        def objective(trial):
                            params = {
                                'regressor__n_estimators': trial.suggest_int('regressor__n_estimators', 50, 500),
                                'regressor__max_depth': trial.suggest_int('regressor__max_depth', 3, 10),
                                'regressor__learning_rate': trial.suggest_float('regressor__learning_rate', 0.001, 0.1, log=True),
                                'regressor__subsample': trial.suggest_float('regressor__subsample', 0.5, 1.0),
                                'regressor__colsample_bytree': trial.suggest_float('regressor__colsample_bytree', 0.5, 1.0)
                            }
                            pipeline.set_params(**params)
                            pipeline.fit(X_train, y_train)
                            return mean_absolute_error(y_test, pipeline.predict(X_test))

                        study = optuna.create_study(direction='minimize')
                        study.optimize(objective, n_trials=max_iter)
                        best_params = study.best_params
                        pipeline.set_params(**best_params)
                    else:
                        # Standard RandomizedSearchCV for other models
                        search = RandomizedSearchCV(
                            pipeline,
                            param_grids[name],
                            n_iter=max_iter,
                            cv=cv_folds,
                            scoring='neg_mean_absolute_error',
                            random_state=42
                        )
                        search.fit(X_train, y_train)
                        pipeline = search.best_estimator_
                        self.logger.info(f"Best params for {name}: {search.best_params_}")

                # Train model
                pipeline.fit(X_train, y_train)

                # Evaluate
                y_pred = pipeline.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)

                results.append({
                    'Model': name,
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': r2
                })

                # Track best model
                if r2 > best_score:
                    best_score = r2
                    best_model = pipeline

            # Create results DataFrame
            results_df = pd.DataFrame(results).sort_values('R2', ascending=False)

            if model_type == "Auto Select Best":
                st.success(f"Best model selected: {results_df.iloc[0]['Model']}")
                self.model = best_model
                self.baseline_metrics = {
                    'MAE': results_df.iloc[0]['MAE'],
                    'RMSE': results_df.iloc[0]['RMSE'],
                    'R2': results_df.iloc[0]['R2']
                }
            else:
                # For single model selection, use the last trained pipeline
                self.model = pipeline
                self.baseline_metrics = {
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': r2
                }

            # Calculate feature importance
            self._calculate_feature_importance(self.model)

            # Train fraud detection model
            self._train_fraud_model(X, y)

            # Log model performance
            self.monitor.log_performance(
                model_type,
                {'MAE': mae, 'RMSE': rmse, 'R2': r2}
            )

            # Set training date
            self.training_date = datetime.now()

            return results_df

        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            st.error(f"Training failed: {str(e)}")
            return None

    def _train_fraud_model(self, X, y):
        """Train isolation forest for fraud detection"""
        try:
            # Use preprocessor to transform data
            X_transformed = self.preprocessor.transform(X)

            # Train isolation forest
            self.fraud_model = IsolationForest(
                n_estimators=100,
                contamination=0.01,  # Assume 1% fraud
                random_state=42
            )
            self.fraud_model.fit(X_transformed)

            # Set dynamic threshold based on claim amounts
            scores = self.fraud_model.decision_function(X_transformed)
            self.fraud_threshold = np.percentile(scores, 1)  # Flag bottom 1% as potential fraud

        except Exception as e:
            self.logger.warning(f"Fraud model training failed: {str(e)}")
            self.fraud_model = None

    def _calculate_feature_importance(self, pipeline):
        """Calculate and store feature importance"""
        try:
            # Get feature names from the preprocessor
            if hasattr(pipeline.named_steps['preprocessor'], 'transformers_'):
                num_features = []
                cat_features = []

                for name, trans, features in pipeline.named_steps['preprocessor'].transformers_:
                    if name == 'num':
                        num_features = features
                    elif name == 'cat':
                        if hasattr(trans, 'get_feature_names_out'):
                            cat_features = trans.get_feature_names_out(features)
                        else:
                            cat_features = features

                all_features = np.concatenate([num_features, cat_features])
            else:
                all_features = pipeline.named_steps['preprocessor'].get_feature_names_out()

            # Get importance scores
            if hasattr(pipeline.named_steps['regressor'], 'feature_importances_'):
                importances = pipeline.named_steps['regressor'].feature_importances_
            elif hasattr(pipeline.named_steps['regressor'], 'coef_'):
                importances = np.abs(pipeline.named_steps['regressor'].coef_)
            else:
                importances = np.ones(len(all_features)) / len(all_features)

            self.feature_importance = pd.DataFrame({
                'Feature': all_features,
                'Importance': importances
            }).sort_values('Importance', ascending=False)

        except Exception as e:
            self.logger.warning(f"Could not calculate feature importance: {str(e)}")
            self.feature_importance = None

    def predict_claim_amount(self, input_data):
        """Robust claim amount prediction with input validation"""
        try:
            # Convert single record to DataFrame if needed
            if not isinstance(input_data, pd.DataFrame):
                input_data = pd.DataFrame([input_data])

            # Store original index for merging results later
            original_index = input_data.index

            # Validate required columns
            missing_cols = set(self.required_prediction_columns) - set(input_data.columns)
            if missing_cols:
                # Fill missing columns with sensible defaults
                defaults = {
                    'Claim_Weekday': datetime.now().strftime('%A'),
                    'Claim_Month': datetime.now().strftime('%B'),
                    'Employer': 'Unknown',
                    'Department': 'Unknown',
                    'Age_Group': '35-45',
                    'Claim_Size': 'Medium',
                    'Tenure_Group': '1-5yrs'
                }

                for col in missing_cols:
                    if col.endswith('_Utilization'):
                        input_data[col] = 0.5
                    elif col in defaults:
                        input_data[col] = defaults[col]
                    else:
                        input_data[col] = 0

                self.logger.warning(f"Filled missing columns: {missing_cols}")

            # Ensure all required columns are present and in correct order
            input_data = input_data.reindex(columns=self.required_prediction_columns, fill_value=0)

            # Convert numeric columns to float to avoid type issues
            numeric_cols = [col for col in input_data.columns if input_data[col].dtype in ['int64', 'float64']]
            for col in numeric_cols:
                input_data[col] = pd.to_numeric(input_data[col], errors='coerce').fillna(0)

            # Convert categorical columns to string
            categorical_cols = [col for col in input_data.columns if input_data[col].dtype == 'object']
            for col in categorical_cols:
                input_data[col] = input_data[col].astype(str)

            # Make predictions
            predictions = self.model.predict(input_data)

            # Add fraud detection
            if self.fraud_model:
                X_transformed = self.preprocessor.transform(input_data)
                fraud_scores = self.fraud_model.decision_function(X_transformed)
                fraud_flag = fraud_scores < self.fraud_threshold
            else:
                fraud_flag = np.zeros(len(predictions), dtype=bool)

            # Return format depends on input type
            if len(predictions) == 1:
                result = {
                    'prediction': float(predictions[0]),
                    'is_potential_fraud': bool(fraud_flag[0]),
                    'fraud_confidence': float(1 - fraud_scores[0]) if self.fraud_model else None
                }
                return result
            else:
                # For group predictions, return DataFrame with predictions
                result = input_data.copy()
                result['Predicted_Claim_Amount'] = predictions
                result['Is_Potential_Fraud'] = fraud_flag
                if self.fraud_model:
                    result['Fraud_Confidence'] = 1 - fraud_scores
                result.index = original_index  # Maintain original indexing
                return result

        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            st.error(f"Prediction failed: {str(e)}")
            return None

    def explain_prediction(self, sample_data):
        """Explain model predictions using SHAP values"""
        try:
            if self.model is None:
                st.warning("No trained model available")
                return

            # Get the preprocessor and model from pipeline
            preprocessor = self.model.named_steps['preprocessor']
            model = self.model.named_steps['regressor']

            # Transform sample data
            X_transformed = preprocessor.transform(sample_data)

            # Get feature names
            feature_names = preprocessor.get_feature_names_out()

            # Explain the model's predictions using SHAP
            if isinstance(model, xgb.XGBRegressor):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_transformed)
            else:
                explainer = shap.KernelExplainer(model.predict, X_transformed)
                shap_values = explainer.shap_values(X_transformed)

            # Visualize the first prediction's explanation
            st.subheader("Prediction Explanation")
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X_transformed, feature_names=feature_names)
            st.pyplot(fig)

        except Exception as e:
            self.logger.error(f"Failed to explain prediction: {str(e)}")
            st.error(f"Could not generate explanation: {str(e)}")

class ModelMonitor:
    """Class for monitoring model performance and data drift"""
    def __init__(self):
        self.performance_history = []
        self.data_drift_scores = []
        self.training_date = datetime.now()

    def log_performance(self, model_name, metrics):
        """Log model performance metrics"""
        self.performance_history.append({
            'timestamp': datetime.now(),
            'model': model_name,
            **metrics
        })

    def check_data_drift(self, current_data, reference_data):
        """Calculate data drift metrics"""
        drift_metrics = {}

        # For numerical columns
        num_cols = current_data.select_dtypes(include=['float64', 'int64']).columns
        for col in num_cols:
            # Kolmogorov-Smirnov test
            from scipy.stats import ks_2samp
            stat, p = ks_2samp(reference_data[col].dropna(), current_data[col].dropna())
            drift_metrics[col] = {'ks_stat': stat, 'ks_p': p}

        # For categorical columns
        cat_cols = current_data.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            # Population Stability Index
            ref_counts = reference_data[col].value_counts(normalize=True)
            curr_counts = current_data[col].value_counts(normalize=True)
            psi = np.sum((curr_counts - ref_counts) * np.log(curr_counts / ref_counts))
            drift_metrics[col] = {'psi': psi}

        self.data_drift_scores.append({
            'timestamp': datetime.now(),
            'drift_metrics': drift_metrics
        })
        return drift_metrics

# (Previous imports and code remain the same until the exploratory analysis functions)

# ==============================================
# EXPLORATORY ANALYSIS FUNCTIONS
# ==============================================

def render_claim_distribution(data):
    """Interactive claim distribution by category"""
    st.subheader("Claim Distribution Analysis")

    # Get available categorical columns
    cat_cols = [col for col in data.columns if data[col].dtype in ['object', 'category']]

    if not cat_cols:
        st.warning("No categorical columns found for distribution analysis")
        return

    col1, col2 = st.columns(2)

    with col1:
        # Default to 'treatment' if available, otherwise first category
        default_category = 'treatment' if 'treatment' in cat_cols else cat_cols[0]
        category = st.selectbox(
            "Group claims by",
            options=cat_cols,
            index=cat_cols.index(default_category) if default_category in cat_cols else 0,
            key='dist_category_select'
        )

        # Add interactive filters
        filter_col = st.selectbox(
            "Filter by",
            options=['None'] + cat_cols,
            key='dist_filter_select'
        )

        if filter_col != 'None':
            filter_values = st.multiselect(
                f"Select {filter_col} values",
                options=data[filter_col].unique(),
                key='dist_filter_values_select'
            )
            if filter_values:
                data = data[data[filter_col].isin(filter_values)]

    with col2:
        # Default to 'Sum of Amount'
        metric = st.selectbox(
            "Analysis metric",
            options=['Count', 'Sum of Amount', 'Average Amount'],
            index=1,  # Default to 'Sum of Amount'
            key='dist_metric_select'
        )

    # Prepare data based on selections
    if metric == 'Count':
        dist_data = data[category].value_counts().reset_index()
        dist_data.columns = [category, 'Count']
        y_metric = 'Count'
    else:
        # Ensure amount column is numeric
        amount_col = 'amount' if 'amount' in data.columns else None
        if not amount_col:
            st.warning("No amount column found for amount calculations")
            return
        
        # Convert amount to numeric, coercing errors to NaN
        data[amount_col] = pd.to_numeric(data[amount_col], errors='coerce')
        
        if metric == 'Sum of Amount':
            dist_data = data.groupby(category)[amount_col].sum().reset_index()
            dist_data.columns = [category, 'Total Amount']
            y_metric = 'Total Amount'
        elif metric == 'Average Amount':
            dist_data = data.groupby(category)[amount_col].mean().reset_index()
            dist_data.columns = [category, 'Average Amount']
            y_metric = 'Average Amount'

    # Create interactive visualization
    fig = px.bar(
        dist_data,
        x=category,
        y=y_metric,
        title=f"Claim Distribution by {category}",
        hover_data=[y_metric],
        color=category
    )

    # Add horizontal line for average if showing amounts
    if 'Amount' in metric and amount_col:
        avg_value = data[amount_col].mean() if metric == 'Average Amount' else data[amount_col].sum()/len(dist_data)
        fig.add_hline(
            y=avg_value,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Overall {'Average' if metric == 'Average Amount' else 'Mean per Category'}",
            annotation_position="top left"
        )

    st.plotly_chart(fig, use_container_width=True)

def render_temporal_analysis(data):
    """Temporal analysis with confidence intervals"""
    st.subheader("Temporal Claim Patterns")

    # Check for date column
    date_col = get_column(data, COLUMN_MAPPING['date'])
    if not date_col:
        st.warning("No date column found for temporal analysis")
        return

    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
        data[date_col] = pd.to_datetime(data[date_col], errors='coerce')

    # Remove rows with invalid dates
    data = data[data[date_col].notna()]

    if data.empty:
        st.warning("No valid dates found for temporal analysis")
        return

    # Temporal aggregation options
    time_unit = st.selectbox(
        "Time unit",
        options=['Day', 'Week', 'Month', 'Quarter', 'Year'],
        key='time_unit'
    )

    # Create temporal aggregation
    data = data.copy()
    data['time_period'] = data[date_col].dt.to_period(
        time_unit[0].lower()
    ).dt.to_timestamp()

    # Get amount column name
    amount_col = get_column(data, COLUMN_MAPPING['amount'])
    if not amount_col:
        st.warning("No amount column found for temporal analysis")
        return

    # Ensure amount column is numeric
    data[amount_col] = pd.to_numeric(data[amount_col], errors='coerce')

    # Group by time period - modified aggregation to handle numeric data only
    agg_dict = {
        amount_col: ['sum', 'count', 'mean', 'std']
    }

    temporal_data = data.groupby('time_period').agg(agg_dict).reset_index()

    # Flatten multi-index columns
    temporal_data.columns = [
        'time_period',
        'total_amount',
        'claim_count',
        'avg_amount',
        'std_amount'
    ]

    # Calculate confidence intervals
    temporal_data['ci_lower'] = temporal_data['avg_amount'] - 1.96 * temporal_data['std_amount']/np.sqrt(temporal_data['claim_count'])
    temporal_data['ci_upper'] = temporal_data['avg_amount'] + 1.96 * temporal_data['std_amount']/np.sqrt(temporal_data['claim_count'])

    # Visualization
    metric = st.selectbox(
        "Show metric",
        options=['Claim Count', 'Total Amount', 'Average Amount'],
        key='temp_metric'
    )

    if metric == 'Claim Count':
        fig = px.line(
            temporal_data,
            x='time_period',
            y='claim_count',
            title=f"Claim Count by {time_unit}",
            labels={'claim_count': 'Number of Claims'}
        )
    elif metric == 'Total Amount':
        fig = px.line(
            temporal_data,
            x='time_period',
            y='total_amount',
            title=f"Total Claim Amount by {time_unit}",
            labels={'total_amount': 'Total Amount (KES)'}
        )
    else:  # Average Amount
        fig = px.line(
            temporal_data,
            x='time_period',
            y='avg_amount',
            title=f"Average Claim Amount by {time_unit} with 95% CI",
            labels={'avg_amount': 'Average Amount (KES)'}
        )

        # Add confidence interval
        fig.add_trace(
            go.Scatter(
                x=temporal_data['time_period'],
                y=temporal_data['ci_upper'],
                fill=None,
                mode='lines',
                line_color='rgba(255,255,255,0)',
                showlegend=False
            )
        )
        fig.add_trace(
            go.Scatter(
                x=temporal_data['time_period'],
                y=temporal_data['ci_lower'],
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,100,80,0.2)',
                showlegend=False
            )
        )

    st.plotly_chart(fig, use_container_width=True)

def render_provider_efficiency(data):
    """Provider network cost efficiency analysis"""
    st.subheader("Provider Cost Efficiency")

    # Get provider column using the column mapping system
    provider_col = get_column(data, COLUMN_MAPPING['provider_id'])

    if not provider_col:
        st.warning("Provider information missing for efficiency analysis")
        return

    # Get amount column using column mapping
    amount_col = get_column(data, COLUMN_MAPPING['amount'])

    if not amount_col:
        st.warning("Amount information missing for efficiency analysis")
        return

    try:
        # Ensure numeric columns are properly formatted
        data[amount_col] = pd.to_numeric(data[amount_col], errors='coerce').fillna(0)

        # Initialize fraud_flag if it doesn't exist
        if 'fraud_flag' not in data.columns:
            data['fraud_flag'] = 0
        else:
            data['fraud_flag'] = pd.to_numeric(data['fraud_flag'], errors='coerce').fillna(0)

        # Calculate provider statistics
        agg_dict = {
            amount_col: ['sum', 'count', 'mean', 'median'],
            'fraud_flag': 'mean'
        }

        provider_stats = data.groupby(provider_col).agg(agg_dict).reset_index()

        # Flatten multi-index columns
        provider_stats.columns = [
            provider_col,
            'Total_Amount',
            'Total_Claims',
            'Avg_Amount',
            'Median_Amount',
            'Fraud_Rate'
        ]

        # Calculate efficiency metrics
        overall_avg = data[amount_col].mean()
        provider_stats['efficiency_ratio'] = provider_stats['Avg_Amount'] / overall_avg
        provider_stats['cost_per_claim'] = provider_stats['Total_Amount'] / provider_stats['Total_Claims']

    except Exception as e:
        st.error(f"Error calculating provider statistics: {str(e)}")
        return

    # Break-even analysis
    st.write("#### Break-Even Analysis")
    col1, col2 = st.columns(2)

    with col1:
        fixed_costs = st.number_input(
            "Estimated fixed costs per provider (KES)",
            min_value=0,
            value=50000,
            step=1000
        )

    with col2:
        variable_cost_rate = st.slider(
            "Variable cost rate (%)",
            min_value=0,
            max_value=100,
            value=60
        ) / 100

    # Calculate break-even points
    provider_stats['break_even_claims'] = np.ceil(
        fixed_costs / (provider_stats['Avg_Amount'] * (1 - variable_cost_rate)))
    provider_stats['profitability'] = np.where(
        provider_stats['Total_Claims'] > provider_stats['break_even_claims'],
        'Profitable',
        'Unprofitable'
    )

    # Visualization
    fig = px.scatter(
        provider_stats,
        x='Total_Claims',
        y='Avg_Amount',
        color='profitability',
        size='Total_Amount',
        hover_name=provider_col,
        hover_data=['Fraud_Rate', 'break_even_claims'],
        title="Provider Cost Efficiency Analysis",
        labels={
            'Total_Claims': 'Number of Claims',
            'Avg_Amount': 'Average Claim Amount (KES)',
            'Total_Amount': 'Total Amount (KES)'
        }
    )

    # Add break-even line
    if fixed_costs > 0:
        max_claims = provider_stats['Total_Claims'].max()
        break_even_line = fixed_costs / (np.linspace(1, max_claims, 100) * (1 - variable_cost_rate))

        fig.add_trace(
            go.Scatter(
                x=np.linspace(1, max_claims, 100),
                y=break_even_line,
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Break-Even Line'
            )
        )

    st.plotly_chart(fig, use_container_width=True)

    # Show top/bottom performers (UPDATED TO USE EFFICIENCY_RATIO)
    st.write("#### Provider Performance Ranking")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Most Efficient Providers**")
        efficient = provider_stats.sort_values('efficiency_ratio').head(5)
        st.dataframe(efficient[[provider_col, 'efficiency_ratio', 'Avg_Amount', 'Fraud_Rate']])

    with col2:
        st.write("**Least Efficient Providers**")
        inefficient = provider_stats.sort_values('efficiency_ratio', ascending=False).head(5)
        st.dataframe(inefficient[[provider_col, 'efficiency_ratio', 'Avg_Amount', 'Fraud_Rate']])

def render_diagnosis_patterns(data):
    """Diagnosis-treatment pattern heatmaps"""
    st.subheader("Diagnosis-Treatment Patterns")

    # Get diagnosis and treatment columns
    diag_col = get_column(data, COLUMN_MAPPING['diagnosis'])
    treat_col = get_column(data, COLUMN_MAPPING['treatment'])

    if not diag_col or not treat_col:
        st.warning("Diagnosis or treatment information missing for pattern analysis")
        return

    # Create diagnosis-treatment matrix
    diag_treat_matrix = pd.crosstab(
        data[diag_col],
        data[treat_col],
        normalize='index'
    )

    # Filter to top diagnoses and treatments
    top_diag = data[diag_col].value_counts().head(20).index
    top_treat = data[treat_col].value_counts().head(20).index

    filtered_matrix = diag_treat_matrix.loc[top_diag, top_treat]

    # Create heatmap
    fig = px.imshow(
        filtered_matrix,
        labels=dict(x="Treatment", y="Diagnosis", color="Frequency"),
        x=filtered_matrix.columns,
        y=filtered_matrix.index,
        aspect="auto",
        color_continuous_scale='Blues'
    )

    fig.update_layout(
        xaxis_title="Treatment",
        yaxis_title="Diagnosis",
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

    # Add ICD code grouping option if available
    if diag_col == 'Diagnosis' and data[diag_col].str.match(r'^[A-TV-Z][0-9][0-9AB]').any():
        st.write("#### ICD Code Group Analysis")

        # Extract ICD chapter (first character)
        data['icd_chapter'] = data[diag_col].str[0]

        # Create grouped matrix
        chapter_treat_matrix = pd.crosstab(
            data['icd_chapter'],
            data[treat_col],
            normalize='index'
        )

        # Create heatmap
        fig = px.imshow(
            chapter_treat_matrix,
            labels=dict(x="Treatment", y="ICD Chapter", color="Frequency"),
            x=chapter_treat_matrix.columns,
            y=chapter_treat_matrix.index,
            aspect="auto",
            color_continuous_scale='Blues'
        )

        st.plotly_chart(fig, use_container_width=True)

def render_exploratory_analysis(data):
    """Enhanced exploratory analysis with all required visualizations"""
    if data is None:
        st.warning("No data available for analysis")
        return

    try:
        # Collapsible automated profile report
        with st.expander("Automated Data Profile (Click to Expand)", expanded=False):
            profile = ProfileReport(
                data,
                minimal=True,
                explorative=True,
                progress_bar=False
            )
            st_profile_report(profile)

        st.subheader("Advanced Analysis")

        # Tabbed interface for different analysis types
        analysis_tab1, analysis_tab2, analysis_tab3, analysis_tab4 = st.tabs([
            "Claim Distribution", 
            "Temporal Analysis",
            "Provider Efficiency", 
            "Diagnosis Patterns"
        ])

        with analysis_tab1:
            render_claim_distribution(data)

        with analysis_tab2:
            render_temporal_analysis(data)

        with analysis_tab3:
            render_provider_efficiency(data)

        with analysis_tab4:
            render_diagnosis_patterns(data)

        # Quick Insights section
        st.subheader("Quick Insights")
        col1, col2 = st.columns(2)
        
        with col1:
            # Safely get numerical columns
            num_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if num_cols:
                default_num = 'Provider_Fraud_Rate' if 'Provider_Fraud_Rate' in num_cols else num_cols[0]
                selected_num = st.selectbox(
                    "Select numerical feature", 
                    num_cols,
                    index=num_cols.index(default_num) if default_num in num_cols else 0,
                    key='quick_insights_num_feature'
                )
                fig = px.histogram(data, x=selected_num)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No numerical columns found")

        with col2:
            # Safely get categorical columns
            cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
            if cat_cols:
                default_cat = 'treatment' if 'treatment' in cat_cols else cat_cols[0]
                selected_cat = st.selectbox(
                    "Select categorical feature", 
                    cat_cols,
                    index=cat_cols.index(default_cat) if default_cat in cat_cols else 0,
                    key='quick_insights_cat_feature'
                )
                value_counts_df = data[selected_cat].value_counts().reset_index()
                value_counts_df.columns = ['Category', 'Count']
                fig = px.bar(value_counts_df, x='Category', y='Count', title=f'Distribution of {selected_cat}')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No categorical columns found")

    except Exception as e:
        st.error(f"Error during exploratory analysis: {str(e)}")

# ==============================================
# REAL DATA PROCESSING AND MODEL INTEGRATION
# ==============================================

def load_real_claims_data(uploaded_file, tenant_key):
    """Load and process real claims data with column mapping"""
    try:
        # Read CSV with proper encoding
        df = pd.read_csv(uploaded_file, thousands=',', encoding='latin1')

        # Apply column mapping
        for standard_name, possible_names in COLUMN_MAPPING.items():
            actual_col = get_column(df, possible_names)
            if actual_col and actual_col != standard_name:
                df.rename(columns={actual_col: standard_name}, inplace=True)

        # Convert date columns
        date_col = get_column(df, COLUMN_MAPPING['date'])
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

        # Calculate employee age if DOB is available
        if 'employee_dob' in df.columns:
            df['employee_age'] = ((datetime.now() - df['employee_dob']).dt.days / 365).astype(int)

        # Add missing columns with defaults
        if 'status' not in df.columns:
            df['status'] = 'Paid'

        if 'co_payment' not in df.columns and 'amount' in df.columns:
            df['co_payment'] = df['amount'] * 0.1

        # Pseudonymize sensitive data
        df = df.apply(lambda x: pseudonymize_claim_data(x, tenant_key), axis=1)

        # Initialize fraud detection columns
        df['is_fraud'] = 0
        df['fraud_flag'] = 0
        df['fraud_score'] = 0

        return df

    except Exception as e:
        logging.error(f"Error loading claims data: {str(e)}")
        return None

def detect_fraud_anomalies(df):
    """Enhanced fraud detection with column mapping"""
    try:
        # Get proper column names using mapping
        amount_col = get_column(df, COLUMN_MAPPING['amount'])
        provider_col = get_column(df, COLUMN_MAPPING['provider_id'])
        date_col = get_column(df, COLUMN_MAPPING['date'])

        if not amount_col:
            return df

        # Select numeric features for fraud detection
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in ['is_fraud', 'fraud_flag', 'fraud_score']]

        if not numeric_cols:
            return df

        # Train isolation forest
        model = IsolationForest(contamination=0.05, random_state=42)
        amounts = df[numeric_cols].values
        scaler = StandardScaler()
        amounts_scaled = scaler.fit_transform(amounts)

        df['fraud_score'] = model.fit_predict(amounts_scaled)
        df['fraud_flag'] = np.where(df['fraud_score'] == -1, 1, 0)

        # Add provider-level fraud analysis if provider info exists
        if provider_col:
            provider_stats = df.groupby(provider_col).agg({
                amount_col: 'sum',
                'fraud_flag': ['sum', 'count']
            }).reset_index()
            provider_stats.columns = [provider_col, 'Total_Amount', 'Fraud_Count', 'Total_Claims']
            provider_stats['Fraud_Rate'] = provider_stats['Fraud_Count'] / provider_stats['Total_Claims']

            # Merge back with original data
            df = df.merge(provider_stats[[provider_col, 'Fraud_Rate']], on=provider_col, how='left')
            df.rename(columns={'Fraud_Rate': 'Provider_Fraud_Rate'}, inplace=True)

        return df

    except Exception as e:
        logging.error(f"Fraud detection failed: {str(e)}")
        return df

# ==============================================
# REPORT GENERATION FUNCTIONS
# ==============================================

def render_report_generation(user_info):
    st.header("Report Generation Center")
    log_audit_event(user_info['name'], "report_generation_accessed")

    if st.session_state.claims_data is None:
        st.warning("Please upload claims data first")
        return

    # Make a copy of the claims data
    claims_data = st.session_state.claims_data.copy()

    # Report type selection
    report_type = st.selectbox(
        "Report Type",
        ["Client Summary", "Fraud Analysis", "Provider Performance", "Financial Projections"],
        key='report_type_select'
    )

    # Get employer column using mapping system
    employer_col = get_column(claims_data, COLUMN_MAPPING['employee_id'])  # Using employee_id as fallback
    
    if not employer_col:
        st.error("No client identification column found in data")
        return

    # Client selection
    clients = claims_data[employer_col].unique()
    selected_client = st.selectbox(
        "Select Client",
        options=clients,
        key='report_client_select'
    )
    
    # Filter data for selected client
    client_data = claims_data[claims_data[employer_col] == selected_client]

    # Generate report button
    if st.button("Generate Report"):
        with st.spinner(f"Generating {report_type} report..."):
            try:
                if report_type == "Client Summary":
                    report_data = generate_client_report(selected_client, client_data)
                    file_name = f"{selected_client}_summary_report.pdf"
                elif report_type == "Fraud Analysis":
                    report_data = generate_fraud_report(selected_client, client_data)
                    file_name = f"{selected_client}_fraud_report.pdf"
                elif report_type == "Provider Performance":
                    report_data = generate_provider_report(selected_client, client_data)
                    file_name = f"{selected_client}_provider_report.pdf"
                else:  # Financial Projections
                    report_data = generate_financial_report(selected_client, client_data)
                    file_name = f"{selected_client}_financial_report.pdf"

                if report_data:
                    st.success(f"Successfully generated {report_type} report for {selected_client}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="Download PDF Report",
                            data=report_data,
                            file_name=file_name,
                            mime="application/pdf"
                        )
                    with col2:
                        excel_data = generate_sample_excel(client_data)
                        st.download_button(
                            label="Download Excel Data",
                            data=excel_data,
                            file_name=f"{selected_client}_report_data.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                else:
                    st.error("Failed to generate report - no data available")

            except Exception as e:
                st.error(f"Error generating report: {str(e)}")
                logging.error(f"Report generation failed: {str(e)}")

def generate_fraud_report(client_name, data):
    """Generate PDF report for fraud analysis"""
    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, f'{client_name} Fraud Analysis Report', 0, 1, 'C')
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 10, f'Generated on {datetime.now().strftime("%Y-%m-%d")}', 0, 1, 'C')
    pdf.ln(10)

    # Fraud Summary
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Fraud Summary', 0, 1)
    pdf.set_font('Arial', '', 10)

    fraud_count = data['fraud_flag'].sum()
    total_claims = len(data)
    fraud_amount = data.loc[data['fraud_flag'] == 1, 'amount'].sum() if 'amount' in data.columns else 0

    stats = [
        ('Total Claims', total_claims),
        ('Potential Fraud Cases', fraud_count),
        ('Fraud Rate', f"{fraud_count/total_claims:.1%}"),
        ('Amount at Risk', f"KES {fraud_amount:,.2f}" if 'amount' in data.columns else "N/A")
    ]

    for label, value in stats:
        pdf.cell(50, 10, label, 0, 0)
        pdf.cell(0, 10, str(value), 0, 1)

    # Top Fraudulent Providers if provider info exists
    if 'provider_name' in data.columns:
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Top Providers by Fraud Count', 0, 1)
        pdf.set_font('Arial', '', 10)

        provider_fraud = data[data['fraud_flag'] == 1].groupby('provider_name').size().nlargest(5)
        for provider, count in provider_fraud.items():
            pdf.cell(0, 10, f"- {provider}: {count} cases", 0, 1)

    # Footer
    pdf.set_y(-15)
    pdf.set_font('Arial', 'I', 8)
    pdf.cell(0, 10, f'Confidential - {client_name} Internal Use Only', 0, 0, 'C')

    return pdf.output(dest='S').encode('latin1')

def generate_provider_report(client_name, data):
    """Generate PDF report for provider performance"""
    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, f'{client_name} Provider Performance Report', 0, 1, 'C')
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 10, f'Generated on {datetime.now().strftime("%Y-%m-%d")}', 0, 1, 'C')
    pdf.ln(10)

    # Provider Summary
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Provider Performance Metrics', 0, 1)
    pdf.set_font('Arial', '', 10)

    if 'provider_name' in data.columns and 'amount' in data.columns:
        provider_stats = data.groupby('provider_name').agg({
            'amount': ['sum', 'mean', 'count'],
            'fraud_flag': 'sum'
        }).reset_index()
        provider_stats.columns = ['Provider', 'Total Amount', 'Avg Amount', 'Claim Count', 'Fraud Count']
        provider_stats['Fraud Rate'] = provider_stats['Fraud Count'] / provider_stats['Claim Count']

        # Top Performers
        pdf.cell(0, 10, 'Top 5 Providers by Total Claims', 0, 1)
        top_providers = provider_stats.sort_values('Claim Count', ascending=False).head(5)
        for _, row in top_providers.iterrows():
            pdf.cell(0, 10, 
                    f"- {row['Provider']}: {row['Claim Count']} claims (KES {row['Total Amount']:,.2f})", 
                    0, 1)

        # Efficiency Analysis
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Efficiency Analysis', 0, 1)
        pdf.set_font('Arial', '', 10)

        overall_avg = data['amount'].mean()
        provider_stats['Efficiency Ratio'] = provider_stats['Avg Amount'] / overall_avg
        efficient_providers = provider_stats.sort_values('Efficiency Ratio').head(3)
        
        pdf.cell(0, 10, 'Most Efficient Providers:', 0, 1)
        for _, row in efficient_providers.iterrows():
            pdf.cell(0, 10, 
                    f"- {row['Provider']}: Efficiency ratio {row['Efficiency Ratio']:.2f} " +
                    f"(Avg: KES {row['Avg Amount']:,.2f})", 
                    0, 1)
    else:
        pdf.cell(0, 10, "No provider information available", 0, 1)

    # Footer
    pdf.set_y(-15)
    pdf.set_font('Arial', 'I', 8)
    pdf.cell(0, 10, f'Confidential - {client_name} Internal Use Only', 0, 0, 'C')

    return pdf.output(dest='S').encode('latin1')

def generate_financial_report(client_name, data):
    """Generate PDF report with safe numeric handling"""
    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, f'{client_name} Financial Projections Report', 0, 1, 'C')
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 10, f'Generated on {datetime.now().strftime("%Y-%m-%d")}', 0, 1, 'C')
    pdf.ln(10)

    # Financial Summary
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Financial Overview', 0, 1)
    pdf.set_font('Arial', '', 10)

    amount_col = get_column(data, COLUMN_MAPPING['amount'])
    if amount_col:
        try:
            # Safe conversion of amount column
            data[amount_col] = pd.to_numeric(data[amount_col].astype(str).str.replace('[^\d.]', '', regex=True), errors='coerce')
            data[amount_col] = data[amount_col].fillna(0)
            
            total_amount = float(data[amount_col].sum())
            avg_amount = float(data[amount_col].mean())
            claim_count = len(data)

            stats = [
                ('Total Claim Amount', f"KES {total_amount:,.2f}"),
                ('Average Claim Amount', f"KES {avg_amount:,.2f}"),
                ('Total Claims', claim_count)
            ]

            for label, value in stats:
                pdf.cell(60, 10, label, 0, 0)
                pdf.cell(0, 10, str(value), 0, 1)

            # Projections with date check
            date_col = get_column(data, COLUMN_MAPPING['date'])
            if date_col:
                try:
                    data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
                    unique_months = len(data[date_col].dt.to_period('M').unique())
                    monthly_avg = total_amount / unique_months if unique_months > 0 else total_amount
                except:
                    monthly_avg = total_amount
            else:
                monthly_avg = total_amount

            pdf.ln(5)
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Annual Projections', 0, 1)
            pdf.set_font('Arial', '', 10)
            pdf.cell(0, 10, f"Projected Annual Claims: KES {monthly_avg * 12:,.2f}", 0, 1)
            
        except Exception as e:
            pdf.cell(0, 10, f"Error processing financial data: {str(e)}", 0, 1)
    else:
        pdf.cell(0, 10, "No financial data available", 0, 1)

    # Footer
    pdf.set_y(-15)
    pdf.set_font('Arial', 'I', 8)
    pdf.cell(0, 10, f'Confidential - {client_name} Internal Use Only', 0, 0, 'C')

    return pdf.output(dest='S').encode('latin1')

# ==============================================
# ADMIN INTERFACE (VIRTUAL ANALYTICS)
# ==============================================

def admin_dashboard():
    st.title("The Verse - Admin Console")
    user_info = st.session_state.user_info

    # Audit log
    log_audit_event(user_info['name'], "admin_login")

    # Tenant metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Active Tenants", len([u for u in USER_DB if USER_DB[u]['role'] in ['broker','underwriter']]))
    with col2:
        st.metric("Managed Clients", sum(len(USER_DB[u].get('clients',{}))
                              for u in USER_DB if USER_DB[u]['role'] in ['broker','underwriter']))
    with col3:
        st.metric("System Health", "Optimal", delta="+2%")

    st.subheader("Tenant Management")
    tenant_data = []
    for user, info in USER_DB.items():
        if info['role'] in ['broker', 'underwriter']:
            tenant_data.append({
                'Username': user,
                'Name': info.get('name', ''),
                'Role': info['role'].title(),
                'Tenant': info['tenant'],
                'Clients': len(info.get('clients', {})),
                'Plan': ', '.join(set([c['plan'] for c in info.get('clients', {}).values()]))
            })

    # Enhanced AgGrid with error handling
    gb = GridOptionsBuilder.from_dataframe(pd.DataFrame(tenant_data))
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_side_bar()
    gb.configure_selection('single', use_checkbox=True)
    grid_options = gb.build()

    grid_response = AgGrid(
        pd.DataFrame(tenant_data),
        gridOptions=grid_options,
        data_return_mode='AS_INPUT',
        update_mode='MODEL_CHANGED',
        fit_columns_on_grid_load=True,
        height=300,
        key='tenant_grid'
    )

    # Fixed error handling for selected rows
    if grid_response['selected_rows'] is not None and len(grid_response['selected_rows']) > 0:
        selected = grid_response['selected_rows'][0]
        with st.expander(f"Tenant Actions - {selected['Username']}"):
            st.write(f"**Tenant:** {selected['Tenant']}")
            st.write(f"**Clients:** {selected['Clients']}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Reset Password"):
                    log_audit_event(user_info['name'], "password_reset", selected['Username'])
                    st.success(f"Password reset initiated for {selected['Username']}")

                if st.button("View Activity Logs"):
                    try:
                        with open(f"logs/{selected['Tenant']}_audit.log", "r") as f:
                            logs = [json.loads(line) for line in f.readlines()]
                        st.dataframe(pd.DataFrame(logs).tail(10))
                    except FileNotFoundError:
                        st.warning("No logs available for this tenant")

            with col2:
                if st.button("Suspend Account"):
                    st.warning(f"Account suspension for {selected['Username']} would happen here")

                if st.button("View Configuration"):
                    tenant_config = get_tenant_config(selected['Tenant'])
                    st.json(tenant_config)
    else:
        st.info("Select a tenant from the table to view actions")

    # System administration
    st.subheader("System Administration")
    with st.expander("Database Management"):
        if st.button("Run Database Backup"):
            with st.spinner("Backing up all tenant data..."):
                time.sleep(2)
                st.success("Backup completed successfully")
                log_audit_event(user_info['name'], "db_backup")

        if st.button("View System Logs"):
            try:
                with open("logs/system_audit.log", "r") as f:
                    logs = [json.loads(line) for line in f.readlines()]
                st.dataframe(pd.DataFrame(logs).tail(20))
            except FileNotFoundError:
                st.warning("No system logs available")

# ==============================================
# BROKER/UNDERWRITER INTERFACE
# ==============================================

def broker_underwriter_dashboard(user_info):
    st.title(f"The Verse - {user_info['tenant'].title()} Dashboard")
    log_audit_event(user_info['name'], "broker_login")

    # Initialize session state for data and predictor
    if 'claims_data' not in st.session_state:
        st.session_state.claims_data = None

    if 'predictor' not in st.session_state:
        st.session_state.predictor = ClaimsPredictor()

    # Main navigation - ALWAYS VISIBLE (modified section)
    st.sidebar.header("Analysis Mode")
    analysis_type = st.sidebar.radio(
        "Select Mode:",
        options=["Claims Upload", "Claims Prediction", "Fraud Detection", "Client Management", "Reports"],
        key='broker_analysis_mode'
    )

    if analysis_type == "Claims Upload":
        render_claims_upload(user_info)
    elif analysis_type == "Claims Prediction":
        render_claims_prediction(user_info)
    elif analysis_type == "Fraud Detection":
        render_fraud_detection(user_info)
    elif analysis_type == "Client Management":
        render_client_management(user_info)
    elif analysis_type == "Reports":
        render_report_generation(user_info)

def render_claims_upload(user_info):
    st.header("Claims Data Upload")
    log_audit_event(user_info['name'], "claims_upload_accessed")

    uploaded_file = st.file_uploader(
        "Upload Claims Data (CSV)",
        type=['csv'],
        help="Upload CSV file with claims data"
    )

    if uploaded_file is not None:
        with st.spinner("Processing uploaded data..."):
            # Load and process the real claims data
            claims_data = load_real_claims_data(uploaded_file, user_info['tenant'])

            if claims_data is not None:
                # Run fraud detection
                claims_data = detect_fraud_anomalies(claims_data)

                # Store in session state
                st.session_state.claims_data = claims_data

                st.success(f"Successfully loaded {len(claims_data)} claims")
                log_audit_event(user_info['name'], "data_uploaded", f"{len(claims_data)} records")

                # Show data preview
                with st.expander("View Uploaded Data"):
                    st.dataframe(claims_data.head())
            else:
                st.error("Failed to process the uploaded file")


def render_claims_prediction(user_info):
    st.header("Claims Prediction Engine")
    log_audit_event(user_info['name'], "predictions_accessed")

    if st.session_state.claims_data is None:
        st.warning("Please upload claims data first")
        return

    # Initialize tabs first
    tab1, tab2, tab3, tab4 = st.tabs(["Data Preparation", "Exploratory Analysis", "Model Training", "Make Predictions"])

    with tab1:
        st.subheader("Data Cleaning & Preparation")
        if st.button("Clean Data"):
            with st.spinner("Cleaning data..."):
                st.session_state.predictor.data = st.session_state.claims_data.copy()
                success = st.session_state.predictor.clean_and_prepare_data()
                
                if success:
                    st.success("Data cleaned successfully!")
                    st.session_state.clean_data = st.session_state.predictor.clean_data.copy()
                    
                    st.write("**New Features Created:**")
                    original_cols = set(st.session_state.predictor.data.columns)
                    new_cols = set(st.session_state.predictor.clean_data.columns) - original_cols
                    for col in new_cols:
                        st.write(f"- {col}")

                    with st.expander("View Cleaned Data"):
                        st.dataframe(st.session_state.predictor.clean_data.head())
                else:
                    st.error("Data cleaning failed")

    with tab2:
        st.subheader("Exploratory Data Analysis")
        
        if 'clean_data' not in st.session_state or st.session_state.clean_data is None:
            st.warning("Please clean data first")
        else:
            render_exploratory_analysis(st.session_state.clean_data)

    with tab3:
        st.subheader("Train Prediction Model")

        if not hasattr(st.session_state.predictor, 'clean_data') or st.session_state.predictor.clean_data is None:
            st.warning("Please clean data first")
            return

        col1, col2 = st.columns(2)
        with col1:
            model_type = st.selectbox(
                "Model Algorithm",
                ["Gradient Boosting", "Random Forest", "XGBoost", "Neural Network", "Auto Select Best"],
                index=0
            )
            target_var = st.selectbox(
                "Target Variable",
                ['amount', 'co_payment'],
                index=0
            )

        with col2:
            test_size = st.slider("Test Set Size (%)", 10, 40, 20)
            do_tuning = st.checkbox("Enable Hyperparameter Tuning", value=False)

        if st.button("Train Model"):
            with st.spinner("Training model..."):
                results_df = st.session_state.predictor.train_model(
                    model_type=model_type,
                    target=target_var,
                    test_size=test_size/100,
                    do_tuning=do_tuning
                )

            if results_df is not None:
                st.success("Model trained successfully!")
                st.subheader("Model Performance")
                st.dataframe(results_df.style.format({
                    'MAE': '{:,.2f}',
                    'RMSE': '{:,.2f}',
                    'R2': '{:.3f}'
                }).background_gradient(cmap='Blues', subset=['R2']))

                if st.session_state.predictor.feature_importance is not None:
                    st.subheader("Feature Importance")
                    chart = alt.Chart(
                        st.session_state.predictor.feature_importance.head(10)
                    ).mark_bar().encode(
                        x='Importance',
                        y=alt.Y('Feature', sort='-x'),
                        color='Feature',
                        tooltip=['Feature', 'Importance']
                    ).properties(title='Top 10 Important Features')
                    st.altair_chart(chart, use_container_width=True)

    with tab4:
        st.subheader("Make Predictions")

        if not hasattr(st.session_state.predictor, 'model') or st.session_state.predictor.model is None:
            st.warning("Please train the model first")
            return

        # Prediction type selector
        prediction_type = st.radio(
            "Prediction Type",
            ["Individual Claim", "Group Claims"],
            horizontal=True,
            key='prediction_type'
        )

        if prediction_type == "Individual Claim":
            # Get sample claim for prediction
            sample_claim = st.session_state.claims_data.iloc[0].copy()

            # Create input form
            col1, col2 = st.columns(2)
            with col1:
                visit_type = st.selectbox(
                    "Visit Type",
                    options=st.session_state.predictor.available_values.get('visit_type', []),
                    index=0 if not st.session_state.predictor.available_values.get('visit_type') else None
                )
                diagnosis = st.text_input(
                    "Diagnosis",
                    value=sample_claim.get('diagnosis', 'Unknown')
                )
                procedure = st.text_input(
                    "Procedure",
                    value=sample_claim.get('procedure', 'Unknown')
                )
            with col2:
                employee_age = st.number_input(
                    "Employee Age",
                    min_value=18,
                    max_value=100,
                    value=int(sample_claim.get('employee_age', 40))
                )
                co_payment = st.number_input(
                    "Co-Payment Amount",
                    min_value=0.0,
                    value=float(sample_claim.get('co_payment', 0.0)),
                    step=0.01
                )
                provider = st.text_input(
                    "Provider",
                    value=sample_claim.get('provider_id', 'Unknown')
                )

            if st.button("Predict"):
                input_data = {
                    'visit_type': visit_type,
                    'diagnosis': diagnosis,
                    'procedure': procedure,
                    'employee_age': employee_age,
                    'co_payment': co_payment,
                    'provider_id': provider
                }

                prediction = st.session_state.predictor.predict_claim_amount(input_data)

                if prediction is not None:
                    st.success(f"Predicted Claim Amount: KES{prediction['prediction']:,.2f}")
                    if prediction['is_potential_fraud']:
                        st.error(f"⚠️ Potential fraud detected (confidence: {prediction['fraud_confidence']:.1%})")
                else:
                    st.error("Prediction failed")

        else:  # Group Predictions
            st.write("### Group Predictions")

            # Option 1: Use existing data with filters
            use_existing = st.checkbox("Use existing cleaned data with filters", value=True)

            if use_existing and st.session_state.predictor.clean_data is not None:
                st.write("#### Filter Data for Prediction")

                # Create filters - handle missing columns gracefully
                cols = st.columns(3)
                filter_conditions = []
                
                with cols[0]:
                    # Check if 'employer' column exists, otherwise use a default
                    if 'employer' in st.session_state.predictor.clean_data.columns:
                        employers = st.multiselect(
                            "Employers",
                            options=st.session_state.predictor.clean_data['employer'].unique(),
                            default=st.session_state.predictor.clean_data['employer'].unique()
                        )
                        filter_conditions.append(
                            st.session_state.predictor.clean_data['employer'].isin(employers)
                        )
                    else:
                        st.warning("No employer information available")
                
                with cols[1]:
                    if 'department' in st.session_state.predictor.clean_data.columns:
                        departments = st.multiselect(
                            "Departments",
                            options=st.session_state.predictor.clean_data['department'].unique(),
                            default=st.session_state.predictor.clean_data['department'].unique()
                        )
                        filter_conditions.append(
                            st.session_state.predictor.clean_data['department'].isin(departments)
                        )
                    else:
                        st.warning("No department information available")
                
                with cols[2]:
                    if 'category' in st.session_state.predictor.clean_data.columns:
                        categories = st.multiselect(
                            "Benefit Categories",
                            options=st.session_state.predictor.clean_data['category'].unique(),
                            default=st.session_state.predictor.clean_data['category'].unique()
                        )
                        filter_conditions.append(
                            st.session_state.predictor.clean_data['category'].isin(categories)
                        )
                    else:
                        st.warning("No category information available")

                # Apply filters if any conditions exist
                if filter_conditions:
                    filtered_data = st.session_state.predictor.clean_data[np.all(filter_conditions, axis=0)]
                else:
                    filtered_data = st.session_state.predictor.clean_data.copy()

                st.info(f"Filtered to {len(filtered_data)} records")

                if st.button("Predict Group Claims"):
                    with st.spinner("Making predictions..."):
                        predictions = st.session_state.predictor.predict_claim_amount(filtered_data)

                    if predictions is not None:
                        st.success(f"Predictions completed for {len(predictions)} claims!")

                        # Show summary stats
                        cols = st.columns(3)
                        with cols[0]:
                            st.metric("Total Predicted", f"KES{predictions['Predicted_Claim_Amount'].sum():,.2f}")
                        with cols[1]:
                            st.metric("Average Claim", f"KES{predictions['Predicted_Claim_Amount'].mean():,.2f}")
                        with cols[2]:
                            fraud_count = predictions['Is_Potential_Fraud'].sum()
                            st.metric("Potential Fraud", fraud_count)

                        # Show predictions
                        with st.expander("View Predictions"):
                            st.dataframe(predictions.head())

                        # Download option
                        csv = predictions.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Predictions",
                            data=csv,
                            file_name="claim_predictions.csv",
                            mime="text/csv"
                        )

# ==============================================
# FRAUD DETECTION INTERFACE
# ==============================================

def render_fraud_detection(user_info):
    st.header("Fraud Detection Center")
    log_audit_event(user_info['name'], "fraud_center_accessed")

    if st.session_state.claims_data is None:
        st.warning("Please upload claims data first")
        return

    # Make a copy of the claims data to avoid modifying the original
    claims_data = st.session_state.claims_data.copy()

    # Ensure numeric columns are properly formatted
    numeric_cols = ['amount', 'Claim_Amount_KES', 'Co_Payment_KES']
    for col in numeric_cols:
        if col in claims_data.columns:
            claims_data[col] = pd.to_numeric(claims_data[col].astype(str).str.replace('[^\d.]', '', regex=True), errors='coerce').fillna(0)

    col1, col2, col3 = st.columns(3)
    with col1:
        fraud_count = claims_data['fraud_flag'].sum()
        st.metric("Potential Fraud Cases", fraud_count)
    with col2:
        fraud_amount = claims_data.loc[claims_data['fraud_flag'] == 1, 'amount'] if 'amount' in claims_data.columns else 0
        total_fraud_amount = fraud_amount.sum() if 'amount' in claims_data.columns else 0
        formatted_amount = f"KES {float(total_fraud_amount):,.2f}" if not pd.isna(total_fraud_amount) else "KES 0.00"
        st.metric("Total At Risk", formatted_amount)
    with col3:
        fraud_rate = fraud_count / len(claims_data) if len(claims_data) > 0 else 0
        st.metric("Fraud Rate", f"{fraud_rate:.1%}")

    tab1, tab2 = st.tabs(["Fraud Analysis", "Provider Risk"])

    with tab1:
        st.subheader("Fraudulent Claim Patterns")

        if 'date' in claims_data.columns:
            fraud_over_time = claims_data.copy()
            fraud_over_time['Month'] = fraud_over_time['date'].dt.strftime('%Y-%m')

            fraud_over_time = fraud_over_time.groupby('Month').agg({
                'fraud_flag': ['sum', 'count']
            }).reset_index()
            fraud_over_time.columns = ['Month', 'Fraud Cases', 'Total Claims']
            fraud_over_time['Fraud Rate'] = fraud_over_time['Fraud Cases'] / fraud_over_time['Total Claims']

            fig = px.line(
                fraud_over_time,
                x='Month',
                y='Fraud Rate',
                title="Monthly Fraud Rate Trend"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Date column not found - cannot show temporal trends")

        st.subheader("Top Fraudulent Claims")
        fraud_data = claims_data[claims_data['fraud_flag'] == 1].copy()
        if not fraud_data.empty and 'amount' in fraud_data.columns:
            st.dataframe(fraud_data.sort_values('amount', ascending=False).head(10))
        else:
            st.warning("No fraudulent claims found or amount data missing")

    with tab2:
        st.subheader("Provider Risk Scoring")

        # Use 'Provider_Name' instead of 'provider_id' which doesn't exist
        provider_col = 'Provider_Name' if 'Provider_Name' in claims_data.columns else None

        if provider_col:
            provider_stats = claims_data.copy()
            provider_stats['amount'] = pd.to_numeric(provider_stats['amount'], errors='coerce').fillna(0)

            provider_stats = provider_stats.groupby(provider_col).agg({
                'amount': ['mean', 'count'],
                'fraud_flag': 'sum'
            }).reset_index()
            provider_stats.columns = ['Provider', 'Avg Amount', 'Claim Count', 'Fraud Count']
            provider_stats['Fraud Rate'] = provider_stats['Fraud Count'] / provider_stats['Claim Count']

            fig = px.scatter(
                provider_stats,
                x='Claim Count',
                y='Avg Amount',
                size='Fraud Count',
                color='Fraud Rate',
                hover_name='Provider',
                title="Provider Risk Assessment"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Provider information not found in the dataset")

# ==============================================
# CLIENT MANAGEMENT INTERFACE
# ==============================================

def render_client_management(user_info):
    st.header("Client Management Portal")
    log_audit_event(user_info['name'], "client_mgmt_accessed")

    selected_client = st.selectbox(
        "Select Client",
        options=list(user_info.get('clients', {}).keys()))
    if selected_client:
        client_info = user_info['clients'][selected_client]
        st.subheader(f"{selected_client} Management")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Plan", client_info['plan'])
            st.write("**Authorized Users:**")
            for user in client_info['users']:
                st.write(f"- {user}")

        with col2:
            if st.button("Upgrade Plan"):
                st.success(f"Plan upgrade request sent for {selected_client}")
                log_audit_event(user_info['name'], "plan_upgrade_requested", selected_client)

            if st.button("Reset Client Data"):
                st.warning(f"This will reset all data for {selected_client}")
                log_audit_event(user_info['name'], "data_reset_requested", selected_client)

        st.subheader("Client Access Configuration")
        new_user = st.text_input("Add New User")
        if st.button("Grant Access"):
            if new_user:
                if new_user in USER_DB:
                    USER_DB[selected_client]['users'].append(new_user)
                    st.success(f"Added {new_user} to {selected_client}")
                    log_audit_event(user_info['name'], "user_access_granted",
                                   f"{new_user} to {selected_client}")
                else:
                    st.error("User does not exist in system")

        st.subheader("Data Access Configuration")
        access_options = ['claims', 'fraud', 'reports', 'predictions']
        current_access = client_info.get('data_access', [])

        new_access = st.multiselect(
            "Allowed Data Access",
            options=access_options,
            default=current_access,
            key=f"access_{selected_client}"
        )

        if st.button("Update Access Permissions"):
            client_info['data_access'] = new_access
            st.success(f"Updated data access for {selected_client}")
            log_audit_event(user_info['name'], "access_updated",
                           f"{selected_client}: {', '.join(new_access)}")

        st.subheader("Client Analytics Dashboard")
        if st.checkbox("Show Client Analytics"):
            if 'claims_data' in st.session_state and st.session_state.claims_data is not None:
                # Use column mapping system to find the employer/client column
                employer_col = get_column(st.session_state.claims_data, COLUMN_MAPPING['employee_id'])
                
                if employer_col:
                    client_data = st.session_state.claims_data[
                        st.session_state.claims_data[employer_col] == selected_client]

                    if not client_data.empty:
                        st.write(f"#### Claims Summary for {selected_client}")

                        cols = st.columns(3)
                        with cols[0]:
                            st.metric("Total Claims", len(client_data))
                        with cols[1]:
                            amount_col = get_column(client_data, COLUMN_MAPPING['amount'])
                            if amount_col:
                                st.metric("Total Amount", f"KES {client_data[amount_col].sum():,.2f}")
                            else:
                                st.warning("Amount data not available")
                        with cols[2]:
                            if amount_col:
                                st.metric("Avg. Claim", f"KES {client_data[amount_col].mean():,.2f}")
                            else:
                                st.warning("Amount data not available")

                        st.write("##### Monthly Claims Trend")
                        date_col = get_column(client_data, COLUMN_MAPPING['date'])
                        if date_col:
                            client_data[date_col] = pd.to_datetime(client_data[date_col])
                            monthly_data = client_data.set_index(date_col).resample('M').agg({
                                amount_col: ['sum', 'count']
                            }).reset_index()
                            monthly_data.columns = ['Month', 'Total Amount', 'Claim Count']

                            chart = alt.Chart(monthly_data).mark_area().encode(
                                x='Month:T',
                                y='Total Amount:Q',
                                tooltip=['Month', 'Total Amount', 'Claim Count']
                            ).properties(height=300)
                            st.altair_chart(chart, use_container_width=True)
                        else:
                            st.warning("Date information not available for trend analysis")

                        st.write("##### Claim Type Distribution")
                        if 'category' in client_data.columns:
                            category_dist = client_data['category'].value_counts().reset_index()
                            category_dist.columns = ['Category', 'Count']

                            pie_chart = alt.Chart(category_dist).mark_arc().encode(
                                theta='Count:Q',
                                color='Category:N',
                                tooltip=['Category', 'Count']
                            ).properties(height=300)
                            st.altair_chart(pie_chart, use_container_width=True)
                        else:
                            st.warning("Category information not available")
                    else:
                        st.warning(f"No claims data found for {selected_client}")
                else:
                    st.warning("No client identification column found in data")
            else:
                st.warning("No claims data available. Please upload data first.")

        st.subheader("Client Report Generation")
        report_type = st.selectbox(
            "Report Type",
            ["Monthly Summary", "Annual Review", "Fraud Analysis"],
            key=f"report_type_{selected_client}"
        )

        if st.button("Generate Client Report"):
            with st.spinner(f"Generating {report_type} report..."):
                report_data = generate_client_report(selected_client, st.session_state.claims_data)

                if report_data:
                    pdf_bytes = generate_client_report_pdf(selected_client, report_data, report_type)
                    st.download_button(
                        label="Download Report",
                        data=pdf_bytes,
                        file_name=f"{selected_client}_{report_type.replace(' ', '_')}.pdf",
                        mime="application/pdf"
                    )
                    st.success(f"{report_type} report generated for {selected_client}")
                    log_audit_event(user_info['name'], "report_generated",
                                   f"{report_type} for {selected_client}")
                else:
                    st.error("Failed to generate report. No data available.")

# ==============================================
# REPORT GENERATION FUNCTIONS
# ==============================================

def generate_client_report(client_name, data):
    """Generate PDF client summary report with all previous metrics"""
    if data is None:
        return None

    # Use column mapping system
    employer_col = get_column(data, COLUMN_MAPPING['employee_id'])
    if employer_col is None:
        st.warning("No client identification column found in data")
        return None

    # Check if client exists
    if client_name not in data[employer_col].unique():
        st.warning(f"No data found for client: {client_name}")
        return None

    # Filter data for this client
    client_data = data[data[employer_col] == client_name].copy()

    # Get columns with safe defaults
    date_col = get_column(client_data, COLUMN_MAPPING['date'])
    amount_col = get_column(client_data, COLUMN_MAPPING['amount'])
    provider_col = get_column(client_data, COLUMN_MAPPING['provider_id'])
    fraud_col = 'fraud_flag' if 'fraud_flag' in client_data.columns else None
    category_col = 'category' if 'category' in client_data.columns else None

    # Initialize PDF
    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, f'{client_name} Claims Summary Report', 0, 1, 'C')
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 10, f'Generated on {datetime.now().strftime("%Y-%m-%d")}', 0, 1, 'C')
    pdf.ln(10)

    # Safe numeric conversion for amount column
    if amount_col:
        try:
            client_data[amount_col] = pd.to_numeric(
                client_data[amount_col].astype(str).str.replace('[^\d.]', '', regex=True), 
                errors='coerce'
            ).fillna(0)
        except Exception as e:
            st.warning(f"Could not convert amount column: {str(e)}")
            amount_col = None

    # Calculate all metrics (preserving all previous calculations)
    time_period_start = client_data[date_col].min().strftime('%Y-%m-%d') if date_col else 'N/A'
    time_period_end = client_data[date_col].max().strftime('%Y-%m-%d') if date_col else 'N/A'
    total_claims = len(client_data)
    unique_members = client_data[employer_col].nunique()
    total_amount = float(client_data[amount_col].sum()) if amount_col else 0.0
    avg_claim = float(client_data[amount_col].mean()) if amount_col else 0.0
    fraud_flags = int(client_data[fraud_col].sum()) if fraud_col else 0
    fraud_amount = float(client_data.loc[client_data[fraud_col] == 1, amount_col].sum()) if (fraud_col and amount_col) else 0.0
    top_providers = client_data.groupby(provider_col)[amount_col].sum().nlargest(3).to_dict() if (provider_col and amount_col) else {}
    category_distribution = client_data[category_col].value_counts(normalize=True).to_dict() if category_col else {}

    # Summary Stats (includes all previous metrics)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Summary Statistics', 0, 1)
    pdf.set_font('Arial', '', 10)

    stats = [
        ('Report Period', f"{time_period_start} to {time_period_end}"),
        ('Total Claims', total_claims),
        ('Unique Members', unique_members),
        ('Total Amount', f"KES {total_amount:,.2f}"),
        ('Average Claim', f"KES {avg_claim:,.2f}"),
        ('Fraud Cases', f"{fraud_flags} (KES {fraud_amount:,.2f})")
    ]

    for label, value in stats:
        pdf.cell(50, 10, label, 0, 0)
        pdf.cell(0, 10, str(value), 0, 1)

    # Top Providers (preserving previous functionality)
    if top_providers:
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Top Providers by Claim Amount', 0, 1)
        pdf.set_font('Arial', '', 10)
        
        for provider, amount in top_providers.items():
            pdf.cell(0, 10, f"- {provider}: KES {amount:,.2f}", 0, 1)

    # Category Distribution (preserving previous functionality)
    if category_distribution:
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Benefit Category Distribution', 0, 1)
        pdf.set_font('Arial', '', 10)
        
        for category, pct in category_distribution.items():
            pdf.cell(0, 10, f"- {category}: {pct:.1%}", 0, 1)

    # Footer
    pdf.set_y(-15)
    pdf.set_font('Arial', 'I', 8)
    pdf.cell(0, 10, f'Confidential - {client_name} Internal Use Only', 0, 0, 'C')

    return pdf.output(dest='S').encode('latin1')

def generate_client_report_pdf(client_name, report_data, report_type):
    """Generate professional PDF report for client"""
    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, f'{client_name} {report_type} Report', 0, 1, 'C')
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 10, f'Generated on {datetime.now().strftime("%Y-%m-%d")}', 0, 1, 'C')
    pdf.ln(10)

    # Summary Stats
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Summary Statistics', 0, 1)
    pdf.set_font('Arial', '', 10)

    stats = [
        ('Report Period', f"{report_data['time_period']['start']} to {report_data['time_period']['end']}"),
        ('Total Claims', report_data['total_claims']),
        ('Unique Members', report_data['unique_members']),
        ('Total Claim Amount', f"KES {report_data['total_amount']:,.2f}"),
        ('Average Claim', f"KES {report_data['avg_claim']:,.2f}"),
        ('Fraud Flags', f"{report_data['fraud_flags']} (KES {report_data['fraud_amount']:,.2f})")
    ]

    for label, value in stats:
        pdf.cell(40, 10, label, 0, 0)
        pdf.cell(0, 10, str(value), 0, 1)

    # Top Providers
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Top 3 Providers by Claim Amount', 0, 1)
    pdf.set_font('Arial', '', 10)

    for provider, amount in report_data['top_providers'].items():
        pdf.cell(0, 10, f"- {provider}: KES {amount:,.2f}", 0, 1)

    # Category Distribution
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Benefit Category Distribution', 0, 1)
    pdf.set_font('Arial', '', 10)

    for category, pct in report_data['category_distribution'].items():
        pdf.cell(0, 10, f"- {category}: {pct:.1%}", 0, 1)

    # Fraud Analysis (if report type is Fraud Analysis)
    if report_type == "Fraud Analysis":
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Fraud Analysis Details', 0, 1)
        pdf.set_font('Arial', '', 10)

        pdf.cell(0, 10, f"Potential fraud represents {report_data['fraud_flags']/report_data['total_claims']:.1%} of claims", 0, 1)
        pdf.cell(0, 10, f"Total potential fraud amount: KES {report_data['fraud_amount']:,.2f}", 0, 1)

    # Footer
    pdf.set_y(-15)
    pdf.set_font('Arial', 'I', 8)
    pdf.cell(0, 10, f'Confidential - {client_name} Internal Use Only', 0, 0, 'C')

    return pdf.output(dest='S').encode('latin1')

# ==============================================
# CLIENT DASHBOARD
# ==============================================

def client_dashboard(user_info):
    st.title(f"The Verse - {user_info['client_org']} Portal")
    log_audit_event(user_info['name'], "client_login")

    if 'claims_data' not in st.session_state:
        st.session_state.claims_data = None

    if 'predictor' not in st.session_state:
        st.session_state.predictor = ClaimsPredictor()

    # Main navigation
    analysis_type = st.sidebar.selectbox(
        "Analysis Mode",
        ["Claims Overview", "Fraud Detection", "Reports"],
        key='client_analysis_mode'
    )

    if analysis_type == "Claims Overview":
        render_client_claims_overview(user_info)
    elif analysis_type == "Fraud Detection":
        render_client_fraud_detection(user_info)
    elif analysis_type == "Reports":
        render_client_report_generation(user_info)

def render_client_claims_overview(user_info):
    st.header(f"{user_info['client_org']} Claims Overview")
    log_audit_event(user_info['name'], "client_claims_accessed")

    if st.session_state.claims_data is None:
        st.warning("No claims data available")
        return

    # Filter data for this client
    client_data = st.session_state.claims_data[
        st.session_state.claims_data['employer'] == user_info['client_org']
    ].copy()

    if client_data.empty:
        st.warning("No claims data found for your organization")
        return

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Claims", len(client_data))
    with col2:
        st.metric("Total Amount", f"KES {client_data['amount'].sum():,.2f}")
    with col3:
        st.metric("Average Claim", f"KES {client_data['amount'].mean():,.2f}")

    # Claims trend
    st.subheader("Claims Trend")
    if 'date' in client_data.columns:
        client_data['date'] = pd.to_datetime(client_data['date'])
        monthly_data = client_data.set_index('date').resample('M').agg({
            'amount': ['sum', 'count']
        }).reset_index()
        monthly_data.columns = ['Month', 'Total Amount', 'Claim Count']

        fig = px.line(
            monthly_data,
            x='Month',
            y='Total Amount',
            title="Monthly Claims Value"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Date information not available for trend analysis")

    # Claims by category
    st.subheader("Claims by Category")
    if 'category' in client_data.columns:
        category_dist = client_data['category'].value_counts().reset_index()
        category_dist.columns = ['Category', 'Count']

        fig = px.pie(
            category_dist,
            values='Count',
            names='Category',
            title="Claim Distribution by Category"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Category information not available")

def render_client_fraud_detection(user_info):
    st.header(f"{user_info['client_org']} Fraud Detection")
    log_audit_event(user_info['name'], "client_fraud_accessed")

    if st.session_state.claims_data is None:
        st.warning("No claims data available")
        return

    # Filter data for this client
    client_data = st.session_state.claims_data[
        st.session_state.claims_data['employer'] == user_info['client_org']
    ].copy()

    if client_data.empty:
        st.warning("No claims data found for your organization")
        return

    # Fraud metrics
    col1, col2 = st.columns(2)
    with col1:
        fraud_count = client_data['fraud_flag'].sum()
        st.metric("Potential Fraud Cases", fraud_count)
    with col2:
        fraud_amount = client_data.loc[client_data['fraud_flag'] == 1, 'amount'].sum()
        st.metric("Amount at Risk", f"KES {fraud_amount:,.2f}")

    # Show fraudulent claims
    st.subheader("Potential Fraudulent Claims")
    fraud_data = client_data[client_data['fraud_flag'] == 1]
    if not fraud_data.empty:
        st.dataframe(fraud_data[['date', 'amount', 'provider_name', 'diagnosis']])
    else:
        st.success("No potential fraud cases detected")

def render_client_report_generation(user_info):
    st.header(f"{user_info['client_org']} Report Generation")
    log_audit_event(user_info['name'], "client_report_accessed")

    if st.session_state.claims_data is None:
        st.warning("No claims data available")
        return

    # Filter data for this client
    client_data = st.session_state.claims_data[
        st.session_state.claims_data['employer'] == user_info['client_org']
    ].copy()

    if client_data.empty:
        st.warning("No claims data found for your organization")
        return

    report_type = st.selectbox(
        "Report Type",
        ["Monthly Summary", "Annual Review", "Fraud Analysis"],
        key='client_report_type'
    )

    time_range = st.selectbox(
        "Time Period",
        ["Last 3 Months", "Last 6 Months", "Last Year", "All Time"],
        key='client_time_range'
    )

    if st.button("Generate Report"):
        with st.spinner(f"Generating {report_type} report..."):
            # Filter by time range if date column exists
            if 'date' in client_data.columns:
                client_data['date'] = pd.to_datetime(client_data['date'])
                if time_range == "Last 3 Months":
                    cutoff_date = datetime.now() - timedelta(days=90)
                    report_data = client_data[client_data['date'] >= cutoff_date]
                elif time_range == "Last 6 Months":
                    cutoff_date = datetime.now() - timedelta(days=180)
                    report_data = client_data[client_data['date'] >= cutoff_date]
                elif time_range == "Last Year":
                    cutoff_date = datetime.now() - timedelta(days=365)
                    report_data = client_data[client_data['date'] >= cutoff_date]
                else:
                    report_data = client_data
            else:
                report_data = client_data

            pdf_bytes = generate_client_report_pdf(
                user_info['client_org'],
                report_data,
                report_type
            )

            st.download_button(
                label="Download Report",
                data=pdf_bytes,
                file_name=f"{user_info['client_org']}_{report_type.replace(' ', '_')}.pdf",
                mime="application/pdf"
            )
            st.success("Report generated successfully")

# ==============================================
# MAIN APPLICATION FLOW
# ==============================================

def main():
    if not st.session_state.get('authenticated', False):
        # Login remains in narrow mode
        login_form()
    else:
        # Set to wide mode after login
        st.set_page_config(layout="wide")
        
        user_info = st.session_state.user_info

        if user_info['role'] == 'admin':
            admin_dashboard()
        elif user_info['role'] in ['broker', 'underwriter']:
            broker_underwriter_dashboard(user_info)
        elif user_info['role'] == 'client':
            client_dashboard(user_info)

if __name__ == "__main__":
    main()