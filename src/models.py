"""
models.py - Unified forecasting suite for Pune Hospital Demand Forecasting:
1. Seasonal Naive Baseline
2. Holt-Winters Exponential Smoothing (ETS) / SARIMA
3. Prophet (with custom Pune festival regressors & monsoon curve)
4. LightGBM Machine Learning model (with autoregressive lags & rolling features)
"""

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import lightgbm as lgb

warnings.filterwarnings("ignore")

class SeasonalNaiveModel:
    """
    Seasonal Naive Baseline:
    Forecasts future values using the observation from the same week in the previous year (s=52).
    If historical data is less than 52 weeks or fallback needed, uses previous cycle or rolling median.
    """
    def __init__(self, season_length=52):
        self.season_length = season_length
        self.history = None
        self.residuals_std = None
        
    def fit(self, y_train):
        self.history = np.array(y_train, dtype=float)
        if len(self.history) >= self.season_length:
            # Empirical residuals on seasonal differences
            diffs = self.history[self.season_length:] - self.history[:-self.season_length]
            self.residuals_std = np.std(diffs) if len(diffs) > 0 else np.std(self.history) * 0.15
        else:
            self.residuals_std = np.std(self.history) * 0.15
        return self
        
    def predict(self, horizon=13):
        n = len(self.history)
        preds = []
        for h in range(1, horizon + 1):
            idx = n - self.season_length + ((h - 1) % self.season_length)
            if idx >= 0 and idx < n:
                preds.append(self.history[idx])
            else:
                preds.append(self.history[-1])
        point = np.array(preds)
        
        # Prediction intervals based on seasonal difference variance
        z_80 = 1.282
        z_95 = 1.960
        # Uncertainty grows slightly with sqrt of horizon cycles
        step_factor = np.sqrt(np.arange(1, horizon + 1) / self.season_length + 1.0)
        
        lower_80 = np.maximum(0, point - z_80 * self.residuals_std * step_factor)
        upper_80 = point + z_80 * self.residuals_std * step_factor
        lower_95 = np.maximum(0, point - z_95 * self.residuals_std * step_factor)
        upper_95 = point + z_95 * self.residuals_std * step_factor
        
        return {
            "point": point,
            "lower_80": lower_80,
            "upper_80": upper_80,
            "lower_95": lower_95,
            "upper_95": upper_95
        }


class ETSModel:
    """
    Holt-Winters Exponential Smoothing model with additive trend and additive seasonality (s=52).
    """
    def __init__(self, seasonal_periods=52):
        self.seasonal_periods = seasonal_periods
        self.fitted_model = None
        self.res_std = None
        
    def fit(self, y_train):
        # Clip zeros to positive values for stability
        y = np.maximum(1.0, np.array(y_train, dtype=float))
        try:
            model = ExponentialSmoothing(
                y,
                trend="add",
                seasonal="add",
                seasonal_periods=self.seasonal_periods,
                initialization_method="estimated"
            )
            self.fitted_model = model.fit(optimized=True)
            residuals = y - self.fitted_model.fittedvalues
            self.res_std = np.std(residuals)
        except Exception:
            # Fallback to simple additive trend if 52-period seasonal fails
            model = ExponentialSmoothing(y, trend="add", seasonal=None, initialization_method="estimated")
            self.fitted_model = model.fit()
            self.res_std = np.std(y - self.fitted_model.fittedvalues)
        return self
        
    def predict(self, horizon=13):
        point = self.fitted_model.forecast(horizon)
        point = np.maximum(0, np.array(point))
        
        z_80 = 1.282
        z_95 = 1.960
        step_factor = np.sqrt(np.arange(1, horizon + 1))
        
        lower_80 = np.maximum(0, point - z_80 * self.res_std * (1.0 + 0.05 * step_factor))
        upper_80 = point + z_80 * self.res_std * (1.0 + 0.05 * step_factor)
        lower_95 = np.maximum(0, point - z_95 * self.res_std * (1.0 + 0.05 * step_factor))
        upper_95 = point + z_95 * self.res_std * (1.0 + 0.05 * step_factor)
        
        return {
            "point": point,
            "lower_80": lower_80,
            "upper_80": upper_80,
            "lower_95": lower_95,
            "upper_95": upper_95
        }


class ProphetModel:
    """
    Prophet model customized for Pune healthcare:
    - Piecewise linear trend
    - Annual Fourier seasonality
    - Exogenous festival regressors: Diwali, Ganesh Chaturthi, Holi, and Monsoon Rain Index
    """
    def __init__(self):
        self.model = None
        self.regressors = [
            "monsoon_rain_index",
            "fest_diwali",
            "fest_diwali_post_surge",
            "fest_ganesh_chaturthi",
            "fest_ganesh_post_surge",
            "fest_holi"
        ]
        
    def fit(self, train_df, target_col="y"):
        prophet_df = pd.DataFrame({
            "ds": pd.to_datetime(train_df["week_start"]),
            "y": train_df[target_col].values
        })
        for reg in self.regressors:
            if reg in train_df.columns:
                prophet_df[reg] = train_df[reg].values
            else:
                prophet_df[reg] = 0.0
                
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.08,
            seasonality_prior_scale=10.0,
            interval_width=0.95
        )
        for reg in self.regressors:
            self.model.add_regressor(reg)
            
        self.model.fit(prophet_df)
        return self
        
    def predict(self, future_df):
        future_prophet = pd.DataFrame({
            "ds": pd.to_datetime(future_df["week_start"])
        })
        for reg in self.regressors:
            if reg in future_df.columns:
                future_prophet[reg] = future_df[reg].values
            else:
                future_prophet[reg] = 0.0
                
        forecast = self.model.predict(future_prophet)
        point = np.maximum(0, forecast["yhat"].values)
        lower_95 = np.maximum(0, forecast["yhat_lower"].values)
        upper_95 = np.maximum(0, forecast["yhat_upper"].values)
        
        # Approximate 80% CI from 95% CI (ratio of z-scores: 1.282 / 1.960 = 0.654)
        half_width_95 = (upper_95 - lower_95) / 2.0
        half_width_80 = half_width_95 * (1.282 / 1.960)
        lower_80 = np.maximum(0, point - half_width_80)
        upper_80 = point + half_width_80
        
        return {
            "point": point,
            "lower_80": lower_80,
            "upper_80": upper_80,
            "lower_95": lower_95,
            "upper_95": upper_95
        }


class LightGBMTimeSeriesModel:
    """
    LightGBM model with lag features, rolling statistics, calendar indicators,
    and quantile regressors for empirical 80% and 95% prediction intervals.
    """
    def __init__(self, lags=[1, 2, 3, 4, 13, 52]):
        self.lags = lags
        self.model_point = None
        self.model_q10 = None
        self.model_q90 = None
        self.model_q025 = None
        self.model_q975 = None
        self.feature_cols = None
        self.train_data_full = None
        self.target_col = None
        
    def _create_features(self, df, target_col):
        data = df.copy().reset_index(drop=True)
        # Autoregressive lags
        for l in self.lags:
            data[f"lag_{l}"] = data[target_col].shift(l)
            
        # Rolling stats
        data["rolling_mean_4"] = data[target_col].shift(1).rolling(4, min_periods=1).mean()
        data["rolling_std_4"] = data[target_col].shift(1).rolling(4, min_periods=1).std().fillna(0)
        data["rolling_mean_13"] = data[target_col].shift(1).rolling(13, min_periods=1).mean()
        data["rolling_min_4"] = data[target_col].shift(1).rolling(4, min_periods=1).min()
        data["rolling_max_4"] = data[target_col].shift(1).rolling(4, min_periods=1).max()
        
        # Calendar & Pune Event features
        dt = pd.to_datetime(data["week_start"])
        data["week_of_year"] = dt.dt.isocalendar().week.astype(int)
        data["month"] = dt.dt.month
        data["quarter"] = dt.dt.quarter
        data["sin_woy"] = np.sin(2 * np.pi * data["week_of_year"] / 52.0)
        data["cos_woy"] = np.cos(2 * np.pi * data["week_of_year"] / 52.0)
        
        event_cols = ["monsoon_rain_index", "fest_diwali", "fest_diwali_post_surge",
                      "fest_ganesh_chaturthi", "fest_ganesh_post_surge", "fest_holi"]
        for ec in event_cols:
            if ec not in data.columns:
                data[ec] = 0.0
                
        feature_cols = [f"lag_{l}" for l in self.lags] + [
            "rolling_mean_4", "rolling_std_4", "rolling_mean_13", "rolling_min_4", "rolling_max_4",
            "week_of_year", "month", "quarter", "sin_woy", "cos_woy"
        ] + event_cols
        
        return data, feature_cols
        
    def fit(self, train_df, target_col="y"):
        self.target_col = target_col
        self.train_data_full = train_df.copy().reset_index(drop=True)
        
        featured_df, feature_cols = self._create_features(self.train_data_full, target_col)
        self.feature_cols = feature_cols
        
        # Drop rows where longest lag is NaN
        max_lag = max(self.lags)
        train_subset = featured_df.iloc[max_lag:].dropna(subset=self.feature_cols + [target_col])
        
        X = train_subset[self.feature_cols]
        y = train_subset[target_col]
        
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "n_estimators": 120,
            "learning_rate": 0.06,
            "num_leaves": 15,
            "min_child_samples": 5,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "random_state": 42,
            "verbose": -1
        }
        
        self.model_point = lgb.LGBMRegressor(**params)
        self.model_point.fit(X, y)
        
        # Quantile models for 80% and 95% intervals
        q_params = params.copy()
        q_params["objective"] = "quantile"
        q_params.pop("metric", None)
        
        self.model_q10 = lgb.LGBMRegressor(**q_params, alpha=0.10).fit(X, y)
        self.model_q90 = lgb.LGBMRegressor(**q_params, alpha=0.90).fit(X, y)
        self.model_q025 = lgb.LGBMRegressor(**q_params, alpha=0.025).fit(X, y)
        self.model_q975 = lgb.LGBMRegressor(**q_params, alpha=0.975).fit(X, y)
        
        return self
        
    def predict(self, future_df, horizon=13):
        """
        Performs multi-step recursive forecasting updating lags and rolling features dynamically.
        """
        extended = pd.concat([self.train_data_full, future_df], ignore_index=True)
        train_len = len(self.train_data_full)
        
        preds_point = []
        preds_q10 = []
        preds_q90 = []
        preds_q025 = []
        preds_q975 = []
        
        for h in range(horizon):
            curr_idx = train_len + h
            feat_df, _ = self._create_features(extended, self.target_col)
            row_x = feat_df.loc[[curr_idx], self.feature_cols]
            
            p_val = max(0.0, float(self.model_point.predict(row_x)[0]))
            q10_val = max(0.0, float(self.model_q10.predict(row_x)[0]))
            q90_val = max(p_val, float(self.model_q90.predict(row_x)[0]))
            q025_val = max(0.0, float(self.model_q025.predict(row_x)[0]))
            q975_val = max(q90_val, float(self.model_q975.predict(row_x)[0]))
            
            # Enforce order: q025 <= q10 <= point <= q90 <= q975
            q10_val = min(q10_val, p_val)
            q025_val = min(q025_val, q10_val)
            
            preds_point.append(p_val)
            preds_q10.append(q10_val)
            preds_q90.append(q90_val)
            preds_q025.append(q025_val)
            preds_q975.append(q975_val)
            
            # Feedback prediction recursively into history
            extended.at[curr_idx, self.target_col] = p_val
            
        return {
            "point": np.array(preds_point),
            "lower_80": np.array(preds_q10),
            "upper_80": np.array(preds_q90),
            "lower_95": np.array(preds_q025),
            "upper_95": np.array(preds_q975)
        }
