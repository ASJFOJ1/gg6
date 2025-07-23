import numpy as np
import pandas as pd
import requests
from datetime import datetime, timezone, timedelta
import json
import math
from scipy import signal
from scipy.signal import hilbert
from scipy.stats import entropy, zscore
from scipy.linalg import svd, norm
from scipy.sparse.linalg import svds
from statsmodels.tsa.stattools import grangercausalitytests, adfuller, coint
from statsmodels.tsa.api import VAR, VECM
from arch import arch_model
from hmmlearn import hmm
import pywt
from pykalman import KalmanFilter
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import talib
from typing import Tuple, Optional, Dict, List, Any
import time
import os
import traceback
from multiprocess import Pool, cpu_count
import optuna
from functools import partial
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.stats import gaussian_kde
from sklearn.feature_selection import mutual_info_regression
import networkx as nx
from itertools import product
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import pdist, squareform
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import threading
from collections import defaultdict, Counter
import pickle
import sys


warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Configure logging properly
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('market_analysis.log'),
        logging.StreamHandler()
    ]
)

class RateLimiter:
    """Centralized rate limiter for API calls"""
    def __init__(self, calls_per_second: float = 0.5):
        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0
        self.lock = threading.Lock()

    def wait(self):
        """Wait if necessary to maintain rate limit"""
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_call

            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                time.sleep(sleep_time)

            self.last_call = time.time()


class PureNumpyTensorDecomposition:
    """Pure NumPy implementation of tensor decomposition methods"""

    def __init__(self, rank: int = 10, max_iter: int = 100, tol: float = 1e-6):
        self.rank = rank
        self.max_iter = max_iter
        self.tol = tol
        self.logger = logging.getLogger(__name__)

    def cp_als(self, tensor: np.ndarray) -> Dict[str, Any]:
        """
        Canonical Polyadic (CP) decomposition using Alternating Least Squares
        Pure NumPy implementation - no external dependencies
        """
        shape = tensor.shape
        ndims = len(shape)

        # Initialize factor matrices randomly
        factors = []
        for i in range(ndims):
            factor = np.random.randn(shape[i], self.rank)
            factor = factor / np.linalg.norm(factor, axis=0)
            factors.append(factor)

        # Store convergence history
        errors = []

        for iteration in range(self.max_iter):
            # Update each factor matrix
            for mode in range(ndims):
                # Unfold tensor along current mode
                unfolded = self._unfold_tensor(tensor, mode)

                # Compute Khatri-Rao product of all factors except current
                kr_product = self._khatri_rao_product(factors, skip_index=mode)

                # Update factor using least squares
                try:
                    factors[mode] = unfolded @ kr_product @ np.linalg.pinv(
                        kr_product.T @ kr_product
                    )
                except:
                    # Fallback for numerical issues
                    factors[mode] = unfolded @ kr_product @ np.linalg.inv(
                        kr_product.T @ kr_product + 1e-6 * np.eye(self.rank)
                    )

                # Normalize columns
                norms = np.linalg.norm(factors[mode], axis=0)
                factors[mode] = factors[mode] / (norms + 1e-10)

            # Calculate reconstruction error
            reconstructed = self._reconstruct_tensor(factors)
            error = np.linalg.norm(tensor - reconstructed) / (np.linalg.norm(tensor) + 1e-10)
            errors.append(error)

            # Check convergence
            if iteration > 0 and abs(errors[-1] - errors[-2]) < self.tol:
                break

        return {
            'factors': factors,
            'weights': np.ones(self.rank),
            'reconstruction_error': errors[-1] if errors else 1.0,
            'iterations': iteration + 1,
            'convergence_history': errors
        }

    def _unfold_tensor(self, tensor: np.ndarray, mode: int) -> np.ndarray:
        """Unfold tensor along specified mode"""
        return np.moveaxis(tensor, mode, 0).reshape((tensor.shape[mode], -1))

    def _khatri_rao_product(self, matrices: List[np.ndarray], skip_index: int = None) -> np.ndarray:
        """Compute Khatri-Rao product of matrices"""
        if skip_index is not None:
            matrices = [m for i, m in enumerate(matrices) if i != skip_index]

        if not matrices:
            return np.array([[1.0]])

        # Start with the last matrix and reverse the list for correct kr product order
        matrices_rev = matrices[::-1]
        result = matrices_rev[0]
        for matrix in matrices_rev[1:]:
            # Khatri-Rao product using broadcasting for efficiency
            result = np.einsum('ij,kj->ikj', matrix, result).reshape(-1, result.shape[1])

        return result

    def _reconstruct_tensor(self, factors: List[np.ndarray]) -> np.ndarray:
        """
        Reconstruct tensor from CP factors using memory-efficient method.
        """
        rank = factors[0].shape[1]
        shape = [f.shape[0] for f in factors]

        # Use einsum for memory efficiency
        reconstructed = np.zeros(shape)

        for r in range(rank):
            # Extract factor columns
            factor_cols = [f[:, r] for f in factors]

            # Use einsum for efficient outer product
            if len(factors) == 3:
                reconstructed += np.einsum('i,j,k->ijk', *factor_cols)
            elif len(factors) == 4:
                reconstructed += np.einsum('i,j,k,l->ijkl', *factor_cols)
            else:
                # Fallback for other dimensions
                rank_1 = factor_cols[0]
                for fc in factor_cols[1:]:
                    rank_1 = np.multiply.outer(rank_1, fc)
                reconstructed += rank_1

        return reconstructed


class SimplifiedTensorAnalysis:
    """Simplified tensor analysis for the main system"""

    def __init__(self, lookback_size: int = 200):
        self.lookback_size = lookback_size
        self.decomposer = PureNumpyTensorDecomposition(rank=10)
        self.tensor_dimensions = {
            'time': lookback_size,
            'features': 10,
            'timeframes': 7  # Updated to include all timeframes
        }
        self.logger = logging.getLogger(__name__)

    def create_simplified_tensor(self, data: Dict[str, pd.DataFrame]) -> np.ndarray:
        """Create simplified 3D tensor for easier computation"""
        n_features = self.tensor_dimensions['features']
        timeframes = ['15m', '30m', '1h', '4h', '6h', '12h', '1d']
        
        # FIX: Normalize to minimum samples across timeframes
        # First pass to find minimum samples
        valid_tfs = []
        for tf in timeframes:
            if tf in data and not data[tf].empty:
                # Check if it's kline data or processed data
                if 'close' in data[tf].columns:
                    valid_tfs.append(tf)
                elif isinstance(data[tf], pd.DataFrame) and len(data[tf].columns) > 0:
                    # It's already processed data
                    valid_tfs.append(tf)
        
        if not valid_tfs:
            return np.zeros((0, n_features, len(timeframes)))
        
        # Find minimum samples across valid timeframes
        min_samples = min(len(data[tf]) for tf in valid_tfs)
        min_samples = min(min_samples, self.lookback_size)
        
        tensor_shape = (min_samples, n_features, len(timeframes))
        try:
            # Check memory before allocation
            self.decomposer.logger.info(f"Allocating tensor of shape {tensor_shape}")
            bytes_needed = np.prod(tensor_shape) * np.dtype(np.float64).itemsize
            if bytes_needed > 1e9:  # 1GB limit
                raise MemoryError(f"Tensor allocation of {bytes_needed/1e9:.2f}GB exceeds limit")
            
            tensor = np.zeros(tensor_shape)
        except MemoryError:
            # Use sparse representation or chunking
            self.decomposer.logger.warning("Using reduced tensor size due to memory constraints")
            min_samples = min(min_samples, 100)  # Reduce to 100 samples max
            tensor = np.zeros((min_samples, n_features, len(timeframes)))

        for tf_idx, timeframe in enumerate(timeframes):
            if timeframe not in data or data[timeframe].empty:
                continue

            df = data[timeframe].iloc[-min_samples:]
            n_samples = len(df)

            # Extract key features
            features = np.zeros((n_samples, n_features))

            # Check what type of data we have
            if 'close' in df.columns:
                # It's kline data
                features[:, 0] = df['close'].values
                features[:, 1] = df.get('high', df['close']).values
                features[:, 2] = df.get('low', df['close']).values
                features[:, 3] = df.get('volume', 0).values

                # Technical features
                if n_samples > 20:
                    features[:, 4] = self._calculate_rsi(df['close'])
                    features[:, 5] = self._calculate_momentum(df['close'])
                    features[:, 6] = self._calculate_volatility(df['close'])

                    # Additional features
                    if 'high' in df.columns and 'low' in df.columns:
                        features[:, 7] = (df['high'] - df['low']).values  # Range
                    if 'open' in df.columns:
                        features[:, 8] = (df['close'] - df['open']).values  # Change
                    if 'volume' in df.columns:
                        features[:, 9] = df['volume'].rolling(10).mean().fillna(0).values  # Avg volume
            else:
                # It's processed data (from _prepare_aligned_data)
                # Use the available columns
                available_cols = df.columns.tolist()
                for i, col in enumerate(available_cols[:n_features]):
                    features[:, i] = df[col].values

            # Normalize features
            for i in range(n_features):
                if np.std(features[:, i]) > 0:
                    features[:, i] = (features[:, i] - np.mean(features[:, i])) / np.std(features[:, i])

            # Fill tensor
            tensor[:, :, tf_idx] = features

        return tensor

    def analyze_tensor(self, tensor: np.ndarray) -> Dict[str, Any]:
        """Perform tensor analysis using pure NumPy methods"""
        # Use CP decomposition
        cp_result = self.decomposer.cp_als(tensor)

        # Extract temporal patterns
        temporal_factors = cp_result['factors'][0]  # Time dimension

        # Calculate predictions
        predictions = {}
        for i in range(min(temporal_factors.shape[1], 5)):  # Limit to first 5 factors
            if len(temporal_factors[:, i]) > 20:
                # Simple linear trend
                x = np.arange(20)
                y = temporal_factors[-20:, i]
                try:
                    trend = np.polyfit(x, y, 1)[0]
                    predictions[f'factor_{i}_trend'] = trend
                except:
                    predictions[f'factor_{i}_trend'] = 0.0

        # Energy Conservation Check
        reconstructed_energy = np.sum(self.decomposer._reconstruct_tensor(cp_result['factors']) ** 2)
        original_energy = np.sum(tensor ** 2)
        energy_conservation_ratio = reconstructed_energy / original_energy if original_energy > 0 else 1.0

        return {
            'decomposition': cp_result,
            'temporal_patterns': temporal_factors,
            'predictions': predictions,
            'reconstruction_error': cp_result['reconstruction_error'],
            'energy_conservation_ratio': energy_conservation_ratio
        }

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> np.ndarray:
        """Calculate RSI"""
        prices_array = prices.values if isinstance(prices, pd.Series) else prices
        rsi = np.zeros_like(prices_array)

        if len(prices_array) < period + 1:
            return rsi

        deltas = np.diff(prices_array)
        seed = deltas[:period]
        up = np.mean(seed[seed > 0]) if len(seed[seed > 0]) > 0 else 0
        down = -np.mean(seed[seed < 0]) if len(seed[seed < 0]) > 0 else 0

        if down == 0:
            rsi[period] = 100
        else:
            rs = up / down
            rsi[period] = 100 - 100 / (1 + rs)

        # Fill first period with 50
        rsi[:period] = 50

        # Calculate remaining RSI values
        for i in range(period + 1, len(prices_array)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0
            else:
                upval = 0
                downval = -delta

            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period

            if down == 0:
                rsi[i] = 100
            else:
                rs = up / down
                rsi[i] = 100 - 100 / (1 + rs)

        return rsi

    def _calculate_momentum(self, prices: pd.Series, period: int = 10) -> np.ndarray:
        """Calculate momentum"""
        prices_array = prices.values if isinstance(prices, pd.Series) else prices
        momentum = np.zeros_like(prices_array)

        if len(prices_array) > period:
            momentum[period:] = prices_array[period:] - prices_array[:-period]

        return momentum

    def _calculate_volatility(self, prices: pd.Series, period: int = 20) -> np.ndarray:
        """Calculate rolling volatility"""
        prices_array = prices.values if isinstance(prices, pd.Series) else prices
        volatility = np.zeros_like(prices_array)

        if len(prices_array) < 2:
            return volatility

        returns = np.diff(np.log(prices_array + 1e-10))

        for i in range(period, len(prices_array)):
            if i - period < len(returns):
                volatility[i] = np.std(returns[max(0, i-period):i])

        return volatility


class HawkesProcess:
    """
    Hawkes Process implementation for modeling self-exciting events like liquidation cascades
    """

    def __init__(self, baseline_intensity: float = 0.1, alpha: float = 0.5, beta: float = 1.0):
        self.mu = baseline_intensity  # Baseline intensity
        self.alpha = alpha  # Jump size on each event
        self.beta = beta   # Decay rate
        self.logger = logging.getLogger(__name__)

    def fit(self, event_times: np.ndarray):
        """
        Fit Hawkes process parameters using simplified estimation
        """
        if len(event_times) < 2:
            self.logger.warning("Not enough events to fit Hawkes process")
            return self
        
        # Basic parameter estimation
        T = event_times[-1] - event_times[0]
        N = len(event_times)
        
        if T <= 0:
            self.logger.warning("Invalid time range for Hawkes process")
            return self
        
        # Estimate baseline intensity
        self.mu = N / T * 0.5  # Conservative estimate
        
        # Estimate alpha and beta using inter-event times
        inter_event_times = np.diff(event_times)
        
        # Simple heuristic: if events cluster, increase alpha
        median_iet = np.median(inter_event_times)
        mean_iet = np.mean(inter_event_times)
        
        if mean_iet > 0 and median_iet > 0:
            # If mean >> median, we have clustering
            clustering_factor = mean_iet / median_iet
            self.alpha = min(0.9, 0.3 * clustering_factor)  # Cap at 0.9 for stability
            
            # Beta controls decay rate - higher beta = faster decay
            self.beta = 1.0 / median_iet
        
        self.logger.info(f"Hawkes process fitted: mu={self.mu:.4f}, alpha={self.alpha:.4f}, beta={self.beta:.4f}")
        return self

    def branching_ratio(self) -> float:
        """
        Calculate branching ratio (α/β)
        Values < 1 indicate stable process, > 1 indicate explosive cascades
        """
        if self.beta == 0:
            return float('inf')
        return self.alpha / self.beta

def branching_ratio(self) -> float:
    """
    Calculate branching ratio (α/β)
    Values < 1 indicate stable process, > 1 indicate explosive cascades
    """
    if self.beta == 0:
        return float('inf')
    return self.alpha / self.beta
    def _calculate_intensities(self, event_times: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        """
        Calculate intensity at each event time or at time t
        """
        if len(event_times) == 0:
            return self.mu if t is not None else np.array([self.mu])
            
        if t is None:
            # Calculate at each event time
            intensities = np.zeros(len(event_times))
            for i, t_i in enumerate(event_times):
                intensity = self.mu
                # Only sum over previous events
                for j in range(i):
                    if t_i > event_times[j]:  # Safety check
                        intensity += self.alpha * np.exp(-self.beta * (t_i - event_times[j]))
                intensities[i] = intensity
            return intensities
        else:
            # Calculate at specific time t
            intensity = self.mu
            for event_time in event_times[event_times < t]:
                time_diff = t - event_time
                if time_diff > 0:  # Safety check
                    # Prevent overflow in exponential
                    if self.beta * time_diff < 100:  # exp(-100) is effectively 0
                        intensity += self.alpha * np.exp(-self.beta * time_diff)
            return intensity

    def predict_next_event_probability(self, event_times: np.ndarray,
                                     time_horizon: float = 1.0,
                                     n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict probability of next event within time horizon
        """
        event_times = np.sort(event_times)
        last_event = event_times[-1]

        time_points = np.linspace(last_event, last_event + time_horizon, n_points)
        probabilities = np.zeros(n_points)

        for i, t in enumerate(time_points):
            intensity = self._calculate_intensities(event_times, t)
            # Probability of event in small interval dt
            dt = time_horizon / n_points
            probabilities[i] = 1 - np.exp(-intensity * dt)

        return time_points, probabilities

    def simulate(self, T: float, n_simulations: int = 1) -> List[np.ndarray]:
        """
        Simulate Hawkes process paths
        """
        simulations = []

        for _ in range(n_simulations):
            events = []
            t = 0
            max_iterations = 10000  # Prevent infinite loops
            iterations = 0

            while t < T and iterations < max_iterations:
                iterations += 1
                
                # Upper bound for intensity
                intensity_upper = self.mu + self.alpha * len(events)
                
                # Prevent extremely small or zero intensity
                if intensity_upper < 1e-10:
                    intensity_upper = 1e-10

                # Generate candidate event time
                tau = np.random.exponential(1 / intensity_upper)
                t += tau

                if t >= T:
                    break

                # Accept/reject based on true intensity
                true_intensity = self._calculate_intensities(np.array(events), t)
                
                # Safety check to prevent division by zero
                if intensity_upper <= 0:
                    continue
                    
                acceptance_prob = true_intensity / intensity_upper
                
                # Ensure acceptance probability is valid
                acceptance_prob = np.clip(acceptance_prob, 0, 1)
                
                if np.random.rand() < acceptance_prob:
                    events.append(t)

            simulations.append(np.array(events))

        return simulations


class UnscentedKalmanFilter:
    """
    Unscented Kalman Filter for non-linear state estimation
    Better than standard Kalman for capturing market non-linearities
    """

    def __init__(self, dim_x: int, dim_z: int, alpha: float = 0.001,
                 beta: float = 2.0, kappa: float = None):
        self.dim_x = dim_x  # State dimension
        self.dim_z = dim_z  # Measurement dimension
        self.logger = logging.getLogger(__name__)

        # UKF parameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa if kappa is not None else 3 - dim_x

        # Sigma point parameters
        self.lambda_ = self.alpha**2 * (dim_x + self.kappa) - dim_x
        self.n_sigma = 2 * dim_x + 1

        # Weights for sigma points
        self.Wm = np.zeros(self.n_sigma)  # Weights for means
        self.Wc = np.zeros(self.n_sigma)  # Weights for covariance

        self.Wm[0] = self.lambda_ / (dim_x + self.lambda_)
        self.Wc[0] = self.lambda_ / (dim_x + self.lambda_) + (1 - self.alpha**2 + self.beta)

        for i in range(1, self.n_sigma):
            self.Wm[i] = 1 / (2 * (dim_x + self.lambda_))
            self.Wc[i] = 1 / (2 * (dim_x + self.lambda_))

        # Initialize state and covariance
        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x)
        self.Q = np.eye(dim_x) * 0.01  # Process noise
        self.R = np.eye(dim_z) * 0.1   # Measurement noise

    def generate_sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """
        Generate sigma points for UKF with improved numerical stability
        """
        n = len(x)
        sigma_points = np.zeros((self.n_sigma, n))
        
        # Ensure P is symmetric
        P = (P + P.T) / 2
        
        # Improved regularization based on matrix condition
        min_eig = np.min(np.linalg.eigvalsh(P))
        if min_eig < 1e-6:
            eps = 1e-3
        else:
            eps = 1e-6
        
        P_reg = P + eps * np.eye(n)
        
        # Use SVD directly for better stability
        try:
            U, s, Vt = np.linalg.svd(P_reg)
            # Ensure all eigenvalues are positive
            s = np.maximum(s, eps)
            # Reconstruct with conditioned eigenvalues
            S = U @ np.diag(np.sqrt(s * (n + self.lambda_)))
        except np.linalg.LinAlgError:
            # Final fallback: Identity scaled by trace
            self.logger.warning("SVD failed, using identity fallback")
            trace_P = np.trace(P)
            S = np.sqrt((n + self.lambda_) * trace_P / n) * np.eye(n)
        
        # Generate sigma points
        sigma_points[0] = x
        for i in range(n):
            sigma_points[i + 1] = x + S[:, i]
            sigma_points[n + i + 1] = x - S[:, i]
        
        return sigma_points

    def state_transition(self, x: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """
        Non-linear state transition function
        Can be customized for specific market dynamics
        """
        # Example: Simple trend + mean reversion model
        x_new = x.copy()

        # Price evolution with momentum and mean reversion
        if self.dim_x >= 2:
            x_new[0] = x[0] + x[1] * dt  # Price = price + velocity
            x_new[1] = x[1] * 0.98  # Velocity decays (mean reversion)

        return x_new

    def measurement_function(self, x: np.ndarray) -> np.ndarray:
        """
        Non-linear measurement function
        Maps state to observations
        """
        # Simple case: observe price directly
        z = np.zeros(self.dim_z)
        z[0] = x[0]  # Observe price

        return z

    def predict(self, dt: float = 1.0):
        """
        Prediction step of UKF
        """
        # Generate sigma points
        sigma_points = self.generate_sigma_points(self.x, self.P)

        # Transform sigma points through state transition
        sigma_points_pred = np.zeros_like(sigma_points)
        for i in range(self.n_sigma):
            sigma_points_pred[i] = self.state_transition(sigma_points[i], dt)

        # Calculate predicted mean and covariance
        self.x = np.sum(self.Wm[:, np.newaxis] * sigma_points_pred, axis=0)

        self.P = self.Q.copy()
        for i in range(self.n_sigma):
            y = sigma_points_pred[i] - self.x
            self.P += self.Wc[i] * np.outer(y, y)

        # Ensure P remains symmetric and positive semi-definite
        self.P = (self.P + self.P.T) / 2
        # Add small regularization
        self.P += 1e-6 * np.eye(self.dim_x)

    def update(self, z: np.ndarray):
        """
        Update step of UKF
        """
        # Generate sigma points
        sigma_points = self.generate_sigma_points(self.x, self.P)

        # Transform sigma points through measurement function
        z_sigma = np.zeros((self.n_sigma, self.dim_z))
        for i in range(self.n_sigma):
            z_sigma[i] = self.measurement_function(sigma_points[i])

        # Calculate predicted measurement
        z_pred = np.sum(self.Wm[:, np.newaxis] * z_sigma, axis=0)

        # Calculate innovation covariance
        Pz = self.R.copy()
        for i in range(self.n_sigma):
            y = z_sigma[i] - z_pred
            Pz += self.Wc[i] * np.outer(y, y)

        # Ensure Pz is positive definite
        Pz = (Pz + Pz.T) / 2
        Pz += 1e-6 * np.eye(self.dim_z)

        # Calculate cross covariance
        Pxz = np.zeros((self.dim_x, self.dim_z))
        for i in range(self.n_sigma):
            Pxz += self.Wc[i] * np.outer(sigma_points[i] - self.x, z_sigma[i] - z_pred)

        # Kalman gain
        try:
            K = Pxz @ np.linalg.inv(Pz)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            K = Pxz @ np.linalg.pinv(Pz)

        # Update state and covariance
        self.x = self.x + K @ (z - z_pred)
        self.P = self.P - K @ Pz @ K.T

        # Ensure P remains symmetric and positive semi-definite
        self.P = (self.P + self.P.T) / 2
        # Force positive eigenvalues
        eigvals, eigvecs = np.linalg.eigh(self.P)
        eigvals = np.maximum(eigvals, 1e-6)
        self.P = eigvecs @ np.diag(eigvals) @ eigvecs.T

    def filter_series(self, observations: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Filter a complete time series
        """
        n = len(observations)
        filtered_states = np.zeros((n, self.dim_x))
        innovations = np.zeros(n)

        for i in range(n):
            self.predict()

            # Innovation before update
            z_pred = self.measurement_function(self.x)
            innovations[i] = observations[i] - z_pred[0]

            self.update(np.array([observations[i]]))
            filtered_states[i] = self.x.copy()

        return {
            'filtered_states': filtered_states,
            'innovations': innovations,
            'final_covariance': self.P
        }


class MarkovSwitchingGARCH:
    """
    Markov-Switching GARCH model combining HMM and GARCH
    """

    def __init__(self, n_states: int = 3, garch_p: int = 1, garch_q: int = 1):
        self.n_states = n_states
        self.garch_p = garch_p
        self.garch_q = garch_q
        self.hmm_model = None
        self.garch_models = {}
        self.state_params = {}
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)

    def fit(self, returns: np.ndarray):
        """
        Fit MS-GARCH model with improved convergence robustness.
        """
        # Step 1: Fit HMM to identify regimes
        self.hmm_model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="diag",
            n_iter=200,  # Increased iterations
            random_state=42,
            tol=1e-3,  # Adjusted tolerance
            init_params="cm",  # Initialize with kmeans and means
            params="cmt"  # Re-estimate covars, means, transmat
        )

        # Scale returns for HMM stability
        X = self.scaler.fit_transform(returns.reshape(-1, 1))

        try:
            self.hmm_model.fit(X)
            states = self.hmm_model.predict(X)
            self.logger.info(f"HMM converged successfully with {self.n_states} states")
        except Exception as e:
            # Fallback if HMM fails to converge
            self.logger.warning(f"HMM fitting failed: {e}. Using fallback k-means clustering.")
            # Use k-means clustering as fallback
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.n_states, random_state=42, n_init=10)
            states = kmeans.fit_predict(X)

        # Step 2: Fit separate GARCH model for each state
        for state in range(self.n_states):
            state_returns = returns[states == state]

            if len(state_returns) > 50:  # Need enough data
                # Fit GARCH model
                model = arch_model(
                    state_returns * 100,  # Scale for numerical stability
                    vol='GARCH',
                    p=self.garch_p,
                    q=self.garch_q,
                    rescale=False
                )

                try:
                    res = model.fit(disp='off', show_warning=False)
                    self.garch_models[state] = res

                    # Store parameters
                    self.state_params[state] = {
                        'omega': res.params['omega'],
                        'alpha': res.params.get('alpha[1]', 0),
                        'beta': res.params.get('beta[1]', 0),
                        'mean': np.mean(state_returns),
                        'unconditional_vol': np.std(state_returns)
                    }
                    self.logger.info(f"GARCH fitted for state {state}")
                except Exception as e:
                    # Fallback to simple volatility
                    self.logger.warning(f"GARCH fitting failed for state {state}: {e}")
                    self.state_params[state] = {
                        'omega': np.var(state_returns),
                        'alpha': 0.1,
                        'beta': 0.8,
                        'mean': np.mean(state_returns),
                        'unconditional_vol': np.std(state_returns)
                    }
            else:
                # Not enough data for this state
                self.state_params[state] = {
                    'omega': np.var(returns) / self.n_states,
                    'alpha': 0.1,
                    'beta': 0.8,
                    'mean': np.mean(returns),
                    'unconditional_vol': np.std(returns)
                }

        return self

    def predict_volatility(self, returns: np.ndarray, horizon: int = 1) -> Dict[str, Any]:
        """
        Predict volatility considering regime switches
        """
        # Get current state probabilities
        X = self.scaler.transform(returns.reshape(-1, 1))
        try:
            state_probs = self.hmm_model.predict_proba(X)
            current_state_probs = state_probs[-1]
            trans_mat = self.hmm_model.transmat_
        except:
            # Fallback
            current_state_probs = np.ones(self.n_states) / self.n_states
            trans_mat = np.ones((self.n_states, self.n_states)) / self.n_states

        # Initialize predictions
        vol_predictions = np.zeros((horizon, self.n_states))
        weighted_vol = np.zeros(horizon)

        # Current volatilities for each state
        current_vols = np.zeros(self.n_states)
        for state in range(self.n_states):
            params = self.state_params[state]
            # Simple GARCH(1,1) update
            if len(returns) > 1:
                current_vols[state] = np.sqrt(
                    params['omega'] +
                    params['alpha'] * returns[-1]**2 +
                    params['beta'] * params['unconditional_vol']**2
                )
            else:
                current_vols[state] = params['unconditional_vol']

        # Predict forward
        state_probs_forward = current_state_probs.copy()

        for h in range(horizon):
            # Volatility for each state
            for state in range(self.n_states):
                params = self.state_params[state]
                if h == 0:
                    vol_predictions[h, state] = current_vols[state]
                else:
                    # GARCH recursion
                    vol_predictions[h, state] = np.sqrt(
                        params['omega'] +
                        (params['alpha'] + params['beta']) * vol_predictions[h-1, state]**2
                    )

            # Weighted average volatility
            weighted_vol[h] = np.sum(state_probs_forward * vol_predictions[h])

            # Update state probabilities
            state_probs_forward = state_probs_forward @ trans_mat

        return {
            'volatility_forecast': weighted_vol,
            'state_volatilities': vol_predictions,
            'current_regime': np.argmax(current_state_probs),
            'regime_probabilities': dict(enumerate(current_state_probs)),
            'state_parameters': self.state_params,
            'transition_matrix': trans_mat
        }


class ConditionalMutualInformation:
    """
    Calculate Conditional Mutual Information for three-way relationships
    """

    @staticmethod
    def calculate_cmi(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                      bins: int = 10) -> float:
        """
        Calculate I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)
        """
        # Ensure arrays
        X = np.array(X).flatten() if hasattr(X, 'values') else X.flatten()
        Y = np.array(Y).flatten() if hasattr(Y, 'values') else Y.flatten()
        Z = np.array(Z).flatten() if hasattr(Z, 'values') else Z.flatten()

        # Discretize continuous variables
        X_disc = pd.qcut(X, q=bins, labels=False, duplicates='drop')
        Y_disc = pd.qcut(Y, q=bins, labels=False, duplicates='drop')
        Z_disc = pd.qcut(Z, q=bins, labels=False, duplicates='drop')

        # Calculate joint entropies
        def joint_entropy(data):
            """Calculate joint entropy of multiple variables"""
            # Create joint distribution
            if data.shape[1] == 1:
                counts = np.bincount(data.flatten())
            else:
                # Convert to tuples for counting
                tuples = [tuple(row) for row in data]
                unique, counts = np.unique(tuples, return_counts=True, axis=0)

            # Calculate probabilities
            probs = counts / np.sum(counts)
            probs = probs[probs > 0]  # Remove zeros

            # Calculate entropy
            return -np.sum(probs * np.log2(probs))

        # Calculate individual entropies
        H_XZ = joint_entropy(np.column_stack([X_disc, Z_disc]))
        H_YZ = joint_entropy(np.column_stack([Y_disc, Z_disc]))
        H_XYZ = joint_entropy(np.column_stack([X_disc, Y_disc, Z_disc]))
        H_Z = joint_entropy(Z_disc.reshape(-1, 1))

        # Calculate CMI
        cmi = H_XZ + H_YZ - H_XYZ - H_Z

        return max(0, cmi)  # Ensure non-negative

    @staticmethod
    def calculate_cmi_matrix(data: Dict[str, np.ndarray],
                           condition_on: str = None) -> pd.DataFrame:
        """
        Calculate CMI matrix for all variable pairs given a conditioning variable
        """
        variables = list(data.keys())
        n_vars = len(variables)

        if condition_on is None:
            # Use the first variable as conditioning by default
            condition_on = variables[0]

        if condition_on not in variables:
            raise ValueError(f"Conditioning variable {condition_on} not found")

        # Remove conditioning variable from list
        test_vars = [v for v in variables if v != condition_on]
        n_test = len(test_vars)

        # Initialize CMI matrix
        cmi_matrix = np.zeros((n_test, n_test))
        Z = data[condition_on]

        for i, var1 in enumerate(test_vars):
            for j, var2 in enumerate(test_vars):
                if i != j:
                    X = data[var1]
                    Y = data[var2]

                    # Ensure same length
                    min_len = min(len(X), len(Y), len(Z))
                    cmi = ConditionalMutualInformation.calculate_cmi(
                        X[:min_len], Y[:min_len], Z[:min_len]
                    )
                    cmi_matrix[i, j] = cmi

        return pd.DataFrame(cmi_matrix, index=test_vars, columns=test_vars)


class MarketPredictionCNN(nn.Module):
    """
    CNN for market prediction using multi-timeframe data
    """

    def __init__(self, n_features: int = 10, n_timeframes: int = 7,
                 sequence_length: int = 50, n_classes: int = 3):
        super(MarketPredictionCNN, self).__init__()

        # First convolution layer - extract local patterns
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=(3, n_features),  # Full feature width
            padding=(1, 0)
        )
        self.bn1 = nn.BatchNorm2d(32)

        # Second convolution layer - combine patterns
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(5, 1),
            padding=(2, 0)
        )
        self.bn2 = nn.BatchNorm2d(64)

        # Third convolution layer - higher level patterns
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 1),
            padding=(1, 0)
        )
        self.bn3 = nn.BatchNorm2d(128)

        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, n_timeframes))

        # Fully connected layers
        self.fc1 = nn.Linear(128 * n_timeframes, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, n_classes)

        # Activation
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Forward pass
        x shape: (batch, 1, sequence_length, n_features * n_timeframes)
        """
        # Convolutional layers
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = self.relu(self.bn3(self.conv3(x)))

        # Global pooling
        x = self.adaptive_pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)

        x = self.relu(self.fc2(x))
        x = self.dropout2(x)

        x = self.fc3(x)

        return x

    def predict_proba(self, x):
        """Get probability predictions"""
        with torch.no_grad():
            logits = self.forward(x)
            return self.softmax(logits)


class CNNMarketPredictor:
    """
    Wrapper class for CNN-based market prediction with Optuna optimization
    """

    def __init__(self, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.logger = logging.getLogger(__name__)
        self.best_params = None
        self.prediction_horizon = 5 # Default, can be updated

    def optimize_hyperparameters(self, X_train: torch.Tensor, y_train: torch.Tensor,
                                X_val: torch.Tensor, y_val: torch.Tensor,
                                n_trials: int = 50) -> Dict[str, Any]:
        """
        Use Optuna to optimize hyperparameters
        """
        def objective(trial):
            # Suggest hyperparameters
            lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
            batch_size = trial.suggest_int('batch_size', 16, 128)
            dropout1 = trial.suggest_float('dropout1', 0.2, 0.7)
            dropout2 = trial.suggest_float('dropout2', 0.1, 0.5)
            weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

            # Create model with suggested parameters
            model = MarketPredictionCNN(
                n_features=X_train.shape[3] // 7,
                n_timeframes=7,
                sequence_length=X_train.shape[2]
            ).to(self.device)

            # Modify dropout rates
            model.dropout1.p = dropout1
            model.dropout2.p = dropout2

            # Train with suggested parameters
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            criterion = nn.CrossEntropyLoss()

            # Create data loader
            dataset = TensorDataset(X_train, y_train)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # Training loop
            model.train()
            for epoch in range(20):  # Quick training for optimization
                for batch_X, batch_y in dataloader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)

                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

            # Evaluate on validation set
            model.eval()
            with torch.no_grad():
                X_val_device = X_val.to(self.device)
                y_val_device = y_val.to(self.device)
                outputs = model(X_val_device)
                _, predicted = outputs.max(1)
                accuracy = (predicted == y_val_device).float().mean().item()

            return accuracy

        # Create study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        self.best_params = study.best_params
        self.logger.info(f"Best hyperparameters: {self.best_params}")

        return self.best_params

    def prepare_data(self, kline_data: Dict[str, pd.DataFrame],
                    sequence_length: int = 50,
                    prediction_horizon: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare data for CNN training/prediction - FIXED to avoid future leakage
        """
        self.prediction_horizon = prediction_horizon
        features_list = []
        timeframes = ['15m', '30m', '1h', '4h', '6h', '12h', '1d']

        # Extract features for each timeframe
        base_len = len(kline_data.get(timeframes[0], pd.DataFrame()))
        for tf in timeframes:
            if tf not in kline_data:
                # Create dummy data if timeframe missing
                features_list.append(np.zeros((base_len, 10)))
                continue

            df = kline_data[tf]
            features = self._extract_features(df)
            features_list.append(features)

        # Combine all timeframes
        combined_features = np.concatenate(features_list, axis=1)

        # Handle NaN and inf values
        combined_features = np.nan_to_num(combined_features, nan=0.0, posinf=1e6, neginf=-1e6)

        # Normalize
        if not self.is_trained:
            combined_features = self.scaler.fit_transform(combined_features)
        else:
            combined_features = self.scaler.transform(combined_features)

        # Create sequences
        X, y = [], []

        # For prediction mode (already trained)
        if self.is_trained:
            if len(combined_features) >= sequence_length:
                X.append(combined_features[-sequence_length:])
            X = torch.FloatTensor(X).unsqueeze(1) if X else torch.FloatTensor()
            return X, torch.LongTensor()

        # For training mode
        base_price_series = kline_data[timeframes[0]]['close']
        
        # FIX: Ensure alignment before indexing
        min_len = min(len(combined_features), len(base_price_series))
        combined_features = combined_features[:min_len]
        base_price_series = base_price_series.iloc[:min_len]
        
        max_idx = len(combined_features) - prediction_horizon

        for i in range(sequence_length, max_idx):
            X.append(combined_features[i-sequence_length:i])

            # Create labels (price direction)
            future_return = (base_price_series.iloc[i+prediction_horizon] -
                           base_price_series.iloc[i]) / base_price_series.iloc[i]

            if future_return > 0.01:  # 1% up
                label = 2  # Bull
            elif future_return < -0.01:  # 1% down
                label = 0  # Bear
            else:
                label = 1  # Neutral

            y.append(label)

        # Convert to tensors
        X = torch.FloatTensor(X).unsqueeze(1)  # Add channel dimension
        y = torch.LongTensor(y)

        return X, y

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract features from OHLCV data with error handling
        """
        features = []
        n_rows = len(df)
        epsilon = 1e-10

        # Price features
        features.append(df['close'].pct_change().fillna(0).values)
        features.append(((df['high'] - df['low']) / (df['close'] + epsilon)).fillna(0).values)
        features.append(((df['close'] - df['open']) / (df['open'] + epsilon)).fillna(0).values)

        # Volume features
        vol_ma = df['volume'].rolling(20).mean().fillna(method='bfill').fillna(0)
        features.append((df['volume'] / (vol_ma + epsilon)).fillna(1).values)

        # Technical indicators
        features.append(self._calculate_rsi(df['close']))

        # Moving averages
        for period in [10, 20, 50]:
            ma = df['close'].rolling(period).mean().fillna(method='bfill').fillna(0)
            features.append(((df['close'] - ma) / (ma + epsilon)).fillna(0).values)

        # Volatility
        features.append(df['close'].pct_change().rolling(20).std().fillna(0).values)

        # FIX: Use NaN and forward fill instead of zero padding
        features_array = np.full((n_rows, len(features)), np.nan)
        for i, f in enumerate(features):
            if len(f) == n_rows:
                features_array[:, i] = f
            else:
                features_array[-len(f):, i] = f
        
        # Forward fill NaN values and then backfill any remaining NaNs at the start
        features_array = pd.DataFrame(features_array).fillna(method='ffill').fillna(method='bfill').fillna(0).values

        return features_array

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> np.ndarray:
        """Calculate RSI with error handling"""
        if len(prices) < period:
            return np.full(len(prices), 50.0) / 100  # Default neutral

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()

        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        return rsi.fillna(50).values / 100  # Normalize to 0-1

    def train(self, X: torch.Tensor, y: torch.Tensor,
              epochs: int = 50, batch_size: int = 32,
              learning_rate: float = 0.001, use_best_params: bool = True):
        """
        Train the CNN model with optional hyperparameter optimization
        """
        # Initialize model if not exists
        if self.model is None:
            n_features = X.shape[3]
            n_timeframes = 7
            sequence_length = X.shape[2]

            self.model = MarketPredictionCNN(
                n_features=n_features // n_timeframes,
                n_timeframes=n_timeframes,
                sequence_length=sequence_length
            ).to(self.device)

        # Use optimized parameters if available
        if use_best_params and self.best_params is not None:
            learning_rate = self.best_params.get('lr', learning_rate)
            batch_size = self.best_params.get('batch_size', batch_size)
            weight_decay = self.best_params.get('weight_decay', 1e-5)
            self.model.dropout1.p = self.best_params.get('dropout1', 0.5)
            self.model.dropout2.p = self.best_params.get('dropout2', 0.3)
        else:
            weight_decay = 1e-5

        # Create data loader
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Loss and optimizer with gradient clipping
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

        # Training loop
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0

            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                # Check for NaN loss
                if torch.isnan(loss):
                    self.logger.warning(f"NaN loss detected at epoch {epoch+1}")
                    continue

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                # Statistics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()

            # Calculate metrics
            avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
            accuracy = 100. * correct / total if total > 0 else 0

            # Update learning rate
            scheduler.step(avg_loss)

            if (epoch + 1) % 10 == 0:
                self.logger.info(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

        self.is_trained = True

    def predict(self, X: torch.Tensor) -> Dict[str, Any]:
        """
        Make predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        self.model.eval()

        with torch.no_grad():
            X = X.to(self.device)
            outputs = self.model(X)
            probabilities = self.model.predict_proba(X)
            _, predicted = outputs.max(1)

        return {
            'predictions': predicted.cpu().numpy(),
            'probabilities': probabilities.cpu().numpy(),
            'labels': ['Bear', 'Neutral', 'Bull']
        }


class AdvancedMathematicalTools:
    """New mathematical tools implementation - TOP 5 PRIORITY"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def cointegration_and_vecm(self, data: pd.DataFrame, lag: int = 1) -> Dict[str, Any]:
        """Perform Cointegration Test and fit VECM with proper stationarity checks"""
        if data.shape[1] < 2:
            return {'error': 'VECM requires at least two time series.'}
        
        # Test for stationarity properly
        stationarity_results = {}
        data_to_use = data.copy()
        
        for col in data.columns:
            series = data[col].dropna()
            
            # Check for constant series
            if series.nunique() <= 1:
                return {'error': f'Series {col} is constant.'}
            
            # ADF test with multiple specifications
            try:
                # Test with constant
                adf_c = adfuller(series, regression='c')
                # Test with constant and trend
                adf_ct = adfuller(series, regression='ct')
                # Test with no constant
                adf_nc = adfuller(series, regression='nc')
                
                # Use the most appropriate specification
                min_pvalue = min(adf_c[1], adf_ct[1], adf_nc[1])
                
                stationarity_results[col] = {
                    'adf_statistic': adf_c[0],
                    'p_value': min_pvalue,
                    'is_stationary': min_pvalue < 0.05
                }
                
                # If not stationary, check if first difference is stationary
                if min_pvalue > 0.05:
                    diff_series = series.diff().dropna()
                    if len(diff_series) < 10:
                        return {'error': f'Series {col} too short after differencing.'}
                    
                    adf_diff = adfuller(diff_series, regression='c')
                    if adf_diff[1] < 0.05:
                        data_to_use[col] = series.diff()
                        stationarity_results[col]['differenced'] = True
                        stationarity_results[col]['diff_p_value'] = adf_diff[1]
                    else:
                        # Try second differencing
                        diff2_series = series.diff().diff().dropna()
                        if len(diff2_series) < 10:
                            return {'error': f'Series {col} is not I(1) or I(2).'}
                        
                        adf_diff2 = adfuller(diff2_series, regression='c')
                        if adf_diff2[1] < 0.05:
                            data_to_use[col] = series.diff().diff()
                            stationarity_results[col]['differenced'] = 2
                            stationarity_results[col]['diff2_p_value'] = adf_diff2[1]
                        else:
                            return {'error': f'Series {col} is not stationary even after second differencing.'}
                        
            except Exception as e:
                return {'error': f'Stationarity test failed for {col}: {str(e)}'}
        
        # Remove NaN from differencing
        data_to_use = data_to_use.dropna()
        
        if len(data_to_use) < 20:
            return {'error': 'Insufficient data after differencing.'}
        
        # Proceed with cointegration test and VECM
        try:
            # Import Johansen test
            from statsmodels.tsa.vector_ar.vecm import coint_johansen
            
            # Determine optimal lag using information criteria
            from statsmodels.tsa.api import VAR
            var_model = VAR(data_to_use)
            lag_order_results = var_model.select_order(maxlags=min(10, len(data_to_use)//5))
            optimal_lag = lag_order_results.aic
            
            # Use optimal lag or provided lag, whichever is smaller
            lag_to_use = min(lag, optimal_lag) if optimal_lag > 0 else lag
            
            # Johansen cointegration test
            joh_result = coint_johansen(data_to_use, det_order=0, k_ar_diff=lag_to_use)
            
            # Get critical values at different significance levels
            cv_90 = joh_result.cvt[:, 0]  # 90% critical values
            cv_95 = joh_result.cvt[:, 1]  # 95% critical values
            cv_99 = joh_result.cvt[:, 2]  # 99% critical values
            
            # Determine number of cointegrating relationships
            n_coint = 0
            for i in range(len(joh_result.lr1)):
                if joh_result.lr1[i] > cv_95[i]:
                    n_coint += 1
                else:
                    break
            
            # Check if cointegrated
            if n_coint > 0:
                # Fit VECM with identified cointegration rank
                model = VECM(data_to_use, k_ar_diff=lag_to_use, coint_rank=n_coint, deterministic='ci')
                vecm_res = model.fit()
                
                # Calculate error correction terms
                ect = data_to_use.values @ vecm_res.beta
                
                # Get short-run dynamics
                gamma_matrices = []
                for i in range(lag_to_use):
                    if hasattr(vecm_res, f'gamma{i}'):
                        gamma_matrices.append(getattr(vecm_res, f'gamma{i}'))
                
                # Calculate half-life of adjustment
                alpha = vecm_res.alpha
                eigenvalues = np.linalg.eigvals(alpha @ vecm_res.beta.T)
                half_lives = -np.log(2) / np.real(eigenvalues[eigenvalues != 0])
                half_lives = half_lives[half_lives > 0]  # Keep only positive half-lives
                
                return {
                    'stationarity_results': stationarity_results,
                    'optimal_lag': optimal_lag,
                    'lag_used': lag_to_use,
                    'johansen_trace_stats': joh_result.lr1.tolist(),
                    'johansen_critical_values': {
                        '90%': cv_90.tolist(),
                        '95%': cv_95.tolist(),
                        '99%': cv_99.tolist()
                    },
                    'n_cointegrating_vectors': n_coint,
                    'is_cointegrated': True,
                    'vecm_summary': str(vecm_res.summary()),
                    'alpha': vecm_res.alpha.tolist(),
                    'beta': vecm_res.beta.tolist(),
                    'gamma': gamma_matrices[0].tolist() if gamma_matrices else None,
                    'error_correction_terms': ect.tolist()[-5:],  # Last 5 ECT values
                    'half_lives': half_lives.tolist() if len(half_lives) > 0 else None,
                    'adjustment_speed': np.abs(alpha).mean()
                }
            else:
                # Not cointegrated, but still return useful information
                return {
                    'stationarity_results': stationarity_results,
                    'optimal_lag': optimal_lag,
                    'johansen_trace_stats': joh_result.lr1.tolist(),
                    'johansen_critical_values': {
                        '90%': cv_90.tolist(),
                        '95%': cv_95.tolist(), 
                        '99%': cv_99.tolist()
                    },
                    'is_cointegrated': False,
                    'error': 'No cointegration found at 5% significance level',
                    'suggestion': 'Consider VAR model instead of VECM for these non-cointegrated series'
                }
                
        except ImportError:
            return {'error': 'statsmodels.tsa.vector_ar.vecm module not available. Please install statsmodels>=0.12.0'}
        except Exception as e:
            return {
                'error': f'VECM fitting failed: {str(e)}',
                'stationarity_results': stationarity_results,
                'suggestion': 'Check data quality and ensure sufficient observations'
            }

    def multifractal_spectrum_analysis(self, signal: np.ndarray,
                                     q_range: np.ndarray = None,
                                     min_scale: int = 4,
                                     max_scale: int = None) -> Dict[str, Any]:
        """
        Multifractal Detrended Fluctuation Analysis (MF-DFA)
        Natural extension of DFA revealing multiple scaling behaviors
        """
        if q_range is None:
            q_range = np.linspace(-5, 5, 21)

        if max_scale is None:
            max_scale = len(signal) // 4

        # Cumulative sum
        y = np.cumsum(signal - np.mean(signal))

        # Scales
        scales = np.logspace(np.log10(min_scale), np.log10(max_scale), 20).astype(int)
        scales = np.unique(scales)

        # Initialize fluctuation function
        Fq = np.zeros((len(q_range), len(scales)))

        for scale_idx, scale in enumerate(scales):
            # Divide into segments
            n_segments = len(y) // scale
            if n_segments == 0:
                continue

            # Fluctuation for each segment
            fluctuations = []

            for seg in range(n_segments):
                # Forward direction
                segment = y[seg * scale:(seg + 1) * scale]
                x = np.arange(len(segment))
                coeffs = np.polyfit(x, segment, 1)
                fit = np.polyval(coeffs, x)
                fluct_fwd = np.sqrt(np.mean((segment - fit) ** 2))
                fluctuations.append(fluct_fwd)

                # Backward direction
                segment_bwd = y[len(y) - (seg + 1) * scale:len(y) - seg * scale]
                if len(segment_bwd) == scale:
                    coeffs_bwd = np.polyfit(x, segment_bwd, 1)
                    fit_bwd = np.polyval(coeffs_bwd, x)
                    fluct_bwd = np.sqrt(np.mean((segment_bwd - fit_bwd) ** 2))
                    fluctuations.append(fluct_bwd)

            # q-order fluctuation function
            for q_idx, q in enumerate(q_range):
                fluctuations_arr = np.array(fluctuations)
                if q == 0:
                    Fq[q_idx, scale_idx] = np.exp(0.5 * np.mean(np.log(fluctuations_arr[fluctuations_arr > 0] ** 2)))
                else:
                    Fq[q_idx, scale_idx] = np.mean(fluctuations_arr ** q) ** (1 / q)

        # Calculate Hurst exponents
        hurst_q = np.zeros(len(q_range))
        for q_idx in range(len(q_range)):
            # Log-log fit
            valid_idx = Fq[q_idx] > 0
            if np.sum(valid_idx) > 1:
                log_scales = np.log(scales[valid_idx])
                log_Fq = np.log(Fq[q_idx, valid_idx])
                hurst_q[q_idx] = np.polyfit(log_scales, log_Fq, 1)[0]

        # Multifractal spectrum
        tau_q = q_range * hurst_q - 1

        # Legendre transform
        alpha = np.gradient(tau_q, q_range)
        f_alpha = q_range * alpha - tau_q

        # Width of spectrum (multifractality measure)
        alpha_range = np.max(alpha) - np.min(alpha) if len(alpha) > 1 else 0.0

        return {
            'q_range': q_range,
            'hurst_q': hurst_q,
            'tau_q': tau_q,
            'alpha': alpha,
            'f_alpha': f_alpha,
            'multifractality': alpha_range,
            'dominant_hurst': hurst_q[len(q_range) // 2],  # q=0 Hurst
            'scales': scales,
            'Fq': Fq
        }

    def kyle_lambda_estimation(self, price_changes: np.ndarray,
                             signed_volume: np.ndarray,
                             method: str = 'regression') -> Dict[str, float]:
        """Kyle Lambda - Measures market depth and price impact"""

        # FIX: Add NaN guards at calculation entry points
        mask = ~(np.isnan(price_changes) | np.isnan(signed_volume) | np.isinf(price_changes) | np.isinf(signed_volume))
        if mask.sum() < 10:
            return {'lambda': 0.0, 'r_squared': 0.0, 'market_depth': np.inf}

        price_changes = price_changes[mask]
        signed_volume = signed_volume[mask]

        if method == 'regression':
            # Simple OLS regression: price_change = lambda * signed_volume
            X = signed_volume.reshape(-1, 1)
            y = price_changes

            # Add small regularization
            XtX = X.T @ X + 1e-10
            Xty = X.T @ y
            lambda_estimate = float(Xty / XtX) if XtX != 0 else 0.0

            # R-squared
            y_pred = lambda_estimate * signed_volume
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / (ss_tot + 1e-10))

        elif method == 'robust':
            # Robust estimation using median
            ratios = price_changes / (signed_volume + 1e-10)
            lambda_estimate = np.median(ratios[np.isfinite(ratios)])

            # Pseudo R-squared
            y_pred = lambda_estimate * signed_volume
            r_squared = np.corrcoef(price_changes, y_pred)[0, 1] ** 2

        # Market depth (inverse of lambda)
        market_depth = 1 / (abs(lambda_estimate) + 1e-10)

        # Additional microstructure metrics
        avg_trade_size = np.mean(np.abs(signed_volume))
        price_volatility = np.std(price_changes)

        # Amihud illiquidity (similar concept)
        amihud = np.mean(np.abs(price_changes) / (np.abs(signed_volume) + 1e-10))

        self.last_kyle_r_squared = r_squared
        
        return {
            'lambda': lambda_estimate,
            'r_squared': r_squared,
            'market_depth': market_depth,
            'avg_trade_size': avg_trade_size,
            'price_volatility': price_volatility,
            'amihud_illiquidity': amihud,
            'price_impact_per_unit': abs(lambda_estimate)
        }

    def recurrence_quantification_analysis(self, signal: np.ndarray,
                                         embedding_dim: int = 3,
                                         time_delay: int = 1,
                                         threshold: float = None) -> Dict[str, float]:
        """
        Recurrence Quantification Analysis (RQA)
        Reveals hidden periodicities and determinism
        """
        # Phase space reconstruction
        n = len(signal)
        m = embedding_dim
        tau = time_delay

        N = n - (m - 1) * tau
        if N <= 0:
            return {
                'recurrence_rate': 0.0,
                'determinism': 0.0,
                'laminarity': 0.0,
                'max_line': 0,
                'entropy': 0.0,
                'trapping_time': 0.0
            }

        # Embedded matrix
        embedded = np.zeros((N, m))
        for i in range(m):
            embedded[:, i] = signal[i * tau:i * tau + N]

        # Distance matrix
        dist_matrix = squareform(pdist(embedded))

        # Threshold selection (10% of max distance if not provided)
        if threshold is None:
            threshold = 0.1 * np.max(dist_matrix)

        # Recurrence matrix
        recurrence_matrix = (dist_matrix < threshold).astype(int)

        # RQA measures
        # 1. Recurrence Rate (RR)
        RR = np.sum(recurrence_matrix) / (N * N)

        # 2. Determinism (DET) - ratio of recurrence points forming diagonal lines
        diagonals = []
        for k in range(1 - N, N):
            diagonal = np.diag(recurrence_matrix, k)
            # Find consecutive 1s
            changes = np.diff(np.concatenate([[0], diagonal, [0]]))
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]
            lengths = ends - starts
            diagonals.extend(lengths[lengths >= 2])  # Min line length = 2

        if diagonals:
            DET = np.sum(diagonals) / np.sum(recurrence_matrix) if np.sum(recurrence_matrix) > 0 else 0.0
            max_line = np.max(diagonals)

            # Entropy of diagonal lines
            hist, _ = np.histogram(diagonals, bins=range(2, max(diagonals) + 2))
            p = hist / np.sum(hist) if np.sum(hist) > 0 else []
            p = p[p > 0]
            entropy_diag = -np.sum(p * np.log(p)) if len(p) > 0 else 0.0
        else:
            DET = 0.0
            max_line = 0
            entropy_diag = 0.0

        # 3. Laminarity (LAM) - ratio forming vertical lines
        verticals = []
        for i in range(N):
            column = recurrence_matrix[:, i]
            changes = np.diff(np.concatenate([[0], column, [0]]))
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]
            lengths = ends - starts
            verticals.extend(lengths[lengths >= 2])

        if verticals:
            LAM = np.sum(verticals) / np.sum(recurrence_matrix) if np.sum(recurrence_matrix) > 0 else 0.0
            trapping_time = np.mean(verticals)
        else:
            LAM = 0.0
            trapping_time = 0.0
        return {
            'recurrence_rate': RR,
            'determinism': DET,
            'laminarity': LAM,
            'max_line': float(max_line),
            'entropy': entropy_diag,
            'trapping_time': trapping_time,
            'threshold': threshold
        }

    def transfer_entropy_spectrum(self, source: np.ndarray,
                                target: np.ndarray,
                                lag_range: range = None,
                                bins: int = 5) -> Dict[str, Any]:
        """
        Transfer Entropy at multiple time lags
        Finds optimal prediction horizons
        """
        if lag_range is None:
            lag_range = range(1, min(30, len(source) // 10))

        # Discretize signals
        source_disc = pd.qcut(source, q=bins, labels=False, duplicates='drop')
        target_disc = pd.qcut(target, q=bins, labels=False, duplicates='drop')

        te_values = []
        
        min_len_base = min(len(source_disc), len(target_disc))

        for lag in lag_range:
            # FIX: Use non-overlapping alignment
            # Ensure we have enough data for the given lag
            if min_len_base <= lag:
                te_values.append(0.0)
                continue
            
            min_len = min_len_base - lag
            
            if min_len < 10:
                te_values.append(0.0)
                continue

            # Align data
            y_future = target_disc[lag : lag + min_len]
            y_past = target_disc[:min_len]
            x_past = source_disc[:min_len]

            # Calculate transfer entropy
            te = self._calculate_transfer_entropy(x_past, y_past, y_future)
            te_values.append(te)

        te_array = np.array(te_values)

        # Find optimal lag
        if len(te_array) > 0 and np.max(te_array) > 0:
            optimal_lag = list(lag_range)[np.argmax(te_array)]
            max_te = np.max(te_array)

            # Find significant lags (above mean + std)
            threshold = np.mean(te_array) + np.std(te_array)
            significant_lags = [lag for lag, te in zip(lag_range, te_array) if te > threshold]
        else:
            optimal_lag = 1
            max_te = 0.0
            significant_lags = []

        # Calculate cumulative information transfer
        cumulative_te = np.cumsum(te_array)

        return {
            'lag_range': list(lag_range),
            'te_values': te_array.tolist(),
            'optimal_lag': int(optimal_lag),
            'max_te': max_te,
            'significant_lags': significant_lags,
            'cumulative_te': cumulative_te.tolist(),
            'mean_te': np.mean(te_array),
            'total_information_transfer': np.sum(te_array)
        }

    def _calculate_transfer_entropy(self, x_past: np.ndarray,
                                  y_past: np.ndarray,
                                  y_future: np.ndarray) -> float:
        """Helper function to calculate transfer entropy with bias correction"""
        # Ensure arrays are the same length
        min_len = min(len(x_past), len(y_past), len(y_future))
        x_past = x_past[:min_len]
        y_past = y_past[:min_len]
        y_future = y_future[:min_len]
        
        if min_len < 10:
            return 0.0
        
        # Combine variables to create joint distributions
        # p(y_future, y_past, x_past)
        xyz = np.vstack([y_future, y_past, x_past]).T
        # p(y_future, y_past)
        yz = np.vstack([y_future, y_past]).T
        # p(y_past, x_past)
        yx = np.vstack([y_past, x_past]).T
        # p(y_past)
        y = y_past.reshape(-1, 1)
        
        # Calculate probabilities by counting unique occurrences
        p_xyz = self._get_probs(xyz)
        p_yz = self._get_probs(yz)
        p_yx = self._get_probs(yx)
        p_y = self._get_probs(y)
        
        # Check for valid probability distributions
        if len(p_xyz) == 0 or len(p_yz) == 0 or len(p_yx) == 0 or len(p_y) == 0:
            return 0.0
        
        # Calculate sample size for bias correction
        n_samples = len(x_past)
        
        # Count number of occupied bins for Miller-Madow bias correction
        n_bins_xyz = len(p_xyz)
        n_bins_yz = len(p_yz)
        n_bins_yx = len(p_yx)
        n_bins_y = len(p_y)
        
        # Miller-Madow bias correction for entropy estimation
        # Bias is approximately (m-1)/(2n) where m is number of occupied bins
        bias_xyz = (n_bins_xyz - 1) / (2 * n_samples)
        bias_yz = (n_bins_yz - 1) / (2 * n_samples)
        bias_yx = (n_bins_yx - 1) / (2 * n_samples)
        bias_y = (n_bins_y - 1) / (2 * n_samples)
        
        # Calculate entropies from probabilities with small constant for numerical stability
        epsilon = 1e-12
        h_xyz = -np.sum(p_xyz * np.log2(p_xyz + epsilon)) + bias_xyz
        h_yz = -np.sum(p_yz * np.log2(p_yz + epsilon)) + bias_yz
        h_yx = -np.sum(p_yx * np.log2(p_yx + epsilon)) + bias_yx
        h_y = -np.sum(p_y * np.log2(p_y + epsilon)) + bias_y
        
        # Transfer Entropy formula: T(X->Y) = H(Y_t, Y_{t-1}) + H(X_{t-1}, Y_{t-1}) - H(Y_{t-1}) - H(Y_t, Y_{t-1}, X_{t-1})
        te = h_yz + h_yx - h_y - h_xyz
        
        # Additional corrections for small sample sizes
        if n_samples < 50:
            # Apply stronger bias correction for small samples
            small_sample_correction = np.log2(n_samples) / n_samples
            te -= small_sample_correction
        
        # Significance testing using surrogate data (simplified)
        if te < 0:
            # Small negative values are likely due to estimation error
            if abs(te) < 0.01:  # Threshold for noise
                return 0.0
            
            # For larger negative values, check if it's significant
            # Estimate standard error using jackknife
            jackknife_estimates = []
            for i in range(min(n_samples, 100)):  # Limit jackknife samples for efficiency
                # Leave one out
                mask = np.ones(n_samples, dtype=bool)
                mask[i] = False
                
                # Recalculate with one sample removed
                xyz_jack = xyz[mask]
                yz_jack = yz[mask]
                yx_jack = yx[mask]
                y_jack = y[mask]
                
                # Quick probability calculation
                p_xyz_jack = self._get_probs(xyz_jack)
                p_yz_jack = self._get_probs(yz_jack)
                p_yx_jack = self._get_probs(yx_jack)
                p_y_jack = self._get_probs(y_jack)
                
                if len(p_xyz_jack) > 0 and len(p_yz_jack) > 0 and len(p_yx_jack) > 0 and len(p_y_jack) > 0:
                    h_xyz_jack = -np.sum(p_xyz_jack * np.log2(p_xyz_jack + epsilon))
                    h_yz_jack = -np.sum(p_yz_jack * np.log2(p_yz_jack + epsilon))
                    h_yx_jack = -np.sum(p_yx_jack * np.log2(p_yx_jack + epsilon))
                    h_y_jack = -np.sum(p_y_jack * np.log2(p_y_jack + epsilon))
                    
                    te_jack = h_yz_jack + h_yx_jack - h_y_jack - h_xyz_jack
                    jackknife_estimates.append(te_jack)
            
            if jackknife_estimates:
                # Calculate standard error
                te_std = np.std(jackknife_estimates)
                
                # If TE is within 2 standard errors of 0, consider it noise
                if abs(te) < 2 * te_std:
                    return 0.0
        
        # Apply Panzeri-Treves correction for limited sampling bias
        # This correction is more sophisticated than Miller-Madow for TE
        pt_correction = (n_bins_xyz - n_bins_yz - n_bins_yx + n_bins_y) / (2 * n_samples * np.log(2))
        te -= pt_correction
        
        # Ensure non-negative (transfer entropy should be non-negative)
        return max(0, te)

    def _get_probs(self, data: np.ndarray) -> np.ndarray:
        """Helper to get probabilities of unique rows/elements."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # Use pandas for efficient unique row counting
        df = pd.DataFrame(data)
        counts = df.value_counts(normalize=True)
        return counts.values

    def sample_entropy(self, signal: np.ndarray,
                      m: int = 2,
                      r: float = None,
                      normalize: bool = True) -> Dict[str, float]:
        """
        Sample Entropy - Robust complexity measure.
        
        Parameters:
        -----------
        signal : np.ndarray
            Input time series
        m : int
            Embedding dimension (pattern length)
        r : float
            Tolerance for matches (if None, set to 0.2 * std)
        normalize : bool
            Whether to normalize the signal first
            
        Returns:
        --------
        dict : Dictionary containing sample entropy and related metrics
        """
        N = len(signal)

        if N < m + 1:
            return {
                'sample_entropy': 0.0,
                'threshold': 0.0,
                'A': 0,
                'B': 0,
                'embedding_dimension': m,
                'signal_length': N,
                'error': 'Signal too short for given embedding dimension'
            }

        # Normalize signal if requested
        if normalize:
            signal_mean = np.mean(signal)
            signal_std = np.std(signal)
            if signal_std > 1e-10:
                signal = (signal - signal_mean) / signal_std
            else:
                return {
                    'sample_entropy': 0.0,
                    'threshold': 0.0,
                    'A': 0,
                    'B': 0,
                    'embedding_dimension': m,
                    'signal_length': N,
                    'error': 'Signal has zero variance'
                }

        # Set threshold as percentage of std if not provided
        if r is None:
            r = 0.2 * np.std(signal, ddof=0)
        
        # Ensure threshold is positive
        if r <= 0:
            r = 0.2  # Default fallback

        # Create templates of length m and m+1
        # More efficient implementation using array operations
        x = np.array([signal[i:i + m + 1] for i in range(N - m)])

        # Templates of length m (exclude last element)
        x_m = x[:, :-1]

        # Count matches for templates of length m
        # Using broadcasting for efficiency
        # Calculate chebyshev distance between all pairs of templates
        dist_m = np.max(np.abs(x_m[:, np.newaxis, :] - x_m[np.newaxis, :, :]), axis=2)

        # Count pairs where distance is less than r (excluding self-matches)
        B_matrix = (dist_m <= r).astype(int)
        np.fill_diagonal(B_matrix, 0)  # Exclude self-matches
        
        # Calculate probability B
        n_templates = N - m
        if n_templates > 1:
            B = np.sum(B_matrix) / ((n_templates) * (n_templates - 1))
        else:
            B = 0

        # Count matches for templates of length m+1 (using full x)
        dist_m_plus_1 = np.max(np.abs(x[:, np.newaxis, :] - x[np.newaxis, :, :]), axis=2)
        A_matrix = (dist_m_plus_1 <= r).astype(int)
        np.fill_diagonal(A_matrix, 0)  # Exclude self-matches
        
        # Calculate probability A
        if n_templates > 1:
            A = np.sum(A_matrix) / ((n_templates) * (n_templates - 1))
        else:
            A = 0

        # Calculate sample entropy
        if A > 0 and B > 0:
            sampen = -np.log(A / B)
        elif A == 0 and B > 0:
            # No matches at length m+1, but matches at length m
            sampen = np.inf  # Or a large number indicating high complexity/irregularity
        else:
            # No matches at all or B = 0
            sampen = np.inf

        # Additional complexity metrics
        # Calculate conditional probability
        conditional_prob = A / B if B > 0 else 0
        
        # Calculate the actual number of matches
        B_matches = np.sum(B_matrix) / 2  # Divide by 2 because matrix is symmetric
        A_matches = np.sum(A_matrix) / 2
        
        # Estimate optimal threshold using percentile of distances
        if N - m > 10:
            all_distances_m = dist_m[np.triu_indices_from(dist_m, k=1)]
            percentiles = np.percentile(all_distances_m, [10, 25, 50, 75, 90])
        else:
            percentiles = [r] * 5

        return {
            'sample_entropy': float(sampen) if np.isfinite(sampen) else -1.0,  # -1 indicates infinity
            'threshold': float(r),
            'A_matches': int(A_matches),  # Number of matches for m+1
            'B_matches': int(B_matches),  # Number of matches for m
            'A_probability': float(A),  # Probability of matches for m+1
            'B_probability': float(B),  # Probability of matches for m
            'conditional_probability': float(conditional_prob),
            'embedding_dimension': m,
            'signal_length': N,
            'n_templates': n_templates,
            'distance_percentiles': {
                '10th': float(percentiles[0]),
                '25th': float(percentiles[1]),
                '50th': float(percentiles[2]),
                '75th': float(percentiles[3]),
                '90th': float(percentiles[4])
            },
            'normalized': normalize,
            'complexity_assessment': self._assess_complexity(sampen)
        }
    
    def _assess_complexity(self, sampen: float) -> str:
        """
        Assess the complexity level based on sample entropy value
        """
        if not np.isfinite(sampen) or sampen < 0:
            return "MAXIMUM_COMPLEXITY"
        elif sampen < 0.1:
            return "VERY_LOW_COMPLEXITY"
        elif sampen < 0.3:
            return "LOW_COMPLEXITY"
        elif sampen < 0.6:
            return "MODERATE_COMPLEXITY"
        elif sampen < 1.0:
            return "HIGH_COMPLEXITY"
        elif sampen < 1.5:
            return "VERY_HIGH_COMPLEXITY"
        else:
            return "EXTREME_COMPLEXITY"


class MultiModalTensorFusion:
    """TOP PRIORITY 1: Unified tensor analysis - Pure NumPy implementation"""

    def __init__(self, lookback_size: int = 200):
        self.lookback_size = lookback_size
        self.simplified_analysis = SimplifiedTensorAnalysis(lookback_size)
        self.logger = logging.getLogger(__name__)

    def create_unified_market_tensor(self, data: Dict[str, pd.DataFrame]) -> np.ndarray:
        """Create simplified tensor for analysis"""
        return self.simplified_analysis.create_simplified_tensor(data)

    def tensor_decomposition_prediction(self, tensor: np.ndarray) -> Dict[str, Any]:
        """Perform tensor decomposition using pure NumPy"""
        return self.simplified_analysis.analyze_tensor(tensor)


class InformationFlowAnalyzer:
    """TOP PRIORITY 2: Analyze information flow between endpoints with NetworkX enhancements"""

    def __init__(self):
        self.endpoints = [
            'kline_price', 'orderflow', 'orderbook_pressure',
            'large_orders', 'liquidations', 'funding_rate',
            'open_interest', 'cvd', 'market_orders'
        ]
        self.max_lag = 20
        self.advanced_tools = AdvancedMathematicalTools()
        self.logger = logging.getLogger(__name__)

    def calculate_transfer_entropy_matrix(self, data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Calculate transfer entropy between all endpoint pairs"""
        active_endpoints = [e for e in data.keys() if not data[e].empty]
        n_endpoints = len(active_endpoints)

        te_matrix = np.zeros((n_endpoints, n_endpoints))
        lag_matrix = np.zeros((n_endpoints, n_endpoints), dtype=int)

        for i, source in enumerate(active_endpoints):
            for j, target in enumerate(active_endpoints):
                if i == j:
                    continue
                
                min_len = min(len(data[source]), len(data[target]))
                if min_len < self.max_lag * 2:
                    continue

                # Use advanced transfer entropy spectrum
                te_spectrum = self.advanced_tools.transfer_entropy_spectrum(
                    data[source].values,
                    data[target].values,
                    lag_range=range(1, min(self.max_lag + 1, min_len // 2))
                )

                te_matrix[i, j] = te_spectrum['max_te']
                lag_matrix[i, j] = te_spectrum['optimal_lag']

                time.sleep(0.01)  # Prevent CPU overload

        # Create readable DataFrame
        te_df = pd.DataFrame(te_matrix, index=active_endpoints, columns=active_endpoints)
        lag_df = pd.DataFrame(lag_matrix, index=active_endpoints, columns=active_endpoints)

        # Create and analyze network
        flow_network = self._create_flow_network(te_df, lag_df, active_endpoints)
        network_analysis = self._analyze_network_structure(flow_network)

        return {
            'transfer_entropy': te_df,
            'optimal_lags': lag_df,
            'information_flow_network': flow_network,
            'network_analysis': network_analysis
        }

    def _create_flow_network(self, te_df: pd.DataFrame, lag_df: pd.DataFrame,
                           active_endpoints: List[str]) -> nx.DiGraph:
        """Create directed graph of information flow"""
        G = nx.DiGraph()

        # Add nodes with attributes
        for endpoint in active_endpoints:
            G.add_node(endpoint, type='data_source')

        # Add edges where transfer entropy is significant
        te_values = te_df.values.flatten()
        te_values = te_values[te_values > 0]
        if len(te_values) == 0:
            return G
            
        threshold = np.mean(te_values) + np.std(te_values)

        for i, source in enumerate(active_endpoints):
            for j, target in enumerate(active_endpoints):
                if i != j and te_df.iloc[i, j] > threshold:
                    G.add_edge(
                        source,
                        target,
                        weight=te_df.iloc[i, j],
                        lag=lag_df.iloc[i, j],
                        significance='high' if te_df.iloc[i, j] > threshold * 1.5 else 'medium'
                    )

        return G

    def _analyze_network_structure(self, G: nx.DiGraph) -> Dict[str, Any]:
        """Enhanced network analysis using NetworkX"""
        if G.number_of_nodes() == 0:
            return {}

        try:
            # Calculate various centrality measures
            in_degree_centrality = nx.in_degree_centrality(G)
            out_degree_centrality = nx.out_degree_centrality(G)
            eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-04)
            pagerank = nx.pagerank(G)

            # Information flow paths
            if nx.is_weakly_connected(G):
                avg_path_length = nx.average_shortest_path_length(G, weight='weight')
            else:
                # Calculate for largest component
                largest_cc = max(nx.weakly_connected_components(G), key=len)
                if len(largest_cc) > 1:
                    subgraph = G.subgraph(largest_cc)
                    avg_path_length = nx.average_shortest_path_length(subgraph, weight='weight')
                else:
                    avg_path_length = 0
                # Add info about connectivity
                self.logger.info(f"Graph has {nx.number_weakly_connected_components(G)} weakly connected components")

            # Clustering coefficient
            clustering = nx.clustering(G)

            # Find information hubs (high in-degree) and sources (high out-degree)
            hubs = sorted(in_degree_centrality.items(), key=lambda x: x[1], reverse=True)[:3]
            sources = sorted(out_degree_centrality.items(), key=lambda x: x[1], reverse=True)[:3]

            # Detect communities using modularity
            communities = []
            if G.to_undirected().number_of_edges() > 0:
                communities = list(nx.community.greedy_modularity_communities(G.to_undirected()))

            return {
                'network_density': nx.density(G),
                'avg_path_length': avg_path_length,
                'clustering_coefficient': np.mean(list(clustering.values())),
                'top_information_hubs': hubs,
                'top_information_sources': sources,
                'pagerank_scores': pagerank,
                'eigenvector_centrality': eigenvector_centrality,
                'n_communities': len(communities),
                'communities': [list(c) for c in communities],
                'is_dag': nx.is_directed_acyclic_graph(G),
                'strongly_connected_components': [list(c) for c in nx.strongly_connected_components(G)]
            }

        except Exception as e:
            self.logger.warning(f"Network analysis error: {e}")
            return {
                'error': str(e),
                'network_density': nx.density(G) if G.number_of_nodes() > 0 else 0,
                'warning': 'Graph analysis incomplete',
                'n_components': len(list(nx.weakly_connected_components(G))) if G.number_of_nodes() > 0 else 0
            }

        except Exception as e:
            self.logger.warning(f"Network analysis error: {e}")
            return {'error': str(e)}

    def partial_information_decomposition(self, data: Dict[str, pd.Series]) -> Dict[str, float]:
        """Decompose information contributions from each endpoint"""
        if 'kline_price' not in data:
            return {}

        target = data['kline_price']
        contributions = {}

        # Calculate mutual information for each source
        for source_name, source_data in data.items():
            if source_name == 'kline_price' or source_data.empty:
                continue

            # Align data
            min_len = min(len(target), len(source_data))
            if min_len < 10:
                continue

            try:
                # Calculate mutual information
                mi = mutual_info_regression(
                    source_data[:min_len].values.reshape(-1, 1),
                    target[:min_len].values
                )[0]

                contributions[source_name] = mi
            except Exception as e:
                self.logger.warning(f"MI calculation failed for {source_name}: {e}")
                contributions[source_name] = 0.0

        # Calculate unique, redundant, and synergistic information
        total_mi = sum(contributions.values())

        return {
            'individual_contributions': contributions,
            'total_information': total_mi,
            'redundancy_factor': self._estimate_redundancy(data),
            'synergy_potential': self._estimate_synergy(data)
        }

    def _estimate_redundancy(self, data: Dict[str, pd.Series]) -> float:
        """Estimate information redundancy between sources"""
        correlations = []

        sources = [k for k in data.keys() if k != 'kline_price' and not data[k].empty]
        for i, source1 in enumerate(sources):
            for j, source2 in enumerate(sources):
                if i < j:
                    min_len = min(len(data[source1]), len(data[source2]))
                    if min_len > 10:
                        try:
                            corr = np.corrcoef(
                                data[source1][:min_len],
                                data[source2][:min_len]
                            )[0, 1]
                            if np.isfinite(corr):
                                correlations.append(abs(corr))
                        except:
                            pass

        return np.mean(correlations) if correlations else 0.0

    def _estimate_synergy(self, data: Dict[str, pd.Series]) -> float:
        """Estimate synergistic information potential"""
        if len(data) < 3:
            return 0.0

        correlations = []
        sources = [k for k in data.keys() if not data[k].empty]
        for key1, key2 in product(sources, repeat=2):
            if key1 < key2:
                min_len = min(len(data[key1]), len(data[key2]))
                if min_len > 10:
                    try:
                        corr = np.corrcoef(data[key1][:min_len], data[key2][:min_len])[0, 1]
                        if np.isfinite(corr):
                            correlations.append(corr)
                    except:
                        pass

        return np.std(correlations) if correlations else 0.0

    def calculate_conditional_mutual_information(self, data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Calculate CMI between data streams"""
        # Prepare data arrays
        data_arrays = {}
        min_len = min((len(series) for series in data.values() if not series.empty), default=0)
        if min_len == 0:
            return {'error': 'No data available for CMI'}

        for key, series in data.items():
            if not series.empty:
                # Convert to numpy array if needed
                if hasattr(series, 'values'):
                    data_arrays[key] = series.values[:min_len]
                else:
                    data_arrays[key] = np.array(series)[:min_len]

        # Calculate CMI matrix with price as conditioning variable
        if 'kline_price' in data_arrays:
            cmi_matrix = ConditionalMutualInformation.calculate_cmi_matrix(
                data_arrays,
                condition_on='kline_price'
            )

            # Calculate synergy scores
            synergy_scores = {}

            # Check three-way relationships
            endpoints = [k for k in data_arrays.keys() if k != 'kline_price']

            for i, var1 in enumerate(endpoints):
                for j, var2 in enumerate(endpoints):
                    if i < j:
                        # Calculate synergy: I(X,Y;Price) - I(X;Price) - I(Y;Price)
                        try:
                            X = data_arrays[var1]
                            Y = data_arrays[var2]
                            Z = data_arrays['kline_price']

                            # Joint MI of X,Y with Price
                            XY = np.column_stack([X, Y])
                            joint_mi = mutual_info_regression(XY, Z)[0]

                            # Individual MIs
                            mi_x = mutual_info_regression(X.reshape(-1, 1), Z)[0]
                            mi_y = mutual_info_regression(Y.reshape(-1, 1), Z)[0]

                            synergy = joint_mi - mi_x - mi_y
                            synergy_scores[f"{var1}_{var2}"] = synergy
                        except Exception as e:
                            self.logger.warning(f"Synergy calculation failed for {var1}-{var2}: {e}")
                            synergy_scores[f"{var1}_{var2}"] = 0.0

            return {
                'cmi_matrix': cmi_matrix,
                'synergy_scores': synergy_scores,
                'top_synergies': sorted(synergy_scores.items(),
                                       key=lambda x: x[1], reverse=True)[:5]
            }

        return {'error': 'No kline_price data available for CMI analysis'}
class MicrostructureKlineBridge:
    """TOP PRIORITY 3: Bridge microstructure to kline predictions"""

    def __init__(self):
        self.aggregation_methods = ['time', 'volume', 'dollar', 'information']
        self.microprice_weights = {'bid': 0.3, 'ask': 0.3, 'mid': 0.4}
        self.advanced_tools = AdvancedMathematicalTools()
        self.xgb_model = None
        self.logger = logging.getLogger(__name__)

    def train_xgboost_model(self, features: pd.DataFrame, target: pd.Series):
        """Train an XGBoost model for prediction."""
        self.xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        self.xgb_model.fit(features, target)

    def predict_with_xgboost(self, features: pd.DataFrame) -> Optional[float]:
        """Predict using the trained XGBoost model."""
        if self.xgb_model:
            return self.xgb_model.predict(features)[0]
        return None

    def predict_current_bar_close(self,
                                  current_bar_data: Dict[str, Any],
                                  orderflow_data: pd.DataFrame,
                                  book_data: pd.DataFrame) -> Dict[str, float]:
        """Predict current kline close with accurate time remaining"""
        
        # Calculate actual time remaining
        current_time = pd.Timestamp.now(tz='UTC')
        
        # Determine bar duration based on context
        if 'timeframe' in current_bar_data:
            tf = current_bar_data['timeframe']
            duration_map = {
                '15m': pd.Timedelta(minutes=15),
                '30m': pd.Timedelta(minutes=30),
                '1h': pd.Timedelta(hours=1),
                '4h': pd.Timedelta(hours=4),
                '6h': pd.Timedelta(hours=6),
                '12h': pd.Timedelta(hours=12),
                '1d': pd.Timedelta(days=1)
            }
            bar_duration = duration_map.get(tf, pd.Timedelta(minutes=15))
        else:
            bar_duration = pd.Timedelta(minutes=15)  # Default to 15m
        
        # Get bar start time from current_bar_data or estimate it
        if 'bar_start_time' in current_bar_data:
            bar_start = pd.Timestamp(current_bar_data['bar_start_time'])
        else:
            # Estimate based on current time and bar duration
            if bar_duration.total_seconds() <= 3600:  # For hour or less
                minutes_in_duration = int(bar_duration.total_seconds() / 60)
                minutes_since_hour = current_time.minute
                bar_index = minutes_since_hour // minutes_in_duration
                bar_start = current_time.replace(minute=bar_index * minutes_in_duration, second=0, microsecond=0)
            else:  # For longer durations
                # Align to day boundaries
                if bar_duration == pd.Timedelta(days=1):
                    bar_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
                else:
                    hours_in_duration = int(bar_duration.total_seconds() / 3600)
                    hour_index = current_time.hour // hours_in_duration
                    bar_start = current_time.replace(hour=hour_index * hours_in_duration, minute=0, second=0, microsecond=0)
        
        time_elapsed = current_time - bar_start
        time_remaining = bar_duration - time_elapsed
        time_remaining_ratio = max(0, min(1, time_remaining.total_seconds() / bar_duration.total_seconds()))
        
        # Update current_bar_data with actual ratio
        current_bar_data['time_remaining_ratio'] = time_remaining_ratio
        current_bar_data['time_elapsed_ratio'] = 1 - time_remaining_ratio

        predictions = {}

        # 1. Microprice estimation
        microprice = self.calculate_microprice(book_data)
        if microprice > 0:
            predictions['microprice'] = microprice

        # 2. VWAP projection
        if not orderflow_data.empty:
            vwap_projection = self.project_vwap(orderflow_data, current_bar_data)
            if vwap_projection > 0:
                predictions['vwap_projection'] = vwap_projection

        # 3. Order flow imbalance projection
        if 'delta' in orderflow_data.columns:
            ofi_projection = self.project_from_order_flow(orderflow_data)
            if ofi_projection > 0:
                predictions['ofi_projection'] = ofi_projection

        # 4. Kyle Lambda based projection
        if not orderflow_data.empty and 'price' in orderflow_data.columns and 'volume' in orderflow_data.columns:
            kyle_projection = self.kyle_lambda_projection(orderflow_data)
            if kyle_projection > 0:
                predictions['kyle_lambda_projection'] = kyle_projection
        elif not orderflow_data.empty and 'total_bid' in orderflow_data.columns:
            # Alternative: use order flow imbalance for Kyle Lambda
            orderflow_data = orderflow_data.copy()
            orderflow_data['volume'] = orderflow_data['total_bid'] + orderflow_data['total_ask']
            orderflow_data['price'] = current_bar_data.get('current', 0)
            if orderflow_data['price'].iloc[0] > 0:
                kyle_projection = self.kyle_lambda_projection(orderflow_data)
                if kyle_projection > 0:
                    predictions['kyle_lambda_projection'] = kyle_projection

        # 5. Momentum-based projection (adjusted for time remaining)
        momentum_projection = self.momentum_projection(current_bar_data)
        if momentum_projection > 0:
            predictions['momentum_projection'] = momentum_projection

        # 6. Time-weighted projection
        current_price = current_bar_data.get('current', 0)
        open_price = current_bar_data.get('open', current_price)
        
        if current_price > 0 and open_price > 0:
            # Simple time-weighted projection
            price_change_so_far = current_price - open_price
            
            # Assume linear progression (can be refined)
            if time_elapsed.total_seconds() > 0:
                rate_of_change = price_change_so_far / (1 - time_remaining_ratio + 1e-10)
                time_weighted_projection = open_price + rate_of_change
                predictions['time_weighted'] = time_weighted_projection

        # 7. XGBoost prediction
        if self.xgb_model:
            features = self._prepare_xgb_features(current_bar_data, orderflow_data, book_data)
            xgb_pred = self.predict_with_xgboost(features)
            if xgb_pred:
                predictions['xgboost_prediction'] = float(xgb_pred)

        # 8. Weighted ensemble prediction (time-aware weights)
        # Adjust weights based on time remaining
        if time_remaining_ratio > 0.7:  # Early in the bar
            weights = {
                'microprice': 0.10,
                'vwap_projection': 0.10,
                'ofi_projection': 0.15,
                'kyle_lambda_projection': 0.15,
                'momentum_projection': 0.15,
                'time_weighted': 0.05,
                'xgboost_prediction': 0.30
            }
        elif time_remaining_ratio > 0.3:  # Middle of the bar
            weights = {
                'microprice': 0.15,
                'vwap_projection': 0.15,
                'ofi_projection': 0.15,
                'kyle_lambda_projection': 0.15,
                'momentum_projection': 0.10,
                'time_weighted': 0.10,
                'xgboost_prediction': 0.20
            }
        else:  # Near the end of the bar
            weights = {
                'microprice': 0.20,
                'vwap_projection': 0.20,
                'ofi_projection': 0.10,
                'kyle_lambda_projection': 0.10,
                'momentum_projection': 0.05,
                'time_weighted': 0.15,
                'xgboost_prediction': 0.20
            }

        valid_predictions = {k: v for k, v in predictions.items() 
                           if v is not None and v > 0 and np.isfinite(v)}
        
        if valid_predictions:
            total_weight = sum(weights.get(k, 0.1) for k in valid_predictions)
            if total_weight > 0:
                ensemble_prediction = sum(
                    valid_predictions[k] * weights.get(k, 0.1) / total_weight
                    for k in valid_predictions
                )
            else:
                ensemble_prediction = current_bar_data.get('current', 0)
        else:
            ensemble_prediction = current_bar_data.get('current', 0)

        predictions['ensemble'] = ensemble_prediction
        
        # Calculate confidence with time-awareness
        predictions['confidence'] = self.calculate_prediction_confidence(
            predictions, current_bar_data
        )
        
        # Adjust confidence based on time remaining
        # Higher confidence as we approach bar close
        time_confidence_multiplier = 0.7 + (0.3 * (1 - time_remaining_ratio))
        predictions['confidence'] *= time_confidence_multiplier
        predictions['confidence'] = float(np.clip(predictions['confidence'], 5, 95))
        
        # Add metadata
        predictions['time_remaining_ratio'] = time_remaining_ratio
        predictions['bar_duration_minutes'] = bar_duration.total_seconds() / 60
        predictions['predictions_used'] = len(valid_predictions)

        return predictions

    def _prepare_xgb_features(self, current_bar_data, orderflow_data, book_data) -> pd.DataFrame:
        """Prepare features for the XGBoost model."""
        features = {}

        # Current bar features
        features['open'] = current_bar_data.get('open', 0)
        features['high'] = current_bar_data.get('high', 0)
        features['low'] = current_bar_data.get('low', 0)
        features['current'] = current_bar_data.get('current', 0)
        features['range'] = features['high'] - features['low']
        features['time_remaining_ratio'] = current_bar_data.get('time_remaining_ratio', 0.5)

        # Orderflow features
        if not orderflow_data.empty:
            features['cum_delta'] = orderflow_data['delta'].sum() if 'delta' in orderflow_data else 0
            features['total_volume'] = orderflow_data['volume'].sum() if 'volume' in orderflow_data else 0
            
            # FIX: Check existence of 'price' and 'volume' before use
            if 'price' in orderflow_data.columns and 'volume' in orderflow_data.columns and features['total_volume'] > 0:
                features['vwap'] = (orderflow_data['price'] * orderflow_data['volume']).sum() / (features['total_volume'] + 1e-10)
            else:
                features['vwap'] = features.get('current', 0)  # Use current price as fallback
        else:
            features['cum_delta'] = 0
            features['total_volume'] = 0
            features['vwap'] = features.get('current', 0)

        # Book features
        if not book_data.empty:
            latest_book = book_data.iloc[-1]
            features['book_imbalance'] = (latest_book['bid_size'] - latest_book['ask_size']) / (latest_book['bid_size'] + latest_book['ask_size'] + 1e-10) if 'bid_size' in latest_book and 'ask_size' in latest_book else 0
            features['microprice'] = self.calculate_microprice(book_data)
        else:
            features['book_imbalance'] = 0
            features['microprice'] = 0

        return pd.DataFrame([features])

    def calculate_microprice(self, book_data: pd.DataFrame) -> float:
        """Calculate microprice from order book"""
        if book_data.empty:
            return 0.0

        latest_book = book_data.iloc[-1]

        # Weighted mid price based on book imbalance
        if 'bid_size' in latest_book and 'ask_size' in latest_book:
            bid_size = float(latest_book['bid_size'])
            ask_size = float(latest_book['ask_size'])
            bid_price = float(latest_book.get('bid_price', latest_book.get('bid', 0)))
            ask_price = float(latest_book.get('ask_price', latest_book.get('ask', 0)))

            if bid_size + ask_size > 0 and bid_price > 0 and ask_price > 0:
                microprice = (bid_price * ask_size + ask_price * bid_size) / (bid_size + ask_size)
            else:
                microprice = (bid_price + ask_price) / 2 if bid_price > 0 and ask_price > 0 else 0
        else:
            microprice = float(latest_book.get('mid', 0))

        return microprice

    def kyle_lambda_projection(self, orderflow_data: pd.DataFrame) -> float:
        """Use Kyle Lambda for price impact projection"""
        if len(orderflow_data) < 10:
            return 0.0

        # Calculate signed volume and price changes
        prices = orderflow_data['price'].values
        volumes = orderflow_data['volume'].values

        # Determine sign based on price movement or delta
        if 'delta' in orderflow_data.columns:
            signed_volume = np.sign(orderflow_data['delta'].values) * volumes
        else:
            price_changes = np.diff(prices)
            price_signs = np.concatenate([[1], np.sign(price_changes)])
            signed_volume = price_signs * volumes

        # Calculate price changes
        price_changes = np.diff(prices)
        if len(price_changes) == 0:
            return float(prices[-1])

        # Estimate Kyle Lambda
        kyle_result = self.advanced_tools.kyle_lambda_estimation(
            price_changes,
            signed_volume[1:]  # Align with price changes
        )

        # Project price based on recent order flow
        recent_signed_volume = np.sum(signed_volume[-5:])  # Last 5 trades
        lambda_estimate = kyle_result['lambda']

        # Price projection
        projected_change = lambda_estimate * recent_signed_volume
        projection = float(prices[-1] + projected_change)

        return projection

    def project_vwap(self, orderflow_data: pd.DataFrame,
                     current_bar_data: Dict[str, Any]) -> float:
        """Project VWAP to end of current bar"""
        if orderflow_data.empty:
            return current_bar_data.get('close', 0)

        # Calculate current VWAP
        if 'price' in orderflow_data.columns and 'volume' in orderflow_data.columns:
            total_volume = orderflow_data['volume'].sum()
            if total_volume > 0:
                vwap = (orderflow_data['price'] * orderflow_data['volume']).sum() / total_volume
            else:
                vwap = orderflow_data['price'].mean()
        else:
            vwap = current_bar_data.get('close', 0)

        # Project based on recent trend
        if len(orderflow_data) > 5 and 'price' in orderflow_data.columns:
            recent_trend = (orderflow_data['price'].iloc[-1] -
                           orderflow_data['price'].iloc[-5]) / 5
            time_remaining = current_bar_data.get('time_remaining_ratio', 0.5)
            vwap_projection = vwap + (recent_trend * time_remaining * 5)
        else:
            vwap_projection = vwap

        return float(vwap_projection)

    def project_from_order_flow(self, orderflow_data: pd.DataFrame) -> float:
        """Project price based on order flow imbalance"""
        if 'delta' not in orderflow_data.columns or orderflow_data.empty:
            return 0.0

        # Cumulative delta
        cum_delta = orderflow_data['delta'].cumsum().iloc[-1]

        # Delta acceleration
        if len(orderflow_data) > 10:
            recent_delta = orderflow_data['delta'].iloc[-10:].mean()
            older_delta = orderflow_data['delta'].iloc[-20:-10].mean() if len(orderflow_data) > 20 else recent_delta
            delta_acceleration = recent_delta - older_delta
        else:
            delta_acceleration = 0

        # Price projection based on delta
        price_base = float(orderflow_data['price'].iloc[-1]) if 'price' in orderflow_data.columns else 0

        if price_base == 0:
            return 0.0

        # Normalize delta impact
        avg_price_change = orderflow_data['price'].diff().abs().mean() if 'price' in orderflow_data.columns else 1
        delta_impact = (cum_delta / (abs(cum_delta) + 1000)) * avg_price_change

        projection = price_base + delta_impact + (delta_acceleration * 0.1)

        return float(projection)

    def momentum_projection(self, current_bar_data: Dict[str, Any]) -> float:
        """Simple momentum-based projection"""
        open_price = current_bar_data.get('open', 0)
        current_price = current_bar_data.get('current', open_price)
        high = current_bar_data.get('high', current_price)
        low = current_bar_data.get('low', current_price)

        if open_price == 0:
            return current_price

        # Current momentum
        momentum = (current_price - open_price) / (open_price + 1e-10)

        # Range position
        range_size = high - low
        if range_size > 0:
            range_position = (current_price - low) / range_size
        else:
            range_position = 0.5

        # Project based on momentum and range position
        if range_position > 0.8:  # Near highs
            projection = current_price + (momentum * current_price * 0.3)
        elif range_position < 0.2:  # Near lows
            projection = current_price + (momentum * current_price * 0.7)
        else:  # Middle of range
            projection = current_price + (momentum * current_price * 0.5)

        return float(projection)

    def calculate_prediction_confidence(self, predictions: Dict[str, float],
                                      current_bar_data: Dict[str, Any]) -> float:
        """Calculate confidence based on multiple factors"""
        # Get prediction values
        pred_values = [v for k, v in predictions.items()
                      if k not in ['ensemble', 'confidence'] and v is not None
                      and np.isfinite(v) and v > 0]
        
        if not pred_values or len(pred_values) < 2:
            return 0.0
        
        # 1. Agreement between methods (40% weight)
        pred_std = np.std(pred_values)
        pred_mean = np.mean(pred_values)
        cv = pred_std / (pred_mean + 1e-10)
        agreement_score = max(0, 1 - cv) * 100  # 0-100
        
        # 2. Price range position (20% weight)
        current = current_bar_data.get('current', 0)
        high = current_bar_data.get('high', current)
        low = current_bar_data.get('low', current)
        range_size = high - low
        
        if range_size > 0:
            # Higher confidence when price is trending (near high/low)
            range_position = (current - low) / range_size
            if range_position > 0.8 or range_position < 0.2:
                range_score = 80
            elif range_position > 0.7 or range_position < 0.3:
                range_score = 60
            else:
                range_score = 40
        else:
            range_score = 50
        
        # 3. Model-specific confidences (30% weight)
        model_scores = []
        
        # Kyle Lambda R-squared
        if 'kyle_lambda_projection' in predictions and hasattr(self.advanced_tools, 'last_kyle_r_squared'):
            kyle_confidence = self.advanced_tools.last_kyle_r_squared * 100
            model_scores.append(kyle_confidence)
        
        # XGBoost feature importance alignment
        if 'xgboost_prediction' in predictions and self.xgb_model:
            # Use prediction variance as confidence proxy
            xgb_confidence = 70  # Base confidence for trained model
            model_scores.append(xgb_confidence)
        
        # VWAP trend strength
        if 'vwap_projection' in predictions:
            vwap_confidence = 60  # Base VWAP confidence
            model_scores.append(vwap_confidence)
        
        avg_model_score = np.mean(model_scores) if model_scores else 50
        
        # 4. Data quality (10% weight)
        data_quality_score = 100
        if 'error' in str(predictions.values()):
            data_quality_score = 20
        elif len(pred_values) < 4:
            data_quality_score = 60
        
        # Calculate weighted confidence
        confidence = (
            agreement_score * 0.4 +
            range_score * 0.2 +
            avg_model_score * 0.3 +
            data_quality_score * 0.1
        )
        
        # Boost confidence for strong directional agreement
        if cv < 0.05:  # Very low coefficient of variation
            confidence = min(95, confidence * 1.2)
        
        # Penalize for high disagreement
        if cv > 0.5:
            confidence = max(20, confidence * 0.7)
        
        return float(np.clip(confidence, 5, 95))


class HeikinAshiAnalysis:
    """
    Merged HeikinAshiMSSignal and ZScoreHeikinAshi into one class.
    """

    def __init__(self, df: pd.DataFrame, z_lookback_period: int = 20):
        self.df = df.copy()
        self.z_lookback_period = z_lookback_period
        self.ha_df = pd.DataFrame(index=df.index)

    def calculate_all_signals(self) -> pd.DataFrame:
        """Calculate all Heikin Ashi related signals and transformations."""
        self._calculate_heikin_ashi_base()

        results = pd.DataFrame(index=self.df.index)
        results['ha_open'] = self.ha_df['open']
        results['ha_high'] = self.ha_df['high']
        results['ha_low'] = self.ha_df['low']
        results['ha_close'] = self.ha_df['close']

        # Calculate Z-Score Heikin Ashi
        z_ha_results = self._calculate_z_score_ha()
        results = results.join(z_ha_results)

        # Calculate MS-Signal indicators
        ms_signal_results = self._calculate_ms_signal_indicators()
        results = results.join(ms_signal_results)

        return results.fillna(0)

    def _calculate_heikin_ashi_base(self):
        """Calculate standard Heikin Ashi candles."""
        self.ha_df['close'] = (self.df['open'] + self.df['high'] + self.df['low'] + self.df['close']) / 4

        ha_open = pd.Series(index=self.df.index, dtype=float)
        ha_open.iloc[0] = (self.df['open'].iloc[0] + self.df['close'].iloc[0]) / 2

        for i in range(1, len(self.df)):
            ha_open.iloc[i] = (ha_open.iloc[i-1] + self.ha_df['close'].iloc[i-1]) / 2

        self.ha_df['open'] = ha_open
        self.ha_df['high'] = pd.concat([self.df['high'], self.ha_df['open'], self.ha_df['close']], axis=1).max(axis=1)
        self.ha_df['low'] = pd.concat([self.df['low'], self.ha_df['open'], self.ha_df['close']], axis=1).min(axis=1)

    def _calculate_z_score_ha(self) -> pd.DataFrame:
        """Calculate Z-Score transformed Heikin Ashi candles."""
        z_ha_df = pd.DataFrame(index=self.df.index)
        for col in ['open', 'high', 'low', 'close']:
            mean = self.ha_df[col].rolling(self.z_lookback_period).mean()
            std = self.ha_df[col].rolling(self.z_lookback_period).std()
            z_ha_df[f'z_ha_{col}'] = (self.ha_df[col] - mean) / (std + 1e-10)
        return z_ha_df.fillna(0)

    def _calculate_ms_signal_indicators(self) -> pd.DataFrame:
        """Calculate the components of the MS-Signal indicator."""
        ms_df = pd.DataFrame(index=self.df.index)

        try:
            ha_rsi = talib.RSI(self.ha_df['close'], timeperiod=14)
        except Exception:
            ha_rsi = self._calculate_fallback_rsi(self.ha_df['close'])
        ms_df['ha_rsi'] = ha_rsi

        ms_df['ha_high_signal'], ms_df['ha_low_signal'] = self._calculate_ha_high_low(ha_rsi)
        ms_df['stoch_rsi'] = self._calculate_stoch_rsi(ha_rsi)

        obv, h1, l1, hlmid, obv_check = self._calculate_obv_signals()
        ms_df['obv'] = obv
        ms_df['obv_check'] = obv_check

        ms_df['dmi_signal'] = self._calculate_dmi_signals()

        mom0, mom1 = self._calculate_momentum_signals()
        ms_df['mom0'] = mom0
        ms_df['mom1'] = mom1

        ms_df['sum_signal'] = (ms_df['dmi_signal'] + ms_df['obv_check'] + ms_df['mom0'] + ms_df['mom1'])

        return ms_df.fillna(0)

    def _calculate_fallback_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/period, adjust=False).mean()
        loss = -delta.where(delta < 0, 0).ewm(alpha=1/period, adjust=False).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def _calculate_ha_high_low(self, rsi: pd.Series) -> Tuple[pd.Series, pd.Series]:
        c = self.ha_df['close']
        o = self.ha_df['open']

        ha_high = pd.Series(np.nan, index=self.df.index, dtype=float)
        ha_low = pd.Series(np.nan, index=self.df.index, dtype=float)

        for i in range(2, len(self.df)):
            if (c.iloc[i-2] > o.iloc[i-2] and c.iloc[i-1] > o.iloc[i-1] and
                c.iloc[i] < o.iloc[i] and rsi.iloc[i-2] >= 70):
                ha_high.iloc[i] = o.iloc[i-2]

            if (c.iloc[i-2] < o.iloc[i-2] and c.iloc[i-1] < o.iloc[i-1] and
                c.iloc[i] > o.iloc[i] and rsi.iloc[i-2] <= 30):
                ha_low.iloc[i] = o.iloc[i-2]

        return ha_high.ffill(), ha_low.ffill()

    def _calculate_stoch_rsi(self, rsi_values: pd.Series, stoch_len: int = 7, k_len: int = 3, d_len: int = 3) -> pd.Series:
        rsi_min = rsi_values.rolling(window=stoch_len).min()
        rsi_max = rsi_values.rolling(window=stoch_len).max()
        stoch = 100 * (rsi_values - rsi_min) / (rsi_max - rsi_min + 1e-10)
        k = stoch.rolling(window=k_len).mean()
        d = k.rolling(window=d_len).mean()
        return ((k + d) / 2).fillna(50)

    def _calculate_obv_signals(self) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        obv = (np.sign(self.df['close'].diff()) * self.df['volume']).fillna(0).cumsum()
        obv_high = obv.rolling(window=18).max()
        obv_low = obv.rolling(window=18).min()

        h1 = obv_high.where(obv_high == obv_high.shift(1)).ffill()
        l1 = obv_low.where(obv_low == obv_low.shift(1)).ffill()

        hlmid = (h1 + l1) / 2

        obv_check = pd.Series(0, index=self.df.index)
        obv_cross_up_h1 = (obv.shift(1) <= h1) & (obv > h1)
        obv_cross_down_l1 = (obv.shift(1) >= l1) & (obv < l1)
        obv_check[obv_cross_up_h1] = 20
        obv_check[obv_cross_down_l1] = -20

        return obv, h1, l1, hlmid, obv_check.replace(0, method='ffill')

    def _calculate_dmi_signals(self) -> pd.Series:
        try:
            plus_di = talib.PLUS_DI(self.df['high'], self.df['low'], self.df['close'], timeperiod=14)
            minus_di = talib.MINUS_DI(self.df['high'], self.df['low'], self.df['close'], timeperiod=14)
            adx = talib.ADX(self.df['high'], self.df['low'], self.df['close'], timeperiod=14)
        except Exception:
            # Fallback calculation
            high_diff = self.df['high'].diff()
            low_diff = -self.df['low'].diff()
            plus_dm = pd.Series(np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0), index=self.df.index)
            minus_dm = pd.Series(np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0), index=self.df.index)
            tr = self._calculate_true_range()
            atr = tr.ewm(span=14, adjust=False).mean()
            plus_di = 100 * plus_dm.ewm(span=14, adjust=False).mean() / (atr + 1e-10)
            minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / (atr + 1e-10)
            adx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)).ewm(span=14, adjust=False).mean()

        adx_signals = pd.Series(0, index=self.df.index)
        adx_signals.loc[(adx > 25) & (adx >= adx.shift(1)) & (plus_di >= minus_di)] = 20
        adx_signals.loc[(adx > 25) & (adx >= adx.shift(1)) & (plus_di < minus_di)] = -20
        adx_signals.loc[(adx < 25) & (adx >= adx.shift(1)) & (plus_di >= minus_di)] = 10
        adx_signals.loc[(adx < 25) & (adx >= adx.shift(1)) & (plus_di < minus_di)] = -10
        return adx_signals

    def _calculate_true_range(self) -> pd.Series:
        high_low = self.df['high'] - self.df['low']
        high_close = np.abs(self.df['high'] - self.df['close'].shift(1))
        low_close = np.abs(self.df['low'] - self.df['close'].shift(1))
        return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).fillna(0)

    def _calculate_momentum_signals(self, period: int = 10) -> Tuple[pd.Series, pd.Series]:
        mom = self.df['close'].diff(period)
        try:
            signal = talib.EMA(mom, timeperiod=9)
        except Exception:
            signal = mom.ewm(span=9, adjust=False).mean()

        mom0 = pd.Series(np.where(mom > 0, 10, -10), index=self.df.index).fillna(0)
        mom1 = pd.Series(np.where(mom > signal, 10, -10), index=self.df.index).fillna(0)

        return mom0.replace(0, method='ffill'), mom1.replace(0, method='ffill')


class IntegratedMathematicalAnalysisSystem:
    def __init__(self, load_existing_models=False):
        # --- CONFIGURATION ---
        self.api_key = "1d1e9478ca4c4c36a1b33b2e008bc4a5"
        self.analyst = "ASJFOJ1"
        self.symbol_config = {
            "name": "BTCUSDT",
            "base_coin": "BTC"
        }
        self.base_url = "https://open-api.coinank.com"
        self.timeframes = ['15m', '30m', '1h', '4h', '6h', '12h', '1d']
        self.limit = 1000
        self.lookback_size = 200
        self.EPSILON = 1e-10

        # Initialize logging
        self.logger = logging.getLogger(__name__)

        # Rate limiter
        self.rate_limiter = RateLimiter(calls_per_second=2.0)

        # Multiprocessing
        self.n_processes = min(cpu_count(), 8)
        self.executor = ThreadPoolExecutor(max_workers=self.n_processes)

        # Use current timestamp
        self.current_time = datetime.strptime("2025-07-23 16:10:35", '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
        self.current_timestamp = int(self.current_time.timestamp() * 1000)

        # Calculate safe end time for API calls (15 minutes ago)
        self.safe_end_time = self.current_timestamp - (15 * 60 * 1000)

        timestamp_str = self.current_time.strftime('%Y%m%d_%H%M%S')
        # Update version to v45 with all fixes
        self.results_file = f"mathematical_analysis_v45_fixed_{timestamp_str}_LATEST_v4.txt"
        self.summary_file = f"mathematical_summary_v45_fixed_{timestamp_str}_LATEST_v4.txt"
        self.error_log_file = f"error_log_v45_fixed_{timestamp_str}_LATEST_v4.txt"

        self._initialize_output_files()

        # Exchange configuration
        self.exchange_groups = {
            "orderflow": ["Binance", "OKX", "Bybit"],
            "large_orders": ["Binance", "OKX"],
            "large_limit": ["Binance", "OKX", "Coinbase"],
            "cvd": ["Binance", "OKX", "Bybit", "Bitget"],
            "liquidation": ["Binance", "OKX", "Bybit"],
            "liquidation_history": ["Binance", "OKX", "Bybit"],
            "market_order": ["Binance", "OKX", "Bybit"],
            "aggregate_swap": ["Binance", "OKX", "Bybit", "Bitget", "Huobi"],
            "aggregate_spot": ["Binance", "OKX", "Bybit", "Coinbase"],
            "longshort": ["Binance", "OKX"],
            "net_positions": ["Binance", "OKX", "Bybit"],
            "open_interest": ["Binance", "OKX", "Bybit"],
            "funding": ["Binance", "OKX", "Bybit"]
        }

        self.best_symbols = {
            "SWAP": {"Binance": "BTCUSDT", "Bybit": "BTCUSDT", "OKX": "BTC-USDT-SWAP",
                     "Coinbase": "BTC-USD", "Bitget": "BTCUSDT", "Huobi": "BTC-USDT"},
            "SPOT": {"Binance": "BTCUSDT", "Bybit": "BTCUSDT", "Coinbase": "BTC-USD",
                     "OKX": "BTC-USDT", "Bitget": "BTCUSDT", "Huobi": "BTCUSDT"}
        }

        self.stats = {
            "total_requests": 0, "successful_requests": 0, "failed_requests": 0,
            "data_points_collected": 0, "sections_completed": []
        }

        # Initialize components
        self.tensor_fusion = MultiModalTensorFusion(self.lookback_size)
        self.info_flow_analyzer = InformationFlowAnalyzer()
        self.microstructure_bridge = MicrostructureKlineBridge()
        self.advanced_tools = AdvancedMathematicalTools()

        # Initialize enhanced components
        self.hawkes_process = HawkesProcess()
        self.ukf_filter = UnscentedKalmanFilter(dim_x=2, dim_z=1)  # Price and velocity
        self.ms_garch = MarkovSwitchingGARCH()
        self.cnn_predictor = CNNMarketPredictor()

        self.collected_data = {}
        self.analysis_results = {}
        self.kline_data = {}

        # Thread safety
        self.data_lock = threading.Lock()
        
        # Load models if requested
        self.load_existing_models = load_existing_models
        if load_existing_models:
            try:
                self.load_models()
            except Exception as e:
                self.logger.warning(f"Could not load models: {e}")
                print(f"⚠ Could not load models: {e}")
                print("  Will train new models during analysis")

    # --- NEW HELPER FUNCTIONS FOR ROBUSTNESS ---
    def safe_divide(self, numerator, denominator, default=0.0):
        """Universal safe division function."""
        if self.is_zero(denominator):
            return default
        return numerator / denominator

    def is_zero(self, value):
        """Use epsilon for float comparisons."""
        return abs(value) < self.EPSILON

    def safe_get_nested(self, data, keys, default=None):
        """Safely get nested dictionary values."""
        for key in keys:
            if isinstance(data, dict):
                data = data.get(key)
            else:
                return default
        return data if data is not None else default

    def align_arrays(self, *arrays):
        """Ensure all arrays have same length by truncating to the shortest."""
        if not arrays:
            return []
        min_len = min((len(arr) for arr in arrays if arr is not None), default=0)
        if min_len == 0:
            return [np.array([]) for _ in arrays]
        return [arr[:min_len] for arr in arrays]

    def check_memory_before_allocation(self, shape, dtype=np.float64, limit_gb=1.0):
        """Check if allocation would exceed memory limits."""
        bytes_needed = np.prod(shape) * np.dtype(dtype).itemsize
        if bytes_needed > limit_gb * 1e9:
            self.logger.error(f"Memory allocation of {bytes_needed/1e9:.2f}GB exceeds limit of {limit_gb}GB.")
            raise MemoryError(f"Allocation of {bytes_needed/1e9:.2f}GB exceeds limit")
        return True

    def validate_dataframe(self, df, required_columns):
        """Ensure DataFrame has required columns."""
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Invalid input: not a pandas DataFrame.")
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")
        return True

    def _safe_resample(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Safely resample dataframe handling non-numeric columns"""
        # Check if index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            self.logger.warning("DataFrame doesn't have DatetimeIndex, cannot resample")
            return df
        
        # Separate numeric and non-numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return pd.DataFrame()
        
        # Resample only numeric columns
        resampled = df[numeric_cols].resample(timeframe).mean()
        return resampled

    def _initialize_output_files(self):
        """Initialize output files"""
        with open(self.results_file, 'w', encoding='utf-8') as f:
            f.write(f"Mathematical Analysis System V45 Fixed & Enhanced v4\n{'='*100}\n")
            f.write(f"Analyst: {self.analyst}\nAnalysis Started: {self.current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC\n")
            f.write(f"Timeframes: {', '.join(self.timeframes)}\n")
            f.write(f"Enhancements: All critical fixes applied, model persistence added\n")
            f.write(f"{'='*100}\n\n")

        with open(self.summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Mathematical Analysis Summary V45 Fixed & Enhanced v4\n{'='*100}\n")
            f.write(f"Generated: {self.current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC\n")
            f.write(f"{'='*100}\n\n")

        with open(self.error_log_file, 'w', encoding='utf-8') as f:
            f.write(f"Error Log V45 Fixed & Enhanced v4\n{'='*80}\n\n")

    def log_error(self, endpoint, params, error_msg, description=""):
        """Log errors to file"""
        with open(self.error_log_file, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now(timezone.utc).strftime('%H:%M:%S')} | DESC: {description} | "
                   f"ENDPOINT: {endpoint} | PARAMS: {json.dumps(params)} | MSG: {error_msg}\n")
        self.logger.error(f"{description}: {error_msg}")

    def validate_timestamp(self, timestamp_ms: int):
        """Check if the requested timestamp is in the future."""
        if timestamp_ms > self.current_timestamp:
            raise ValueError(f"Requesting future data is not allowed: {timestamp_ms} > {self.current_timestamp}")

    def make_request(self, endpoint, params, description="", retries=3, retry_delay=2):
        """
        Make API request with error handling and a retry mechanism - FIXED
        """
        # Use centralized rate limiter
        self.rate_limiter.wait()

        timeout = 60 if endpoint == "/api/liquidation/orders" else 20

        # Validate and adjust timestamps
        if 'endTime' in params:
            try:
                end_time_ms = int(params['endTime'])
                self.validate_timestamp(end_time_ms)
            except (ValueError, TypeError) as e:
                self.log_error(endpoint, params, f"Invalid endTime: {e}", description)
                # Use safe historical timestamp
                params['endTime'] = str(self.safe_end_time)

        for attempt in range(retries):
            self.stats["total_requests"] += 1
            try:
                response = requests.get(
                    f"{self.base_url}{endpoint}",
                    headers={'apikey': self.api_key},
                    params=params,
                    timeout=timeout
                )

                # Retry on server errors (5xx)
                if response.status_code >= 500:
                    response.raise_for_status()

                # Check for successful response
                data = response.json()
                if data.get('success') and str(data.get('code')) == '1':
                    self.stats["successful_requests"] += 1
                    return data
                else:
                    error_msg = f"API Error: {data.get('msg', 'Unknown error')}"

                    # If it's a "system error", treat it as a retryable server issue
                    if "system error" in error_msg.lower():
                        if attempt < retries - 1:
                            self.logger.warning(f"{description}: {error_msg}. Retrying ({attempt+1}/{retries})...")
                            time.sleep(retry_delay * (attempt + 1))
                            continue
                        else:
                            self.logger.error(f"{description}: {error_msg}. Final attempt failed.")

                    # Log non-retryable API errors
                    self.stats["failed_requests"] += 1
                    self.log_error(endpoint, params, error_msg, description)
                    return None

            except requests.exceptions.RequestException as e:
                error_msg = f"RequestException: {str(e)}"
                if attempt < retries - 1:
                    self.logger.warning(f"{description}: {error_msg}. Retrying ({attempt+1}/{retries})...")
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    self.stats["failed_requests"] += 1
                    self.log_error(endpoint, params, error_msg, description)
                    self.logger.error(f"{description}: {error_msg}. Final attempt failed.")
                    return None

        return None

    def fetch_klines_concurrent(self):
        """Fetch kline data for all timeframes concurrently"""
        self.logger.info("Starting concurrent kline data fetch")
        print("\n[FETCHING KLINE DATA - CONCURRENT]")
        url = "https://api.binance.com/api/v3/klines"

        def fetch_single_timeframe(interval):
            params = {
                'symbol': self.symbol_config['name'],
                'interval': interval,
                'limit': self.limit
            }

            try:
                self.logger.info(f"Fetching {interval} kline data")
                response = requests.get(url, params=params, timeout=30)

                if response.status_code != 200:
                    self.logger.error(f"Error fetching {interval}: {response.text}")
                    return interval, None

                data = response.json()
                self.logger.info(f"Received {len(data)} candles for {interval}")

                if not data:
                    return interval, None

                # Convert to DataFrame
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])

                # Convert to numeric and set index
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.set_index('timestamp')
                for col in ['close', 'high', 'low', 'open', 'volume', 'taker_buy_base', 'taker_buy_quote']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                return interval, df

            except Exception as e:
                self.logger.exception(f"Error fetching {interval}")
                return interval, None

        # Fetch all timeframes concurrently
        with ThreadPoolExecutor(max_workers=len(self.timeframes)) as executor:
            futures = {executor.submit(fetch_single_timeframe, tf): tf
                      for tf in self.timeframes}

            for future in as_completed(futures):
                try:
                    interval, df = future.result()
                    if df is not None and not df.empty:
                        with self.data_lock:
                            self.kline_data[interval] = df
                except Exception as e:
                    self.logger.exception(f"Error processing future")

        self.logger.info(f"Successfully fetched kline data for: {list(self.kline_data.keys())}")
        print(f"\nSuccessfully fetched data for: {list(self.kline_data.keys())}")

    def collect_all_market_data_by_endpoint(self):
        """
        Collect market data by iterating through endpoints first, then timeframes - FIXED
        """
        print("\n[COLLECTING MARKET DATA - BY ENDPOINT]")
        self.logger.info("Starting market data collection by endpoint")

        # Define all data collection tasks
        tasks_to_run = [
            (self.collect_order_flow_data, "Order Flow"),
            (self.collect_large_order_data, "Large Orders"),
            (self.collect_orderbook_data, "Orderbook"),
            (self.collect_open_interest_data, "Open Interest"),
            (self.collect_positioning_data, "Positioning"),
            (self.collect_liquidation_data, "Liquidations"),
            (self.collect_funding_data, "Funding Rates"),
            (self.collect_fund_flow_data, "Fund Flow"),
            (self.collect_cvd_data, "CVD"),
            (self.collect_market_order_data, "Market Orders"),
            (self.collect_net_positions_data, "Net Positions")
        ]

        # Iterate through each data type (endpoint group)
        for func, name in tasks_to_run:
            print(f"\n--- Collecting {name} Data ---")
            self.logger.info(f"Collecting {name} data")

            # Create a list of futures for all timeframes for the current data type
            with ThreadPoolExecutor(max_workers=self.n_processes) as executor:
                futures = {
                    executor.submit(func, timeframe=tf): tf
                    for tf in self.timeframes
                }

                # Process results as they complete
                for future in as_completed(futures):
                    try:
                        future.result(timeout=180)
                    except Exception as e:
                        timeframe = futures[future]
                        self.logger.exception(f"Failed {name} for {timeframe}")
                        self.log_error(name, {'timeframe': timeframe}, str(e), f"Data collection for {name}")

        print(f"\nTotal data points collected: {self.stats['data_points_collected']:,}")
        self.logger.info(f"Total data points collected: {self.stats['data_points_collected']:,}")

    def collect_order_flow_data(self, timeframe='1h', api_interval='1h'):
        """Collect order flow data - FIXED with safe timestamps"""
        order_flow_data = []

        for product_type in ["SWAP", "SPOT"]:
            for exchange in self.exchange_groups["orderflow"][:2]:
                for tick_count in [1, 5]:
                    symbol = self.best_symbols[product_type].get(exchange, "BTCUSDT")
                    params = {
                        'exchange': exchange,
                        'symbol': symbol,
                        'interval': timeframe,
                        'endTime': str(self.safe_end_time),  # Use safe timestamp
                        'size': '300',
                        'productType': product_type,
                        'tickCount': tick_count
                    }

                    data = self.make_request("/api/orderFlow/lists", params,
                                           f"{exchange} {product_type} Order Flow {timeframe}")
                    if data and data.get('data'):
                        for record in data.get('data', []):
                            if isinstance(record, dict):
                                order_flow_data.append({
                                    'exchange': record.get('exchangeName', exchange),
                                    'symbol': record.get('symbol', symbol),
                                    'timestamp': record.get('ts'),
                                    'step': record.get('step'),
                                    'prices': record.get('prices', []),
                                    'total_ask': sum(record.get('asks', [])),
                                    'total_bid': sum(record.get('bids', [])),
                                    'asks': record.get('asks', []),
                                    'bids': record.get('bids', []),
                                    'tick_count': tick_count,
                                    'timeframe': timeframe
                                })
                                self.stats["data_points_collected"] += 1

        with self.data_lock:
            if timeframe not in self.collected_data:
                self.collected_data[timeframe] = {}
            self.collected_data[timeframe]['order_flow'] = order_flow_data

    def collect_cvd_data(self, timeframe='1h', api_interval='1h'):
        """Collect Cumulative Volume Delta data - FIXED"""
        cvd_data = []

        for product_type in ["SWAP", "SPOT"]:
            # Get individual exchange CVD data with proper structure
            for exchange in self.exchange_groups["cvd"][:2]:
                symbol = self.best_symbols[product_type].get(exchange, "BTCUSDT")
                params = {
                    'exchange': exchange,
                    'symbol': symbol,
                    'interval': timeframe,
                    'endTime': str(self.safe_end_time),
                    'size': '300',
                    'productType': product_type,
                    'type': 'CVD'
                }

                data = self.make_request("/api/cvd/getCvdKline", params,
                                       f"{exchange} {product_type} CVD {timeframe}")
                if data and data.get('data'):
                    for record in data.get('data', []):
                        if isinstance(record, list) and len(record) >= 4:
                            cvd_data.append({
                                'exchange': exchange,
                                'timestamp': record[0],
                                'cvd_high': float(record[1]),
                                'cvd_low': float(record[2]),
                                'cvd_close': float(record[3]),
                                'timeframe': timeframe
                            })
                            self.stats["data_points_collected"] += 1

            # Get Aggregated CVD with proper structure
            params = {
                'baseCoin': self.symbol_config['base_coin'],
                'interval': timeframe,
                'endTime': str(self.safe_end_time),
                'size': '300',
                'productType': product_type,
                'type': 'CVD'
            }

            data = self.make_request("/api/cvd/getAggCvdKline", params,
                                   f"Aggregated {product_type} CVD {timeframe}")
            if data and data.get('data'):
                for record in data.get('data', []):
                    if isinstance(record, list) and len(record) >= 4:
                        cvd_data.append({
                            'exchange': 'AGGREGATE',
                            'timestamp': record[0],
                            'cvd_high': float(record[1]),
                            'cvd_low': float(record[2]),
                            'cvd_close': float(record[3]),
                            'timeframe': timeframe
                        })
                        self.stats["data_points_collected"] += 1

            # Add new CVD endpoint
            for exchange in self.exchange_groups["cvd"][:2]:
                symbol = self.best_symbols[product_type].get(exchange, "BTCUSDT")
                params = {
                    'exchange': exchange,
                    'symbol': symbol,
                    'interval': timeframe,
                    'endTime': str(self.safe_end_time),
                    'size': '300',
                    'productType': product_type
                }

                data = self.make_request("/api/marketOrder/getCvd", params,
                                       f"{exchange} {product_type} Market CVD {timeframe}")
                if data and data.get('data') and data['data'].get('data'):
                    for record in data['data']['data']:
                        if isinstance(record, list) and len(record) >= 4:
                            cvd_data.append({
                                'exchange': f"{exchange}_MARKET",
                                'timestamp': record[0],
                                'cvd_high': float(record[1]),
                                'cvd_low': float(record[2]),
                                'cvd_close': float(record[3]),
                                'timeframe': timeframe
                            })
                            self.stats["data_points_collected"] += 1

        with self.data_lock:
            if timeframe not in self.collected_data:
                self.collected_data[timeframe] = {}
            self.collected_data[timeframe]['cvd'] = cvd_data

    def collect_market_order_data(self, timeframe='1h', api_interval='1h'):
        """
        Collect market order metrics for both SWAP and SPOT - FIXED
        """
        market_order_data = []

        for product_type in ["SWAP", "SPOT"]:
            # --- Per-Exchange Data Collection ---
            for exchange in self.exchange_groups["market_order"][:2]:
                symbol = self.best_symbols[product_type].get(exchange, "BTCUSDT")
                params = {
                    'exchange': exchange,
                    'symbol': symbol,
                    'interval': timeframe,
                    'endTime': str(self.safe_end_time),
                    'size': '300',
                    'productType': product_type
                }

                # Get buy/sell volume (per-exchange)
                data_vol = self.make_request("/api/marketOrder/getBuySellVolume", params,
                                             f"{exchange} {product_type} Buy/Sell Volume {timeframe}")
                if data_vol and data_vol.get('data'):
                    for record in data_vol.get('data', []):
                        if isinstance(record, dict):
                            market_order_data.append({
                                'exchange': exchange,
                                'timestamp': record.get('begin'),
                                'buy_volume': float(record.get('buyVolume', 0)),
                                'sell_volume': float(record.get('sellVolume', 0)),
                                'net_volume': float(record.get('buyVolume', 0)) - float(record.get('sellVolume', 0)),
                                'metric_type': 'volume',
                                'timeframe': timeframe
                            })
                            self.stats["data_points_collected"] += 1

                # Get buy/sell value (per-exchange) - NEW
                data_val = self.make_request("/api/marketOrder/getBuySellValue", params,
                                             f"{exchange} {product_type} Buy/Sell Value {timeframe}")
                if data_val and data_val.get('data') and data_val['data'].get('data'):
                    for record in data_val['data']['data']:
                        if isinstance(record, list) and len(record) >= 3:
                            market_order_data.append({
                                'exchange': exchange,
                                'timestamp': record[0],
                                'buy_value': float(record[1]),
                                'sell_value': float(record[2]),
                                'net_value': float(record[1]) - float(record[2]),
                                'metric_type': 'value',
                                'timeframe': timeframe
                            })
                            self.stats["data_points_collected"] += 1

                # Get buy/sell count (per-exchange)
                data_count = self.make_request("/api/marketOrder/getBuySellCount", params,
                                               f"{exchange} {product_type} Buy/Sell Count {timeframe}")
                if data_count and data_count.get('data'):
                    for record in data_count.get('data', []):
                        if isinstance(record, dict):
                            market_order_data.append({
                                'exchange': exchange,
                                'timestamp': record.get('begin'),
                                'buy_count': int(record.get('buyCount', 0)),
                                'sell_count': int(record.get('sellCount', 0)),
                                'net_count': int(record.get('buyCount', 0)) - int(record.get('sellCount', 0)),
                                'metric_type': 'count',
                                'timeframe': timeframe
                            })
                            self.stats["data_points_collected"] += 1

            # --- Aggregated Data Collection (Called Once per product type) ---
            agg_params = {
                'baseCoin': self.symbol_config['base_coin'],
                'exchanges': ','.join(self.exchange_groups["market_order"][:2]),
                'interval': timeframe,
                'endTime': str(self.safe_end_time),
                'size': '300',
                'productType': product_type
            }

            # Get aggregated buy/sell value
            data_agg_val = self.make_request("/api/marketOrder/getAggBuySellValue", agg_params,
                                         f"Aggregated {product_type} Buy/Sell Value {timeframe}")
            if data_agg_val and data_agg_val.get('data') and data_agg_val['data'].get('data'):
                for record in data_agg_val['data']['data']:
                    if isinstance(record, list) and len(record) >= 3:
                        market_order_data.append({
                            'exchange': 'AGGREGATE',
                            'timestamp': record[0],
                            'buy_value': float(record[1]),
                            'sell_value': float(record[2]),
                            'net_value': float(record[1]) - float(record[2]),
                            'metric_type': 'agg_value',
                            'timeframe': timeframe
                        })
                        self.stats["data_points_collected"] += 1

            # Get aggregated buy/sell volume - NEW
            data_agg_vol = self.make_request("/api/marketOrder/getAggBuySellVolume", agg_params,
                                             f"Aggregated {product_type} Buy/Sell Volume {timeframe}")
            if data_agg_vol and data_agg_vol.get('data') and data_agg_vol['data'].get('data'):
                for record in data_agg_vol['data']['data']:
                    if isinstance(record, list) and len(record) >= 3:
                        market_order_data.append({
                            'exchange': 'AGGREGATE',
                            'timestamp': record[0],
                            'buy_volume': float(record[1]),
                            'sell_volume': float(record[2]),
                            'net_volume': float(record[1]) - float(record[2]),
                            'metric_type': 'agg_volume',
                            'timeframe': timeframe
                        })
                        self.stats["data_points_collected"] += 1

        with self.data_lock:
            if timeframe not in self.collected_data:
                self.collected_data[timeframe] = {}
            self.collected_data[timeframe]['market_orders'] = market_order_data

    def collect_large_order_data(self, timeframe='1h', api_interval='1h'):
        """Collect large order data for specific timeframe - FIXED"""
        large_order_data = []

        # Regular large trades first
        thresholds = ['1000000', '5000000']
        for product_type in ["SWAP", "SPOT"]:
            for exchange in self.exchange_groups["large_orders"][:2]:
                symbol = self.best_symbols[product_type].get(exchange, "BTCUSDT")
                for threshold in thresholds:
                    params = {
                        'exchange': exchange,
                        'symbol': symbol,
                        'productType': product_type,
                        'amount': threshold,
                        'size': '300',
                        'endTime': str(self.safe_end_time)
                    }

                    data = self.make_request("/api/trades/largeTrades", params,
                                           f"{exchange} {product_type} Large Trades {timeframe}")
                    if data and data.get('data'):
                        for record in data.get('data', []):
                            if isinstance(record, dict):
                                large_order_data.append({
                                    'exchange': exchange,
                                    'product_type': product_type,
                                    'timestamp': record.get('ts'),
                                    'side': record.get('side'),
                                    'price': float(record.get('price', 0)),
                                    'amount': float(record.get('amount', 0)),
                                    'turnover': float(record.get('tradeTurnover', 0)),
                                    'threshold': threshold,
                                    'order_type': 'large_trade',
                                    'timeframe': timeframe
                                })
                                self.stats["data_points_collected"] += 1

        # FIXED: Big Order Query List implementation
        exchanges_to_query = ["Binance", "OKX", "Coinbase"]
        exchange_types = ["SWAP", "SPOT", "FUTURES"]
        sides = ['ask', 'bid']

        for exchange in exchanges_to_query:
            if exchange not in ["Binance", "OKX", "Coinbase"]:
                continue

            for exchange_type in exchange_types:
                # Get appropriate symbol for this exchange/type combination
                if exchange_type in ["SWAP", "FUTURES"]:
                    symbol = self.best_symbols.get("SWAP", {}).get(exchange, "BTCUSDT")
                else:  # SPOT
                    symbol = self.best_symbols.get("SPOT", {}).get(exchange, "BTCUSDT")

                # Adjust symbol format for different exchanges
                if exchange == "OKX" and exchange_type == "SWAP":
                    symbol = "BTC-USDT-SWAP"
                elif exchange == "OKX" and exchange_type == "SPOT":
                    symbol = "BTC-USDT"
                elif exchange == "Coinbase":
                    symbol = "BTC-USD"

                for side in sides:
                    params = {
                        'exchange': exchange,
                        'symbol': symbol,
                        'exchangeType': exchange_type,
                        'amount': '10000000',  # 10M minimum
                        'side': side,
                        'isHistory': 'true',
                        'startTime': str(self.safe_end_time),  # Returns data BEFORE this time
                        'size': '500'
                    }

                    data = self.make_request("/api/bigOrder/queryOrderList", params,
                                           f"{exchange} {exchange_type} Big {side} Orders {timeframe}")

                    if data and data.get('data'):
                        for record in data.get('data', []):
                            if isinstance(record, dict):
                                large_order_data.append({
                                    'exchange': record.get('exchangeName', exchange),
                                    'product_type': record.get('exchangeType', exchange_type),
                                    'timestamp': record.get('openTime'),
                                    'close_time': record.get('closeTime'),
                                    'side': record.get('side', side),
                                    'price': float(record.get('price', 0)),
                                    'current_amount': float(record.get('entrustAmount', 0)),
                                    'current_turnover': float(record.get('entrustTurnover', 0)),
                                    'initial_amount': float(record.get('firstAmount', 0)),
                                    'initial_turnover': float(record.get('firstTurnover', 0)),
                                    'filled_amount': float(record.get('turnoverNumber', 0)),
                                    'filled_turnover': float(record.get('turnoverAmount', 0)),
                                    'trade_count': int(record.get('count', 0)),
                                    'is_active': record.get('on', False),
                                    'order_type': 'big_limit_order',
                                    'timeframe': timeframe
                                })
                                self.stats["data_points_collected"] += 1

        with self.data_lock:
            if timeframe not in self.collected_data:
                self.collected_data[timeframe] = {}
            self.collected_data[timeframe]['large_orders'] = large_order_data

    def collect_orderbook_data(self, timeframe='1h', api_interval='1h'):
        """
        Collect orderbook data for specific timeframe - FIXED
        """
        orderbook_data = []

        # --- Orderbook by Symbol ---
        rates = ['0.01', '0.05']
        for product_type in ["SWAP", "SPOT"]:
            exchanges = self.exchange_groups.get(f'aggregate_{product_type.lower()}', [])[:2]

            for exchange in exchanges:
                symbol = self.best_symbols[product_type].get(exchange, "BTCUSDT")
                for rate in rates:
                    params = {
                        'symbol': symbol,
                        'exchange': exchange,
                        'rate': rate,
                        'productType': product_type,
                        'interval': timeframe,
                        'endTime': str(self.safe_end_time),
                        'size': '300'
                    }
                    data = self.make_request("/api/orderBook/v2/bySymbol", params,
                                           f"{exchange} {product_type} OrderBook {timeframe} (Rate: {rate})")
                    if data and data.get('data'):
                        for record in data.get('data', []):
                            if isinstance(record, list) and len(record) >= 5:
                                buy_usd, sell_usd = float(record[1]), float(record[3])
                                imbalance = self.safe_divide(buy_usd - sell_usd, buy_usd + sell_usd)
                                orderbook_data.append({
                                    'exchange': exchange, 'product_type': product_type, 'timestamp': record[0],
                                    'rate': rate, 'buy_usd': buy_usd, 'sell_usd': sell_usd,
                                    'imbalance': imbalance, 'timeframe': timeframe
                                })
                                self.stats["data_points_collected"] += 1

            # --- Orderbook by Exchange ---
            book_types = ['0.0025', '0.005', '0.01', '0.05']
            for book_type in book_types:
                params = {
                    'baseCoin': self.symbol_config['base_coin'],
                    'productType': product_type,
                    'interval': timeframe,
                    'endTime': str(self.safe_end_time),
                    'size': '300',
                    'exchanges': ','.join(exchanges),
                    'type': book_type
                }
                data = self.make_request("/api/orderBook/v2/byExchange", params,
                                       f"By Exchange {product_type} OrderBook {timeframe} (Type: {book_type})")

                if data and data.get('data'):
                    for record in data.get('data', []):
                        if isinstance(record, list) and len(record) >= 4:
                            buy_usd, sell_usd = float(record[1]), float(record[3])
                            imbalance = self.safe_divide(buy_usd - sell_usd, buy_usd + sell_usd)
                            orderbook_data.append({
                                'exchange': 'AGGREGATE', 'product_type': product_type, 'timestamp': record[0],
                                'rate': book_type, 'buy_usd': buy_usd, 'sell_usd': sell_usd,
                                'imbalance': imbalance, 'timeframe': timeframe
                            })
                            self.stats["data_points_collected"] += 1

        with self.data_lock:
            if timeframe not in self.collected_data:
                self.collected_data[timeframe] = {}
            self.collected_data[timeframe]['orderbook'] = orderbook_data

    def collect_open_interest_data(self, timeframe='1h', api_interval='1h'):
        """Collect open interest data for specific timeframe - FIXED"""
        oi_data = []

        # Aggregated OI
        params = {
            'baseCoin': self.symbol_config['base_coin'],
            'interval': timeframe,
            'endTime': str(self.safe_end_time),
            'size': '300'
        }

        data = self.make_request("/api/openInterest/aggKline", params, f"Aggregated OI {timeframe}")
        if data and data.get('data'):
            prev_oi = None
            for record in data.get('data', []):
                if isinstance(record, dict):
                    curr_oi = float(record.get('close', 0))
                    oi_change = curr_oi - prev_oi if prev_oi is not None else 0
                    change_pct = self.safe_divide(oi_change, prev_oi) * 100 if prev_oi else 0

                    oi_data.append({
                        'type': 'aggregated',
                        'timestamp': record.get('begin'),
                        'oi_value': curr_oi,
                        'oi_change': oi_change,
                        'change_pct': change_pct,
                        'timeframe': timeframe
                    })
                    self.stats["data_points_collected"] += 1
                    prev_oi = curr_oi

        # Individual Exchange OI
        for product_type in ["SWAP"]:
            for exchange in self.exchange_groups["open_interest"][:2]:
                symbol = self.best_symbols[product_type].get(exchange, "BTCUSDT")
                params = {
                    'exchange': exchange,
                    'symbol': symbol,
                    'interval': timeframe,
                    'endTime': str(self.safe_end_time),
                    'size': '300',
                    'productType': product_type
                }

                data = self.make_request("/api/openInterest/kline", params, f"{exchange} OI {timeframe}")
                if data and data.get('data'):
                    prev_oi = None
                    for record in data.get('data', []):
                        if isinstance(record, dict):
                            curr_oi = float(record.get('close', 0))
                            oi_change = curr_oi - prev_oi if prev_oi is not None else 0
                            change_pct = self.safe_divide(oi_change, prev_oi) * 100 if prev_oi else 0

                            oi_data.append({
                                'type': exchange,
                                'timestamp': record.get('begin'),
                                'oi_value': curr_oi,
                                'oi_change': oi_change,
                                'change_pct': change_pct,
                                'timeframe': timeframe
                            })
                            self.stats["data_points_collected"] += 1
                            prev_oi = curr_oi

        with self.data_lock:
            if timeframe not in self.collected_data:
                self.collected_data[timeframe] = {}
            self.collected_data[timeframe]['open_interest'] = oi_data

    def collect_positioning_data(self, timeframe='1h', api_interval='1h'):
        """Collect positioning data for specific timeframe - FIXED"""
        positioning_data = []

        for exchange in self.exchange_groups["longshort"][:1]:
            symbol = self.best_symbols["SWAP"].get(exchange, "BTCUSDT")

            # Global Account Ratio
            params = {
                'exchange': exchange,
                'symbol': symbol,
                'interval': timeframe,
                'endTime': str(self.safe_end_time),
                'size': '300'
            }

            data = self.make_request("/api/longshort/person", params, f"{exchange} Account Ratio {timeframe}")
            if data and data.get('data') and isinstance(data['data'], dict):
                tss = data['data'].get('tss', [])
                ratios = data['data'].get('longShortRatio', [])

                for ts, ratio in zip(tss, ratios):
                    positioning_data.append({
                        'exchange': exchange,
                        'type': 'global_account',
                        'timestamp': ts,
                        'ls_ratio': ratio,
                        'timeframe': timeframe
                    })
                    self.stats["data_points_collected"] += 1

            # Position Ratio
            data = self.make_request("/api/longshort/position", params, f"{exchange} Position Ratio {timeframe}")
            if data and data.get('data') and isinstance(data['data'], dict):
                tss = data['data'].get('tss', [])
                ratios = data['data'].get('longShortRatio', [])

                for ts, ratio in zip(tss, ratios):
                    positioning_data.append({
                        'exchange': exchange,
                        'type': 'position',
                        'timestamp': ts,
                        'ls_ratio': ratio,
                        'timeframe': timeframe
                    })
                    self.stats["data_points_collected"] += 1

            # Top Trader Account Ratio
            data = self.make_request("/api/longshort/account", params, f"{exchange} Top Trader Ratio {timeframe}")
            if data and data.get('data') and isinstance(data['data'], dict):
                tss = data['data'].get('tss', [])
                ratios = data['data'].get('longShortRatio', [])

                for ts, ratio in zip(tss, ratios):
                    positioning_data.append({
                        'exchange': exchange,
                        'type': 'top_trader',
                        'timestamp': ts,
                        'ls_ratio': ratio,
                        'timeframe': timeframe
                    })
                    self.stats["data_points_collected"] += 1

        with self.data_lock:
            if timeframe not in self.collected_data:
                self.collected_data[timeframe] = {}
            self.collected_data[timeframe]['positioning'] = positioning_data

    def collect_net_positions_data(self, timeframe='1h', api_interval='1h'):
        """Collect net positions data - FIXED"""
        net_positions_data = []

        for exchange in self.exchange_groups["net_positions"][:2]:
            symbol = self.best_symbols["SWAP"].get(exchange, "BTCUSDT")
            params = {
                'exchange': exchange,
                'symbol': symbol,
                'interval': timeframe,
                'endTime': str(self.safe_end_time),
                'size': '300'
            }

            data = self.make_request("/api/netPositions/getNetPositions", params,
                                   f"{exchange} Net Positions {timeframe}")
            if data and data.get('data'):
                for record in data.get('data', []):
                    if isinstance(record, dict):
                        net_positions_data.append({
                            'exchange': exchange,
                            'timestamp': record.get('begin'),
                            'net_longs_high': float(record.get('netLongsHigh', 0)),
                            'net_longs_close': float(record.get('netLongsClose', 0)),
                            'net_longs_low': float(record.get('netLongsLow', 0)),
                            'net_shorts_high': float(record.get('netShortsHigh', 0)),
                            'net_shorts_close': float(record.get('netShortsClose', 0)),
                            'net_shorts_low': float(record.get('netShortsLow', 0)),
                            'net_position': float(record.get('netLongsClose', 0)) - float(record.get('netShortsClose', 0)),
                            'timeframe': timeframe
                        })
                        self.stats["data_points_collected"] += 1

        with self.data_lock:
            if timeframe not in self.collected_data:
                self.collected_data[timeframe] = {}
            self.collected_data[timeframe]['net_positions'] = net_positions_data

    def collect_liquidation_data(self, timeframe='1h', api_interval='1h'):
        """Collect liquidation data for specific timeframe - FIXED"""
        liquidation_data = []

        # All Exchange liquidation intervals
        params = {
            'baseCoin': self.symbol_config['base_coin'],
            'interval': timeframe,
            'endTime': str(self.safe_end_time),
            'size': '300'
        }

        data = self.make_request("/api/liquidation/allExchange/intervals", params,
                               f"All Exchange Liquidation {timeframe}")
        if data and data.get('data'):
            for record in data.get('data', []):
                if isinstance(record, dict):
                    long_val = float(record.get('longTurnover', 0))
                    short_val = float(record.get('shortTurnover', 0))
                    total_val = long_val + short_val

                    liquidation_data.append({
                        'timestamp': record.get('ts'),
                        'long_value': long_val,
                        'short_value': short_val,
                        'total_value': total_val,
                        'long_count': int(record.get('longCount', 0)),
                        'short_count': int(record.get('shortCount', 0)),
                        'type': 'all_exchange',
                        'timeframe': timeframe
                    })
                    self.stats["data_points_collected"] += 1

        # Aggregated liquidation history
        data = self.make_request("/api/liquidation/aggregated-history", params,
                               f"Aggregated Liquidation {timeframe}")
        if data and data.get('data'):
            for record in data.get('data', []):
                if isinstance(record, dict) and 'all' in record:
                    all_data = record['all']
                    long_val = float(all_data.get('longTurnover', 0))
                    short_val = float(all_data.get('shortTurnover', 0))
                    total_val = long_val + short_val
                    ls_ratio = self.safe_divide(long_val, short_val)

                    liquidation_data.append({
                        'timestamp': record.get('ts'),
                        'long_value': long_val,
                        'short_value': short_val,
                        'total_value': total_val,
                        'ls_ratio': ls_ratio,
                        'type': 'aggregated',
                        'timeframe': timeframe
                    })
                    self.stats["data_points_collected"] += 1

        # Individual exchange liquidation history
        for product_type in ["SWAP", "SPOT"]:
            for exchange in self.exchange_groups["liquidation_history"][:1]:
                symbol = self.best_symbols[product_type].get(exchange, "BTCUSDT")
                params = {
                    'exchange': exchange,
                    'symbol': symbol,
                    'interval': timeframe,
                    'endTime': str(self.safe_end_time),
                    'size': '300',
                    'productType': product_type
                }

                data = self.make_request("/api/liquidation/history", params,
                                       f"{exchange} {product_type} Liquidation History {timeframe}")
                if data and data.get('data'):
                    for record in data.get('data', []):
                        if isinstance(record, dict):
                            long_val = float(record.get('longTurnover', 0))
                            short_val = float(record.get('shortTurnover', 0))

                            liquidation_data.append({
                                'timestamp': record.get('ts'),
                                'exchange': exchange,
                                'long_value': long_val,
                                'short_value': short_val,
                                'total_value': long_val + short_val,
                                'type': 'exchange_specific',
                                'timeframe': timeframe
                            })
                            self.stats["data_points_collected"] += 1

            # Recent liquidation orders (real-time)
            for exchange in self.exchange_groups["liquidation"][:1]:
                symbol = self.best_symbols[product_type].get(exchange, "BTCUSDT")
                params = {
                    'exchange': exchange,
                    'symbol': symbol,
                    'size': '300',
                    'productType': product_type
                }

                data = self.make_request("/api/liquidation/orders", params,
                                       f"{exchange} {product_type} Liquidation Orders {timeframe}")
                if data and data.get('data'):
                    for record in data.get('data', []):
                        if isinstance(record, dict):
                            liquidation_data.append({
                                'timestamp': record.get('ts'),
                                'exchange': exchange,
                                'side': record.get('side'),
                                'price': float(record.get('price', 0)),
                                'amount': float(record.get('liqAmount', 0)),
                                'type': 'order',
                                'timeframe': timeframe
                            })
                            self.stats["data_points_collected"] += 1

        with self.data_lock:
            if timeframe not in self.collected_data:
                self.collected_data[timeframe] = {}
            self.collected_data[timeframe]['liquidation'] = liquidation_data

    def collect_funding_data(self, timeframe='1h', api_interval='1h'):
        """Collect funding rate data for specific timeframe - FIXED"""
        funding_data = []

        # Aggregated funding history
        params = {
            'baseCoin': self.symbol_config['base_coin'],
            'exchangeType': 'USDT',
            'endTime': str(self.safe_end_time),
            'size': '300',
        }
        data = self.make_request("/api/fundingRate/hist", params,
                               f"Aggregated Funding History {timeframe}")
        if data and data.get('data'):
            for record in data.get('data', []):
                if isinstance(record, dict) and 'details' in record:
                    ts = record.get('ts')
                    for exchange, details in record.get('details', {}).items():
                        if isinstance(details, dict) and 'fundingRate' in details:
                            funding_data.append({
                                'exchange': exchange,
                                'timestamp': ts,
                                'rate': float(details['fundingRate']),
                                'type': 'aggregated_hist',
                                'timeframe': timeframe
                            })
                            self.stats["data_points_collected"] += 1

        # Per-pair history from /api/fundingRate/indicator
        for exchange in self.exchange_groups["funding"][:2]:
            symbol = self.best_symbols["SWAP"].get(exchange, "BTCUSDT")
            params = {
                'exchange': exchange,
                'symbol': symbol,
                'interval': timeframe,
                'endTime': str(self.safe_end_time),
                'size': '300',
            }
            data = self.make_request("/api/fundingRate/indicator", params,
                                   f"{exchange} Funding History {timeframe}")
            if data and data.get('data'):
                for record in data.get('data', []):
                    if isinstance(record, dict):
                        funding_data.append({
                            'exchange': exchange,
                            'timestamp': record.get('ts'),
                            'rate': float(record.get('fundingRate', 0)),
                            'type': 'historical',
                            'timeframe': timeframe
                        })
                        self.stats["data_points_collected"] += 1

        # Weighted funding rates (existing endpoint)
        params = {
            'baseCoin': self.symbol_config['base_coin'],
            'interval': timeframe,
            'endTime': str(self.safe_end_time),
            'size': '300'
        }
        data = self.make_request("/api/fundingRate/getWeiFr", params,
                               f"Weighted Funding {timeframe}")
        if data and data.get('data'):
            prev_rate = None
            for record in data.get('data', []):
                if isinstance(record, dict):
                    curr_rate = float(record.get('openFundingRate', 0))
                    turnover_rate = float(record.get('turnoverFundingRate', 0))
                    trend = ("INCREASING" if prev_rate is not None and curr_rate > prev_rate
                            else "DECREASING" if prev_rate is not None and curr_rate < prev_rate
                            else "STABLE")

                    funding_data.append({
                        'timestamp': record.get('ts'),
                        'oi_weighted_rate': curr_rate,
                        'turnover_weighted_rate': turnover_rate,
                        'trend': trend,
                        'type': 'weighted',
                        'timeframe': timeframe
                    })
                    self.stats["data_points_collected"] += 1
                    prev_rate = curr_rate

        # ADD: New endpoint /api/fundingRate/frHeatmap
        for heatmap_type in ['openInterest', 'marketCap']:
            params = {
                'type': heatmap_type,
                'interval': '1M'
            }
            data = self.make_request("/api/fundingRate/frHeatmap", params,
                                   f"Funding Heatmap ({heatmap_type}) {timeframe}")
            if data and data.get('data'):
                if isinstance(data['data'], list):
                    for record in data['data']:
                        record['type'] = f'heatmap_{heatmap_type}'
                        record['timeframe'] = timeframe
                        funding_data.append(record)
                        self.stats["data_points_collected"] += 1

        with self.data_lock:
            if timeframe not in self.collected_data:
                self.collected_data[timeframe] = {}
            self.collected_data[timeframe]['funding'] = funding_data

    def collect_fund_flow_data(self, timeframe='1h', api_interval='h'):
        """Collect fund flow data - FIXED"""
        fund_flow_data = []

        # Map timeframe to API interval correctly
        interval_mapping = {
            '15m': '15m',
            '30m': '30m', 
            '1h': '1h',
            '4h': '4h',
            '6h': '6h',
            '12h': '12h',
            '1d': '1d'
        }
        
        # Use the exact interval from mapping
        api_interval = interval_mapping.get(timeframe, '1h')

        # Fund Real-time Flow - Check if this needs different params
        params = {
            'productType': 'SWAP',
            'sortBy': 'h1net',
            'sortType': 'desc',
            'size': '10',
            'page': '1'
        }

        data = self.make_request("/api/fund/fundReal", params, f"Real-time Fund Flow {timeframe}")
        if data and data.get('data') and data['data'].get('list'):
            for record in data['data']['list']:
                if isinstance(record, dict) and record.get('baseCoin') == self.symbol_config['base_coin']:
                    fund_flow_data.append({
                        'timestamp': record.get('lastUpdateTime', self.current_timestamp),
                        'product_type': record.get('productType', 'SWAP'),
                        'm5_net': float(record.get('m5net', 0)),
                        'h1_net': float(record.get('h1net', 0)),
                        'h4_net': float(record.get('h4net', 0)),
                        'd1_net': float(record.get('d1net', 0)),
                        'type': 'realtime',
                        'timeframe': timeframe
                    })
                    self.stats["data_points_collected"] += 1

        # Historical fund flow - CORRECT PARAMETERS (no exchange param)
        for product_type in ["SWAP", "SPOT"]:
            params = {
                'baseCoin': self.symbol_config['base_coin'],
                'endTime': str(self.safe_end_time),
                'productType': product_type,
                'size': '300',
                'interval': api_interval  # Use the mapped interval
            }

            data = self.make_request("/api/fund/getFundHisList", params,
                                   f"Fund Flow {product_type} {timeframe}")
            if data and data.get('data'):
                for record in data.get('data', []):
                    if isinstance(record, dict):
                        m5 = float(record.get('m5net', 0))
                        h1 = float(record.get('h1net', 0))
                        h4 = float(record.get('h4net', 0))
                        d1 = float(record.get('d1net', 0))

                        signs = sum([1 if x > 0 else -1 if x < 0 else 0
                                   for x in [m5, h1, h4, d1]])
                        momentum = ("STRONG" if signs >= 3 else
                                  "WEAK" if abs(signs) <= 1 else "MODERATE")

                        fund_flow_data.append({
                            'product_type': product_type,
                            'timestamp': record.get('ts'),
                            'm5_net': m5,
                            'h1_net': h1,
                            'h4_net': h4,
                            'd1_net': d1,
                            'momentum': momentum,
                            'composite_score': (m5 * 0.1 + h1 * 0.2 + h4 * 0.3 + d1 * 0.4),
                            'type': 'historical',
                            'timeframe': timeframe
                        })
                        self.stats["data_points_collected"] += 1

        with self.data_lock:
            if timeframe not in self.collected_data:
                self.collected_data[timeframe] = {}
            self.collected_data[timeframe]['fund_flow'] = fund_flow_data

    def perform_comprehensive_analysis_enhanced(self):
        """Enhanced analysis with new mathematical tools"""
        print("\n[PERFORMING ENHANCED MATHEMATICAL ANALYSIS WITH ADVANCED MODELS]")
        self.logger.info("Starting comprehensive enhanced analysis")

        # Process each timeframe
        for timeframe in self.timeframes:
            if timeframe not in self.kline_data:
                print(f"Skipping {timeframe} - no kline data")
                continue

            print(f"\nAnalyzing {timeframe}...")
            self.logger.info(f"Analyzing timeframe: {timeframe}")
            df = self.kline_data[timeframe]

            # Handle empty or invalid dataframes
            if df.empty or len(df) < 20:
                print(f"  ✗ Insufficient data for {timeframe}")
                self.logger.warning(f"Insufficient data for {timeframe}")
                continue

            prices = df['close'].values
            returns = df['close'].pct_change().fillna(0).values

            # Store timeframe results
            self.analysis_results[timeframe] = {}

            # Advanced Mathematical Tools
            print(f"  1. Running Multifractal Spectrum Analysis...")
            self.analysis_results[timeframe]['multifractal'] = self.advanced_tools.multifractal_spectrum_analysis(returns)

            print(f"  2. Running Sample Entropy...")
            self.analysis_results[timeframe]['sample_entropy'] = self.advanced_tools.sample_entropy(returns)

            print(f"  3. Running Recurrence Quantification Analysis...")
            self.analysis_results[timeframe]['rqa'] = self.advanced_tools.recurrence_quantification_analysis(returns)

            # Kyle Lambda calculation
            if 'order_flow' in self.collected_data.get(timeframe, {}):
                print(f"  4. Calculating Kyle Lambda...")
                order_flow_df = pd.DataFrame(self.collected_data[timeframe]['order_flow'])
                if not order_flow_df.empty and 'total_bid' in order_flow_df.columns:
                    # Create signed volume
                    signed_volume = order_flow_df['total_bid'] - order_flow_df['total_ask']
                    # Use price changes aligned with order flow
                    price_changes, signed_volume = self.align_arrays(np.diff(prices), signed_volume.values[1:])
                    if len(price_changes) > 10:
                        kyle_result = self.advanced_tools.kyle_lambda_estimation(
                            price_changes,
                            signed_volume
                        )
                        self.analysis_results[timeframe]['kyle_lambda'] = kyle_result

            # Traditional Analysis
            print(f"  5. Running EEMD analysis...")
            self.analysis_results[timeframe]['eemd'] = self.eemd(prices)

            print(f"  6. Running Unscented Kalman Filter...")
            self.analysis_results[timeframe]['ukf'] = self.unscented_kalman_filter(prices)

            print(f"  7. Running DFA & Hurst...")
            self.analysis_results[timeframe]['dfa'] = self.dfa_hurst(returns)

            print(f"  8. Running Permutation Entropy...")
            self.analysis_results[timeframe]['pe'] = self.permutation_entropy(returns)

            print(f"  9. Running Hilbert Transform...")
            self.analysis_results[timeframe]['hilbert'] = self.hilbert_homodyne(prices)

            print(f"  10. Running Matrix Profile...")
            self.analysis_results[timeframe]['matrix_profile'] = self.matrix_profile(prices)

            print(f"  11. Running Markov-Switching GARCH...")
            self.analysis_results[timeframe]['ms_garch'] = self.markov_switching_garch(returns)

            print(f"  12. Running Hidden Markov Models...")
            self.analysis_results[timeframe]['hmm'] = self.hidden_markov_model(returns)

            print(f"  13. Running Heikin Ashi MS-Signal with Z-Score...")
            self.analysis_results[timeframe]['ha_ms_signal'] = self.calculate_heikin_ashi_ms_signal(df)

            # VECM Analysis (FIXED)
            print(f"  14. Running Cointegration and VECM analysis...")
            try:
                # Prepare price data
                price_df = self.kline_data[timeframe][['close']].copy()

                # Prepare order flow data
                of_data = self.collected_data.get(timeframe, {}).get('order_flow', [])
                if of_data:
                    of_df = pd.DataFrame(of_data)
                    if 'timestamp' in of_df.columns:
                        of_df['timestamp'] = pd.to_datetime(of_df['timestamp'], unit='ms')
                        of_df = of_df.set_index('timestamp')
                        of_df['imbalance'] = of_df['total_bid'] - of_df['total_ask']
                        of_df = self._safe_resample(of_df, timeframe)
                        of_df = of_df[['imbalance']].rename(columns={'imbalance': 'orderflow'})
                    else:
                        of_df = pd.DataFrame(columns=['orderflow'])
                else:
                    of_df = pd.DataFrame(columns=['orderflow'])

                # Prepare CVD data
                cvd_data = self.collected_data.get(timeframe, {}).get('cvd', [])
                if cvd_data:
                    cvd_df = pd.DataFrame(cvd_data)
                    if 'timestamp' in cvd_df.columns:
                        cvd_df['timestamp'] = pd.to_datetime(cvd_df['timestamp'], unit='ms')
                        cvd_df = cvd_df.set_index('timestamp')
                        cvd_df = self._safe_resample(cvd_df, timeframe)
                        cvd_df = cvd_df[['cvd_close']].rename(columns={'cvd_close': 'cvd'})
                    else:
                        cvd_df = pd.DataFrame(columns=['cvd'])
                else:
                    cvd_df = pd.DataFrame(columns=['cvd'])

                # FIX: Use outer join with forward fill:
                vecm_data = price_df.join(of_df, how='outer').join(cvd_df, how='outer')
                vecm_data = vecm_data.ffill().dropna()

                if len(vecm_data) > 20 and vecm_data.shape[1] >= 2:
                    self.analysis_results[timeframe]['vecm'] = self.advanced_tools.cointegration_and_vecm(vecm_data)
                else:
                    self.analysis_results[timeframe]['vecm'] = {'error': 'Insufficient aligned data for VECM.'}
            except Exception as e:
                self.analysis_results[timeframe]['vecm'] = {'error': f'VECM data preparation failed: {str(e)}'}
                self.logger.exception("VECM analysis error")

            # Hawkes Process for Liquidations
            print(f"  15. Running Hawkes Process for Liquidation Cascades...")
            self.analysis_results[timeframe]['hawkes'] = self.analyze_liquidation_hawkes(timeframe)

        # Multi-Modal Tensor Fusion Analysis
        print("\n16. Performing Multi-Modal Tensor Fusion...")
        self.perform_tensor_fusion_analysis()

        # Information Flow Analysis with Transfer Entropy Spectrum
        print("\n17. Analyzing Information Flow Dynamics...")
        self.perform_information_flow_analysis()

        # Conditional Mutual Information Analysis
        print("\n18. Analyzing Conditional Mutual Information...")
        self.perform_cmi_analysis()

        # Microstructure Bridge Analysis
        print("\n19. Building Microstructure-Kline Bridge...")
        self.perform_microstructure_bridge_analysis()

        # CNN-based Market Prediction
        if not self.load_existing_models:
            print("\n20. Training CNN for Market Prediction...")
            self.perform_cnn_prediction()
        else:
            print("\n20. Using Pre-loaded CNN for Prediction...")
            self.perform_cnn_prediction(skip_training=True)


        # Market Data Analysis
        print("\n21. Analyzing Order Flow Patterns...")
        self.analyze_order_flow_patterns_enhanced()

        print("22. Analyzing Market Positioning...")
        self.analyze_market_positioning_enhanced()

        print("23. Analyzing Liquidation Cascades...")
        self.analyze_liquidation_cascades_enhanced()

        print("24. Generating Comprehensive Summary...")
        self.generate_comprehensive_summary_enhanced()

    def eemd(self, signal: np.ndarray) -> Dict[str, Any]:
        """Ensemble Empirical Mode Decomposition"""
        # Force wavelet fallback to avoid PyEMD issues
        use_wavelet_fallback = True
        
        if not use_wavelet_fallback:
            try:
                from PyEMD import EEMD
                
                # Disable parallel processing to avoid conflicts
                eemd_analyzer = EEMD(parallel=False, trials=50, noise_width=0.05)
                imfs = eemd_analyzer(signal)

                # Feature extraction
                dominant_periods = []
                energy_distribution = []

                for imf in imfs[:5]:  # First 5 IMFs
                    # Dominant period via FFT
                    fft = np.fft.fft(imf)
                    freqs = np.fft.fftfreq(len(imf))
                    dominant_freq_idx = np.argmax(np.abs(fft[1:len(fft)//2])) + 1
                    if not self.is_zero(freqs[dominant_freq_idx]):
                        dominant_period = 1 / freqs[dominant_freq_idx]
                    else:
                        dominant_period = len(imf)
                    dominant_periods.append(abs(dominant_period))

                    # Energy
                    energy = self.safe_divide(np.sum(imf**2), np.sum(signal**2))
                    energy_distribution.append(energy)

                # Trend (residual)
                trend = signal - np.sum(imfs, axis=0)

                # Trend strength
                trend_strength = np.polyfit(np.arange(len(trend)), trend, 1)[0]

                return {
                    'n_imfs': len(imfs),
                    'dominant_periods': dominant_periods,
                    'energy_distribution': energy_distribution,
                    'trend': trend,
                    'trend_strength': trend_strength,
                    'intrinsic_modes': imfs[:3]  # First 3 IMFs
                }

            except Exception as e:
                self.logger.warning(f"PyEMD failed: {e}, using wavelet fallback")
                use_wavelet_fallback = True

        if use_wavelet_fallback:
            self.logger.info("Using wavelet decomposition instead of EEMD")
            # Use wavelet decomposition as alternative
            import pywt
            
            # Use wavelet decomposition as alternative
            coeffs = pywt.wavedec(signal, 'db4', level=5)
            imfs = []
            for i, coeff in enumerate(coeffs[1:]):  # Skip approximation
                # Reconstruct each level
                rec_coeffs = [np.zeros_like(c) if j != i+1 else c 
                              for j, c in enumerate(coeffs)]
                imf = pywt.waverec(rec_coeffs, 'db4', mode='symmetric')
                # Trim to original length
                imfs.append(imf[:len(signal)])
            
            # Continue with similar analysis structure
            dominant_periods = []
            energy_distribution = []
            
            for i, imf in enumerate(imfs[:5]):
                # Estimate period from zero crossings
                zero_crossings = np.where(np.diff(np.sign(imf)))[0]
                if len(zero_crossings) > 1:
                    avg_period = 2 * len(imf) / len(zero_crossings)
                else:
                    avg_period = len(imf)
                dominant_periods.append(avg_period)
                
                # Calculate energy
                energy = np.sum(imf**2) / (np.sum(signal**2) + 1e-10)
                energy_distribution.append(energy)
            
            # Trend is the approximation coefficient
            trend = pywt.waverec([coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]], 
                                'db4', mode='symmetric')[:len(signal)]
            trend_strength = np.polyfit(np.arange(len(trend)), trend, 1)[0]
            
            return {
                'n_imfs': len(imfs),
                'dominant_periods': dominant_periods,
                'energy_distribution': energy_distribution,
                'trend': trend,
                'trend_strength': trend_strength,
                'intrinsic_modes': imfs[:3],
                'method': 'wavelet_fallback'
            }

    def unscented_kalman_filter(self, observations: np.ndarray) -> Dict[str, Any]:
        """Apply Unscented Kalman Filter to price data"""
        # Initialize UKF with 2D state (price and velocity)
        ukf = UnscentedKalmanFilter(dim_x=2, dim_z=1)

        # Set initial state
        ukf.x = np.array([observations[0], 0])  # Initial price and zero velocity
        ukf.P = np.eye(2) * 10  # Reduced initial uncertainty from 100 to 10

        # Process and measurement noise - adjusted for better stability
        ukf.Q = np.array([[0.001, 0], [0, 0.0001]])  # Reduced process noise
        ukf.R = np.array([[0.1]])  # Measurement noise

        # Filter the series
        try:
            results = ukf.filter_series(observations)
        except Exception as e:
            # Fallback to simple estimates if UKF fails
            self.logger.warning(f"UKF failed, using fallback: {e}")
            return {
                'filtered': observations,
                'velocity': np.zeros_like(observations),
                'innovations': np.zeros_like(observations),
                'future_price': observations[-1],
                'future_velocity': 0,
                'final_covariance': np.eye(2),
                'price_uncertainty': np.std(observations),
                'velocity_uncertainty': 0,
                'error': str(e)
            }

        # Extract predictions
        filtered_prices = results['filtered_states'][:, 0]
        filtered_velocity = results['filtered_states'][:, 1]

        # Future prediction (1 step ahead)
        ukf.predict()
        future_price = ukf.x[0]
        future_velocity = ukf.x[1]

        return {
            'filtered': filtered_prices,
            'velocity': filtered_velocity,
            'innovations': results['innovations'],
            'future_price': future_price,
            'future_velocity': future_velocity,
            'final_covariance': results['final_covariance'],
            'price_uncertainty': np.sqrt(results['final_covariance'][0, 0]),
            'velocity_uncertainty': np.sqrt(results['final_covariance'][1, 1])
        }

    def markov_switching_garch(self, returns: np.ndarray) -> Dict[str, Any]:
        """Apply Markov-Switching GARCH model"""
        try:
            # Fit the MS-GARCH model
            ms_garch = MarkovSwitchingGARCH(n_states=3)
            ms_garch.fit(returns)

            # Predict volatility
            vol_forecast = ms_garch.predict_volatility(returns, horizon=10)

            # Get current regime probabilities
            current_probs = vol_forecast['regime_probabilities']

            # Determine current regime
            regime_labels = ['Low Vol', 'Medium Vol', 'High Vol']
            current_regime = regime_labels[vol_forecast['current_regime']]

            return {
                'volatility_forecast': vol_forecast['volatility_forecast'],
                'state_volatilities': vol_forecast['state_volatilities'],
                'current_regime': current_regime,
                'regime_probabilities': vol_forecast['regime_probabilities'],
                'state_parameters': ms_garch.state_params,
                'transition_matrix': vol_forecast.get('transition_matrix')
            }
        except Exception as e:
            self.logger.exception("MS-GARCH error")
            # Fallback to standard volatility
            vol = np.std(returns)
            return {
                'volatility_forecast': np.full(10, vol),
                'state_volatilities': np.array([[vol] * 10]),
                'current_regime': 'Unknown',
                'regime_probabilities': {'Low Vol': 0.33, 'Medium Vol': 0.33, 'High Vol': 0.34},
                'error': str(e)
            }

    def dfa_hurst(self, signal: np.ndarray) -> Dict[str, float]:
        """Detrended Fluctuation Analysis and Hurst exponent"""
        signal = signal[~np.isnan(signal)]
        if len(signal) < 100:
            return {'hurst': 0.5, 'dfa_alpha': 0.5, 'market_type': 'UNKNOWN'}

        # Cumulative sum
        y = np.cumsum(signal - np.mean(signal))

        # Scales
        scales = np.logspace(1, np.log10(len(signal)//4), 20).astype(int)
        scales = np.unique(scales)

        fluctuations = []

        for scale in scales:
            # Divide into segments
            n_segments = len(y) // scale
            if n_segments == 0:
                continue

            segment_variances = []

            for i in range(n_segments):
                segment = y[i*scale:(i+1)*scale]

                # Detrend using polynomial fit
                x = np.arange(len(segment))
                coeffs = np.polyfit(x, segment, 1)
                fit = np.polyval(coeffs, x)

                # Calculate fluctuation
                segment_variance = np.mean((segment - fit)**2)
                segment_variances.append(segment_variance)

            # Average fluctuation for this scale
            if segment_variances:
                fluctuations.append(np.sqrt(np.mean(segment_variances)))

        if len(fluctuations) < 2:
            return {'hurst': 0.5, 'dfa_alpha': 0.5, 'market_type': 'UNKNOWN'}

        # Log-log fit
        log_scales = np.log(scales[:len(fluctuations)])
        log_fluctuations = np.log(fluctuations)

        alpha = np.polyfit(log_scales, log_fluctuations, 1)[0]

        # Interpret
        if alpha < 0.5:
            market_type = "MEAN_REVERTING"
        elif alpha > 0.5:
            market_type = "TRENDING"
        else:
            market_type = "RANDOM_WALK"

        return {
            'hurst': alpha,
            'dfa_alpha': alpha,
            'market_type': market_type,
            'persistence': "ANTI_PERSISTENT" if alpha < 0.5 else "PERSISTENT" if alpha > 0.5 else "UNCORRELATED"
        }

    def permutation_entropy(self, signal: np.ndarray, order: int = 3) -> Dict[str, float]:
        """Calculate permutation entropy"""
        n = len(signal)
        if n < order + 1:
            return {'pe': 1.0, 'complexity': 'HIGH', 'predictability': 0.0}

        # Create ordinal patterns
        patterns = []
        for i in range(n - order + 1):
            # Get order indices
            indices = np.argsort(signal[i:i+order])
            pattern = tuple(indices)
            patterns.append(pattern)

        # Count pattern occurrences
        pattern_counts = Counter(patterns)

        # Calculate probabilities
        probs = np.array(list(pattern_counts.values())) / len(patterns)

        # Shannon entropy
        pe = -np.sum(probs * np.log(probs))

        # Normalize
        max_entropy = np.log(math.factorial(order))
        pe_normalized = self.safe_divide(pe, max_entropy, 1.0)

        # Interpret
        if pe_normalized < 0.3:
            complexity = "LOW"
        elif pe_normalized < 0.7:
            complexity = "MEDIUM"
        else:
            complexity = "HIGH"

        predictability = 1 - pe_normalized

        return {
            'pe': pe_normalized,
            'complexity': complexity,
            'predictability': predictability,
            'n_unique_patterns': len(pattern_counts),
            'max_patterns': math.factorial(order)
        }

    def hilbert_homodyne(self, signal: np.ndarray) -> Dict[str, Any]:
        """Hilbert Transform and Homodyne Demodulation"""
        # Remove DC component
        signal_ac = signal - np.mean(signal)

        # Hilbert transform
        analytic_signal = hilbert(signal_ac)

        # Instantaneous attributes
        inst_amplitude = np.abs(analytic_signal)
        inst_phase = np.angle(analytic_signal)
        inst_frequency = np.diff(np.unwrap(inst_phase)) / (2.0 * np.pi)

        # Phase continuity
        phase_unwrapped = np.unwrap(inst_phase)
        phase_trend = np.polyfit(np.arange(len(phase_unwrapped)), phase_unwrapped, 1)[0]

        # Dominant cycle
        avg_frequency = np.mean(inst_frequency[inst_frequency > 0]) if len(inst_frequency[inst_frequency > 0]) > 0 else 0
        dominant_period = self.safe_divide(1, avg_frequency, len(signal))

        # Signal strength
        signal_strength = self.safe_divide(np.mean(inst_amplitude), np.std(signal_ac))

        return {
            'instantaneous_amplitude': inst_amplitude,
            'instantaneous_phase': inst_phase,
            'instantaneous_frequency': inst_frequency,
            'dominant_period': dominant_period,
            'phase_trend': phase_trend,
            'signal_strength': signal_strength,
            'avg_frequency': avg_frequency
        }

    def matrix_profile(self, signal: np.ndarray, window_size: int = None) -> Dict[str, Any]:
        """Calculate Matrix Profile for pattern discovery"""
        n = len(signal)

        if window_size is None:
            window_size = max(4, n // 20)

        if n < 2 * window_size:
            return {
                'matrix_profile': np.array([]),
                'motif_idx': -1,
                'discord_idx': -1,
                'pattern_score': 0.0
            }

        # Simplified matrix profile calculation
        mp = np.full(n - window_size + 1, np.inf)

        for i in range(n - window_size + 1):
            subsequence = signal[i:i + window_size]

            # Calculate distances to all other subsequences
            distances = []
            for j in range(n - window_size + 1):
                if abs(i - j) > window_size:  # Exclude trivial matches
                    other = signal[j:j + window_size]
                    dist = np.linalg.norm(subsequence - other)
                    distances.append(dist)

            if distances:
                mp[i] = np.min(distances)

        # Find motif (most repeated pattern)
        finite_mp = mp[np.isfinite(mp)]
        if len(finite_mp) > 0:
            motif_idx = np.argmin(mp)
            discord_idx = np.argmax(finite_mp)

            # Pattern score (inverse of average distance)
            pattern_score = self.safe_divide(1, np.mean(finite_mp))
        else:
            motif_idx = discord_idx = -1
            pattern_score = 0.0

        return {
            'matrix_profile': mp,
            'motif_idx': int(motif_idx),
            'discord_idx': int(discord_idx),
            'pattern_score': pattern_score,
            'window_size': window_size
        }

    def hidden_markov_model(self, returns: np.ndarray) -> Dict[str, Any]:
        """Hidden Markov Model for regime detection"""
        try:
            # Clean the returns data first
            returns_clean = returns[~np.isnan(returns) & ~np.isinf(returns)]
            
            if len(returns_clean) < 20:
                raise ValueError("Insufficient clean data for HMM")
            
            # Clip extreme values to prevent numerical issues
            returns_clean = np.clip(returns_clean, -10, 10)
            
            # Reshape returns
            X = returns_clean.reshape(-1, 1)
            
            # Scale the data to prevent numerical issues
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Fit HMM with diagonal covariance to avoid singular matrices
            model = hmm.GaussianHMM(
                n_components=3, 
                covariance_type="diag",  # Use diagonal instead of full
                n_iter=100,
                random_state=42,
                init_params="kmeans",  # Better initialization
                tol=0.01  # Increase tolerance
            )
            
            model.fit(X_scaled)

            # Get states
            states = model.predict(X_scaled)

            # State characteristics - unscale for interpretation
            state_means = scaler.inverse_transform(model.means_).flatten()
            state_vars = np.sqrt(model.covars_.flatten()) * scaler.scale_[0]

            # Sort states by mean return
            sorted_idx = np.argsort(state_means)

            # Label states
            state_labels = ['BEAR', 'NEUTRAL', 'BULL']
            current_state = states[-1]
            current_label = state_labels[np.where(sorted_idx == current_state)[0][0]]

            # Transition probabilities
            trans_probs = model.transmat_

            # State durations
            state_durations = []
            for i in range(3):
                duration = self.safe_divide(1, 1 - trans_probs[i, i], np.inf)
                state_durations.append(duration)

            # Most likely sequence
            log_prob, state_sequence = model.decode(X_scaled)

            return {
                'current_state': current_label,
                'state_probabilities': model.predict_proba(X_scaled)[-1].tolist(),
                'state_means': dict(zip(state_labels, state_means[sorted_idx])),
                'state_volatilities': dict(zip(state_labels, state_vars[sorted_idx])),
                'transition_matrix': trans_probs.tolist(),
                'expected_durations': dict(zip(state_labels, [state_durations[i] for i in sorted_idx])),
                'log_likelihood': log_prob
            }

        except Exception as e:
            self.logger.exception("HMM error")
            return {
                'current_state': 'UNKNOWN',
                'state_probabilities': [0.33, 0.33, 0.34],
                'state_means': {'BEAR': -0.01, 'NEUTRAL': 0, 'BULL': 0.01},
                'error': str(e)
            }

    def calculate_heikin_ashi_ms_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Heikin Ashi MS-Signal with Z-Score transformation"""
        try:
            # Create HA analysis instance
            ha_analyzer = HeikinAshiAnalysis(df)

            # Calculate all signals
            ha_results = ha_analyzer.calculate_all_signals()

            # Extract key metrics
            latest_idx = -1

            # Current HA candle state
            ha_close = ha_results['ha_close'].iloc[latest_idx]
            ha_open = ha_results['ha_open'].iloc[latest_idx]

            # Determine HA trend
            ha_trend = 'BULLISH' if ha_close > ha_open else 'BEARISH'

            # Z-Score HA values
            z_ha_close = ha_results['z_ha_close'].iloc[latest_idx]
            z_ha_extreme = 'OVERBOUGHT' if z_ha_close > 2 else 'OVERSOLD' if z_ha_close < -2 else 'NEUTRAL'

            # MS-Signal components
            sum_signal = ha_results['sum_signal'].iloc[latest_idx]
            signal_strength = 'STRONG_BUY' if sum_signal >= 40 else 'BUY' if sum_signal >= 20 else 'SELL' if sum_signal <= -20 else 'STRONG_SELL' if sum_signal <= -40 else 'NEUTRAL'

            # Calculate trend consistency
            lookback = min(10, len(ha_results))
            recent_ha_trends = [(ha_results['ha_close'].iloc[i] > ha_results['ha_open'].iloc[i])
                               for i in range(-lookback, 0)]
            trend_consistency = self.safe_divide(sum(recent_ha_trends), len(recent_ha_trends), 0.5)

            return {
                'ha_trend': ha_trend,
                'z_score_status': z_ha_extreme,
                'ms_signal': signal_strength,
                'sum_signal_value': float(sum_signal),
                'trend_consistency': trend_consistency,
                'ha_rsi': float(ha_results['ha_rsi'].iloc[latest_idx]),
                'momentum_aligned': bool(ha_results['mom0'].iloc[latest_idx] == ha_results['mom1'].iloc[latest_idx]),
                'obv_signal': 'BULLISH' if ha_results['obv_check'].iloc[latest_idx] > 0 else 'BEARISH' if ha_results['obv_check'].iloc[latest_idx] < 0 else 'NEUTRAL',
                'components': {
                    'dmi': float(ha_results['dmi_signal'].iloc[latest_idx]),
                    'obv': float(ha_results['obv_check'].iloc[latest_idx]),
                    'mom0': float(ha_results['mom0'].iloc[latest_idx]),
                    'mom1': float(ha_results['mom1'].iloc[latest_idx])
                }
            }

        except Exception as e:
            self.logger.exception("Heikin Ashi MS-Signal error")
            return {
                'ha_trend': 'UNKNOWN',
                'z_score_status': 'UNKNOWN',
                'ms_signal': 'NEUTRAL',
                'error': str(e)
            }

    def analyze_liquidation_hawkes(self, timeframe: str) -> Dict[str, Any]:
        """Analyze liquidation cascades using Hawkes Process"""
        if timeframe not in self.collected_data or 'liquidation' not in self.collected_data[timeframe]:
            return {'error': 'No liquidation data available'}

        liq_data = pd.DataFrame(self.collected_data[timeframe]['liquidation'])

        if liq_data.empty or 'timestamp' not in liq_data.columns:
            return {'error': 'Invalid liquidation data'}

        try:
            # Convert timestamps to seconds from start
            timestamps = pd.to_datetime(liq_data['timestamp'], unit='ms')
            event_times = (timestamps - timestamps.min()).dt.total_seconds().values
            
            # Remove any NaN values
            event_times = event_times[~np.isnan(event_times)]

            # Filter significant liquidations (above threshold)
            if 'total_value' in liq_data.columns and liq_data['total_value'].nunique() > 1:
                threshold = liq_data['total_value'].quantile(0.75)
                mask = liq_data['total_value'] > threshold
                significant_events = event_times[mask.values[:len(event_times)]]
            else:
                significant_events = event_times

            if len(significant_events) < 10:
                return {'error': 'Insufficient liquidation events'}

            # Fit Hawkes process
            hawkes = HawkesProcess()
            hawkes.fit(significant_events)

            # Predict next event probability
            time_horizon = 3600  # 1 hour ahead
            time_points, probabilities = hawkes.predict_next_event_probability(
                significant_events,
                time_horizon
            )

            # Calculate cascade risk
            branching_ratio = hawkes.branching_ratio()
            cascade_risk = 'HIGH' if branching_ratio > 0.9 else 'MEDIUM' if branching_ratio > 0.7 else 'LOW'

            # Simulate future paths
            simulations = hawkes.simulate(time_horizon, n_simulations=100)
            avg_events = np.mean([len(sim) for sim in simulations])

            return {
                'baseline_intensity': hawkes.mu,
                'jump_size': hawkes.alpha,
                'decay_rate': hawkes.beta,
                'branching_ratio': branching_ratio,
                'cascade_risk': cascade_risk,
                'next_hour_probability': float(np.max(probabilities)),
                'expected_events_next_hour': avg_events,
                'current_intensity': float(hawkes._calculate_intensities(significant_events, significant_events[-1]))
            }
            
        except Exception as e:
            self.logger.exception(f"Hawkes analysis error: {e}")
            return {'error': f'Hawkes analysis failed: {str(e)}'}

    def perform_tensor_fusion_analysis(self):
        """Perform multi-modal tensor fusion analysis"""
        try:
            tensor_data = {}
            for timeframe in self.timeframes:
                # Check if we have kline data for this timeframe
                if timeframe not in self.kline_data:
                    continue
                    
                # Use kline data directly if available
                if not self.kline_data[timeframe].empty:
                    tensor_data[timeframe] = self.kline_data[timeframe].copy()
                
                # Alternatively, use aligned data if more comprehensive
                # aligned_dfs = self._prepare_aligned_data(timeframe)
                # if aligned_dfs:
                #     # Join all dataframes into one for this timeframe
                #     combined_df = pd.concat(aligned_dfs.values(), axis=1)
                #     combined_df = combined_df.ffill().dropna()
                #     tensor_data[timeframe] = combined_df

            if not tensor_data:
                print("  ✗ No data available for tensor fusion.")
                self.logger.warning("No data available for tensor fusion")
                return
            
            # FIX: Add memory estimation
            shape = (self.lookback_size, 10, len(self.timeframes))
            self.check_memory_before_allocation(shape)
            
            market_tensor = self.tensor_fusion.create_unified_market_tensor(tensor_data)
            decomposition_results = self.tensor_fusion.tensor_decomposition_prediction(market_tensor)

            self.analysis_results['tensor_fusion'] = {
                'tensor_shape': market_tensor.shape,
                'decomposition_rank': 10,
                'temporal_patterns': decomposition_results['temporal_patterns'],
                'factor_predictions': decomposition_results['predictions'],
                'reconstruction_error': decomposition_results['reconstruction_error']
            }

            print(f"  ✓ Tensor shape: {market_tensor.shape}")
            print(f"  ✓ Reconstruction error: {decomposition_results['reconstruction_error']:.4f}")

        except Exception as e:
            self.logger.exception("Tensor fusion error")
            print(f"  ✗ Tensor fusion error: {str(e)}")
            self.analysis_results['tensor_fusion'] = {
                'error': str(e),
                'tensor_shape': (0, 0, 0),
                'reconstruction_error': 1.0
            }

    def perform_information_flow_analysis(self):
        """Analyze information flow between different data sources"""
        try:
            # Prepare data series for information flow analysis using the robust alignment method
            aligned_data_dfs = self._prepare_aligned_data('1h')

            if not aligned_data_dfs:
                print("  ✗ No aligned data available for information flow analysis.")
                self.logger.warning("No aligned data for information flow analysis")
                return

            info_flow_data = {name: df[name] for name, df in aligned_data_dfs.items()}

            if len(info_flow_data) >= 3:
                # Calculate transfer entropy matrix
                te_results = self.info_flow_analyzer.calculate_transfer_entropy_matrix(info_flow_data)

                # Partial information decomposition
                pid_results = self.info_flow_analyzer.partial_information_decomposition(info_flow_data)

                self.analysis_results['information_flow'] = {
                    'transfer_entropy': te_results['transfer_entropy'].to_dict(),
                    'optimal_lags': te_results['optimal_lags'].to_dict(),
                    'network_analysis': te_results.get('network_analysis', {}),
                    'information_contributions': pid_results['individual_contributions'],
                    'total_information': pid_results['total_information'],
                    'redundancy_factor': pid_results['redundancy_factor'],
                    'synergy_potential': pid_results['synergy_potential']
                }

                # Identify top information sources
                contributions = pid_results['individual_contributions']
                if contributions:
                    top_sources = sorted(contributions.items(), key=lambda x: x[1], reverse=True)[:3]
                    self.analysis_results['information_flow']['top_sources'] = top_sources

                print(f"  ✓ Analyzed information flow between {len(info_flow_data)} endpoints")
                print(f"  ✓ Total information: {pid_results['total_information']:.4f}")

        except Exception as e:
            self.logger.exception("Information flow error")
            print(f"  ✗ Information flow error: {str(e)}")
            self.analysis_results['information_flow'] = {'error': str(e)}

    def perform_cmi_analysis(self):
        """Perform Conditional Mutual Information analysis"""
        try:
            # Prepare data for CMI analysis
            cmi_data = self._prepare_aligned_data('1h')

            # Calculate CMI
            if len(cmi_data) >= 3:
                # Convert dataframes to series for the analyzer
                cmi_series_data = {name: df[name] for name, df in cmi_data.items()}

                cmi_results = self.info_flow_analyzer.calculate_conditional_mutual_information(cmi_series_data)
                self.analysis_results['conditional_mutual_information'] = cmi_results

                print(f"  ✓ Analyzed CMI for {len(cmi_data)} data streams")
                if 'top_synergies' in cmi_results:
                    print("  ✓ Top synergistic relationships identified:")
                    for pair, synergy in cmi_results.get('top_synergies', [])[:3]:
                        print(f"    - {pair}: {synergy:.4f}")
            else:
                print("  ✗ Insufficient data for CMI analysis")
                self.logger.warning("Insufficient data for CMI analysis")

        except Exception as e:
            self.logger.exception("CMI analysis error")
            print(f"  ✗ CMI analysis error: {str(e)}")
            self.analysis_results['conditional_mutual_information'] = {'error': str(e)}

    def perform_microstructure_bridge_analysis(self):
        """Bridge microstructure data to kline predictions"""
        try:
            # Use 15m timeframe for higher resolution
            if '15m' in self.kline_data:
                kline_df = self.kline_data['15m']

                # Get current bar info
                current_bar = {
                    'open': float(kline_df.iloc[-1]['open']),
                    'high': float(kline_df.iloc[-1]['high']),
                    'low': float(kline_df.iloc[-1]['low']),
                    'current': float(kline_df.iloc[-1]['close']),
                    'time_remaining_ratio': 0.5  # Assume midway through bar
                }

                # Get orderflow data
                orderflow_data = pd.DataFrame()
                if '15m' in self.collected_data and 'order_flow' in self.collected_data['15m']:
                    orderflow_data = pd.DataFrame(self.collected_data['15m']['order_flow'])
                    if not orderflow_data.empty:
                        # Add price column if missing
                        if 'price' not in orderflow_data.columns:
                            # Use a simple forward fill from kline close for approximation
                            temp_price_series = kline_df['close'].reindex(pd.to_datetime(orderflow_data['timestamp'], unit='ms'), method='ffill')
                            orderflow_data['price'] = temp_price_series.values
                        # Add volume column
                        if 'volume' not in orderflow_data.columns and 'total_bid' in orderflow_data.columns:
                            orderflow_data['volume'] = orderflow_data['total_bid'] + orderflow_data['total_ask']
                        # Add delta column
                        if 'delta' not in orderflow_data.columns and 'total_bid' in orderflow_data.columns:
                            orderflow_data['delta'] = orderflow_data['total_bid'] - orderflow_data['total_ask']

                # Get orderbook data
                book_data = pd.DataFrame()
                if '15m' in self.collected_data and 'orderbook' in self.collected_data['15m']:
                    book_df = pd.DataFrame(self.collected_data['15m']['orderbook'])
                    if not book_df.empty:
                        # Create book data format
                        book_data = pd.DataFrame({
                            'bid_size': book_df['buy_usd'],
                            'ask_size': book_df['sell_usd'],
                            'bid_price': [current_bar['current'] * 0.9995] * len(book_df),  # Approximate bid
                            'ask_price': [current_bar['current'] * 1.0005] * len(book_df),  # Approximate ask
                            'mid': [current_bar['current']] * len(book_df)
                        })

                # Predict current bar close
                predictions = self.microstructure_bridge.predict_current_bar_close(
                    current_bar, orderflow_data.dropna(), book_data.dropna()
                )

                self.analysis_results['microstructure_bridge'] = {
                    'current_price': current_bar['current'],
                    'predictions': predictions,
                    'best_prediction': predictions.get('ensemble', current_bar['current']),
                    'confidence': predictions.get('confidence', 0)
                }

                print(f"  ✓ Current price: ${current_bar['current']:,.2f}")
                print(f"  ✓ Predicted close: ${predictions.get('ensemble', 0):,.2f} "
                      f"(confidence: {predictions.get('confidence', 0):.1f}%)")

        except Exception as e:
            self.logger.exception("Microstructure bridge error")
            print(f"  ✗ Microstructure bridge error: {str(e)}")
            self.analysis_results['microstructure_bridge'] = {'error': str(e)}

    def perform_cnn_prediction(self, skip_training=False):
        """Train and use CNN for market prediction - FIXED"""
        try:
            if skip_training and self.cnn_predictor.is_trained:
                print("    Using pre-loaded CNN model for prediction.")
            else:
                if len(self.kline_data) < 3:
                    print("  ✗ Insufficient timeframes for CNN training")
                    return
                
                X, y = self.cnn_predictor.prepare_data(self.kline_data)
                
                if len(X) < 100:
                    print("  ✗ Insufficient samples for CNN training")
                    return
                
                train_size = int(0.8 * len(X))
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]
                
                print("    Training CNN model...")
                self.cnn_predictor.train(X_train, y_train, epochs=30, batch_size=32)
                
                test_predictions = self.cnn_predictor.predict(X_test)
                accuracy = np.mean(test_predictions['predictions'] == y_test.numpy())
                self.analysis_results['cnn_prediction'] = {'test_accuracy': float(accuracy * 100)}
                print(f"  ✓ CNN trained with {accuracy*100:.2f}% test accuracy")

            # Predict next period
            X_pred, _ = self.cnn_predictor.prepare_data(self.kline_data)
            
            if len(X_pred) > 0:
                next_prediction = self.cnn_predictor.predict(X_pred[-1:])
                
                self.analysis_results.setdefault('cnn_prediction', {}).update({
                    'next_period_prediction': next_prediction['labels'][next_prediction['predictions'][0]],
                    'prediction_probabilities': next_prediction['probabilities'][0].tolist(),
                    'labels': next_prediction['labels']
                })
                print(f"  ✓ Next period prediction: {next_prediction['labels'][next_prediction['predictions'][0]]}")
            
        except Exception as e:
            self.logger.exception(f"CNN prediction error: {e}")
            print(f"  ✗ CNN prediction error: {str(e)}")
            self.analysis_results['cnn_prediction'] = {'error': str(e)}

    def _prepare_aligned_data(self, timeframe: str) -> Dict[str, pd.DataFrame]:
        """Helper to prepare timestamp-aligned data with improved missing data handling"""
        aligned_data = {}
        if timeframe not in self.kline_data:
            return aligned_data

        # Start with k-line data as the base
        base_df = self.kline_data[timeframe][['close']].copy()
        base_df.columns = ['kline_price']

        # Helper function to process and merge data
        def process_and_merge(data_key, col_map, agg_method='mean'):
            if timeframe in self.collected_data and data_key in self.collected_data[timeframe]:
                df_raw = pd.DataFrame(self.collected_data[timeframe][data_key])
                if not df_raw.empty and 'timestamp' in df_raw.columns:
                    # Ensure timestamp is numeric before conversion
                    df_raw['timestamp'] = pd.to_numeric(df_raw['timestamp'], errors='coerce')
                    df_raw = df_raw.dropna(subset=['timestamp'])
                    
                    if df_raw.empty:
                        return
                    
                    df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'], unit='ms')
                    df_raw = df_raw.set_index('timestamp')
                    
                    # Use safe resample
                    df_proc = self._safe_resample(df_raw, timeframe)
                    
                    for old_name, new_name in col_map.items():
                        if old_name in df_proc.columns:
                            # Use interpolation instead of forward fill for smoother transitions
                            series = df_proc[old_name].reindex(base_df.index)
                            
                            # Interpolate only for small gaps (max 3 periods)
                            mask = series.isna()
                            gap_groups = mask.ne(mask.shift()).cumsum()
                            gap_sizes = mask.groupby(gap_groups).transform('sum')
                            
                            # Interpolate small gaps, keep large gaps as NaN
                            small_gaps = gap_sizes <= 3
                            series.loc[mask & small_gaps] = series.interpolate(method='linear').loc[mask & small_gaps]
                            
                            # For remaining NaN, use the median of the series
                            if series.isna().any():
                                median_val = series.median()
                                if pd.isna(median_val):
                                    # If median is also NaN, use 0 as last resort
                                    median_val = 0
                                series.fillna(median_val, inplace=True)
                            
                            aligned_data[new_name] = pd.DataFrame({new_name: series})

        # Define data sources and their column mappings
        data_sources = {
            'order_flow': ({'total_bid': 'orderflow'}, 'mean'),
            'orderbook': ({'imbalance': 'orderbook_pressure'}, 'mean'),
            'cvd': ({'cvd_close': 'cvd'}, 'mean'),
            'liquidation': ({'total_value': 'liquidations'}, 'sum'),
            'funding': ({'oi_weighted_rate': 'funding_rate'}, 'mean'),
            'open_interest': ({'oi_value': 'open_interest'}, 'mean'),
            'market_orders': ({'net_volume': 'market_orders'}, 'sum')
        }

        # Always include kline_price
        aligned_data['kline_price'] = base_df

        # Process each data source
        for key, (mapping, agg) in data_sources.items():
            process_and_merge(key, mapping, agg)

        # Special handling for large orders - combine different types
        if timeframe in self.collected_data and 'large_orders' in self.collected_data[timeframe]:
            large_orders_df = pd.DataFrame(self.collected_data[timeframe]['large_orders'])
            if not large_orders_df.empty and 'timestamp' in large_orders_df.columns:
                # Ensure timestamp is numeric
                large_orders_df['timestamp'] = pd.to_numeric(large_orders_df['timestamp'], errors='coerce')
                large_orders_df = large_orders_df.dropna(subset=['timestamp'])
                
                if not large_orders_df.empty:
                    large_orders_df['timestamp'] = pd.to_datetime(large_orders_df['timestamp'], unit='ms')
                    large_orders_df = large_orders_df.set_index('timestamp')
                    
                    # Calculate net large order flow
                    if 'side' in large_orders_df.columns and 'amount' in large_orders_df.columns:
                        large_orders_df['signed_amount'] = large_orders_df.apply(
                            lambda x: x['amount'] if x['side'] == 'buy' else -x['amount'], axis=1
                        )
                        
                        # Resample
                        resampled = self._safe_resample(large_orders_df[['signed_amount']], timeframe)
                        if not resampled.empty:
                            series = resampled['signed_amount'].reindex(base_df.index)
                            
                            # Use same interpolation logic
                            mask = series.isna()
                            if mask.any():
                                gap_groups = mask.ne(mask.shift()).cumsum()
                                gap_sizes = mask.groupby(gap_groups).transform('sum')
                                small_gaps = gap_sizes <= 3
                                series.loc[mask & small_gaps] = series.interpolate(method='linear').loc[mask & small_gaps]
                                
                                # Fill remaining with 0 (no large orders)
                                series.fillna(0, inplace=True)
                            
                            aligned_data['large_orders'] = pd.DataFrame({'large_orders': series})

        # Return only non-empty dataframes
        return {k: v for k, v in aligned_data.items() if not v.empty}

    def analyze_order_flow_patterns_enhanced(self):
        """Enhanced order flow analysis with new mathematical tools"""
        print("\n[ANALYZING ORDER FLOW PATTERNS - ENHANCED]")
        self.write_to_file("\n" + "="*100)
        self.write_to_file("ORDER FLOW ANALYSIS - ENHANCED WITH ADVANCED MODELS")
        self.write_to_file("="*100)

        for timeframe in self.timeframes[:3]:  # Focus on shorter timeframes
            if timeframe not in self.collected_data or 'order_flow' not in self.collected_data[timeframe]:
                continue

            order_flow_data = self.collected_data[timeframe]['order_flow']
            if not order_flow_data:
                continue

            self.write_to_file(f"\n{timeframe} ORDER FLOW PATTERNS:")
            self.write_to_file("-" * 50)

            exchange_data = defaultdict(list)

            for record in order_flow_data:
                key = f"{record['exchange']}_{record.get('tick_count', 1)}"
                exchange_data[key].append(record)

            for key, records in exchange_data.items():
                if len(records) < 10:
                    continue

                self.write_to_file(f"\n{key}:")

                # Calculate cumulative metrics
                total_ask_volume = sum(r['total_ask'] for r in records)
                total_bid_volume = sum(r['total_bid'] for r in records)

                if total_ask_volume + total_bid_volume > 0:
                    bid_ratio = total_bid_volume / (total_ask_volume + total_bid_volume)

                    # Calculate flow imbalance
                    imbalances = [self.safe_divide(r['total_bid'] - r['total_ask'], r['total_bid'] + r['total_ask'])
                                 for r in records]

                    # FIX: Match by timestamp for Kyle Lambda - AVOIDING LOOKAHEAD BIAS
                    if timeframe in self.kline_data:
                        # Get timestamps from records
                        record_timestamps = pd.to_datetime([r['timestamp'] for r in records], unit='ms')
                        
                        # Use forward fill to avoid lookahead bias
                        price_series = self.kline_data[timeframe]['close']
                        aligned_prices = price_series.reindex(record_timestamps, method='ffill').dropna()
                        
                        # Additional check to ensure no future data
                        current_time = self.current_timestamp
                        mask = aligned_prices.index <= pd.Timestamp(current_time, unit='ms')
                        aligned_prices = aligned_prices[mask]

                        if len(aligned_prices) > 1:
                            price_changes = np.diff(aligned_prices.values)
                            
                            # Align signed volume with aligned prices
                            of_df = pd.DataFrame(records).set_index(pd.to_datetime(pd.DataFrame(records)['timestamp'], unit='ms'))
                            aligned_of = of_df.reindex(aligned_prices.index)
                            
                            # Check if we have the required columns
                            if 'total_bid' in aligned_of.columns and 'total_ask' in aligned_of.columns:
                                signed_volume = (aligned_of['total_bid'] - aligned_of['total_ask']).values[1:]
                                
                                # Ensure arrays are aligned
                                price_changes, signed_volume = self.align_arrays(price_changes, signed_volume)
                                
                                if len(price_changes) > 10:
                                    # Remove any NaN or inf values
                                    mask = ~(np.isnan(price_changes) | np.isnan(signed_volume) | 
                                           np.isinf(price_changes) | np.isinf(signed_volume))
                                    
                                    if mask.sum() > 10:
                                        price_changes = price_changes[mask]
                                        signed_volume = signed_volume[mask]
                                        
                                        kyle_result = self.advanced_tools.kyle_lambda_estimation(
                                            price_changes,
                                            signed_volume
                                        )

                                        self.write_to_file(f"  Kyle Lambda: {kyle_result['lambda']:.6f}")
                                        self.write_to_file(f"  Market Depth: ${kyle_result['market_depth']:,.0f}")
                                        self.write_to_file(f"  Price Impact per $1M: ${abs(kyle_result['lambda'] * 1_000_000):,.2f}")
                                        self.write_to_file(f"  Kyle Lambda R²: {kyle_result['r_squared']:.3f}")
                                        
                                        # Additional microstructure metrics
                                        if 'amihud_illiquidity' in kyle_result:
                                            self.write_to_file(f"  Amihud Illiquidity: {kyle_result['amihud_illiquidity']:.6f}")

                    # Analyze patterns
                    self.write_to_file(f"  Total Volume: {total_ask_volume + total_bid_volume:,.0f}")
                    self.write_to_file(f"  Bid Ratio: {bid_ratio:.2%}")
                    self.write_to_file(f"  Avg Imbalance: {np.mean(imbalances):.3f}")
                    self.write_to_file(f"  Imbalance Volatility: {np.std(imbalances):.3f}")

                    # Detect trends
                    recent_imbalance = np.mean(imbalances[-5:]) if len(imbalances) >= 5 else np.mean(imbalances)
                    if recent_imbalance > 0.1:
                        self.write_to_file(f"  Status: BULLISH FLOW (Recent Imbalance: {recent_imbalance:.3f})")
                    elif recent_imbalance < -0.1:
                        self.write_to_file(f"  Status: BEARISH FLOW (Recent Imbalance: {recent_imbalance:.3f})")
                    else:
                        self.write_to_file(f"  Status: NEUTRAL FLOW")
                    
                    # Advanced flow analysis
                    if len(imbalances) >= 20:
                        # Calculate flow momentum
                        flow_momentum = np.polyfit(range(len(imbalances)), imbalances, 1)[0]
                        self.write_to_file(f"  Flow Momentum: {flow_momentum:.4f}")
                        
                        # Check for flow reversals
                        recent_5 = np.mean(imbalances[-5:])
                        recent_10_to_5 = np.mean(imbalances[-10:-5]) if len(imbalances) >= 10 else recent_5
                        
                        if abs(recent_5) > 0.1 and abs(recent_10_to_5) > 0.1:
                            if np.sign(recent_5) != np.sign(recent_10_to_5):
                                self.write_to_file(f"  ⚠️ FLOW REVERSAL DETECTED")
                        
                        # Volume-weighted imbalance
                        volumes = [r['total_bid'] + r['total_ask'] for r in records[-len(imbalances):]]
                        if sum(volumes) > 0:
                            vw_imbalance = np.average(imbalances, weights=volumes)
                            self.write_to_file(f"  Volume-Weighted Imbalance: {vw_imbalance:.3f}")

    def analyze_market_positioning_enhanced(self):
        """Enhanced positioning analysis"""
        print("\n[ANALYZING MARKET POSITIONING - ENHANCED]")
        self.write_to_file("\n" + "="*100)
        self.write_to_file("MARKET POSITIONING ANALYSIS - ENHANCED")
        self.write_to_file("="*100)

        all_positioning = []

        for timeframe in self.timeframes[:2]:  # Use shorter timeframes
            if timeframe not in self.collected_data:
                continue

            # Positioning data
            if 'positioning' in self.collected_data[timeframe]:
                positioning_data = self.collected_data[timeframe]['positioning']
                all_positioning.extend(positioning_data)

            # Open Interest
            if 'open_interest' in self.collected_data[timeframe]:
                self.write_to_file(f"\n{timeframe} OPEN INTEREST ANALYSIS:")
                self.write_to_file("-" * 50)

                oi_data = self.collected_data[timeframe]['open_interest']

                for oi_type in ['aggregated', 'Binance', 'OKX', 'Bybit']:
                    type_data = [d for d in oi_data if d['type'] == oi_type]
                    if not type_data:
                        continue

                    self.write_to_file(f"\n{oi_type}:")

                    recent_oi = [d['oi_value'] for d in type_data[-10:]]
                    recent_changes = [d['change_pct'] for d in type_data[-10:] if 'change_pct' in d]

                    if recent_oi:
                        self.write_to_file(f"  Current OI: ${recent_oi[-1]:,.0f}")
                        self.write_to_file(f"  10-period Change: {self.safe_divide(recent_oi[-1] - recent_oi[0], recent_oi[0]) * 100:.2f}%")

                        if recent_changes:
                            avg_change = np.mean(recent_changes)
                            if avg_change > 2:
                                self.write_to_file(f"  Trend: INCREASING RAPIDLY ({avg_change:.2f}% avg)")
                            elif avg_change > 0:
                                self.write_to_file(f"  Trend: INCREASING ({avg_change:.2f}% avg)")
                            elif avg_change < -2:
                                self.write_to_file(f"  Trend: DECREASING RAPIDLY ({avg_change:.2f}% avg)")
                            else:
                                self.write_to_file(f"  Trend: STABLE ({avg_change:.2f}% avg)")

        # Analyze positioning ratios
        if all_positioning:
            self.write_to_file("\nPOSITIONING RATIOS SUMMARY:")
            self.write_to_file("-" * 50)

            for pos_type in ['global_account', 'position', 'top_trader']:
                type_data = [d for d in all_positioning if d['type'] == pos_type]
                if not type_data:
                    continue

                recent_ratios = [d['ls_ratio'] for d in type_data[-20:]]
                if recent_ratios:
                    current_ratio = recent_ratios[-1]
                    avg_ratio = np.mean(recent_ratios)

                    self.write_to_file(f"\n{pos_type.upper()}:")
                    self.write_to_file(f"  Current L/S Ratio: {current_ratio:.3f}")
                    self.write_to_file(f"  20-period Average: {avg_ratio:.3f}")

                    if current_ratio > 1.5: sentiment = "EXTREMELY BULLISH"
                    elif current_ratio > 1.2: sentiment = "BULLISH"
                    elif current_ratio < 0.8: sentiment = "BEARISH"
                    elif current_ratio < 0.5: sentiment = "EXTREMELY BEARISH"
                    else: sentiment = "NEUTRAL"
                    self.write_to_file(f"  Sentiment: {sentiment}")

                    if len(recent_ratios) >= 5:
                        trend = np.polyfit(range(len(recent_ratios)), recent_ratios, 1)[0]
                        if trend > 0.01: self.write_to_file(f"  Trend: INCREASING ({trend:.4f}/period)")
                        elif trend < -0.01: self.write_to_file(f"  Trend: DECREASING ({trend:.4f}/period)")
                        else: self.write_to_file(f"  Trend: STABLE")

    def analyze_liquidation_cascades_enhanced(self):
        """Enhanced liquidation analysis with Hawkes Process"""
        print("\n[ANALYZING LIQUIDATION CASCADES - ENHANCED WITH HAWKES]")
        self.write_to_file("\n" + "="*100)
        self.write_to_file("LIQUIDATION CASCADE ANALYSIS - HAWKES PROCESS")
        self.write_to_file("="*100)

        for timeframe in self.timeframes[:3]:
            if timeframe not in self.collected_data or 'liquidation' not in self.collected_data[timeframe]:
                continue

            liq_data = self.collected_data[timeframe]['liquidation']
            if not liq_data:
                continue

            self.write_to_file(f"\n{timeframe} LIQUIDATION PATTERNS:")
            self.write_to_file("-" * 50)

            # Hawkes analysis results
            if timeframe in self.analysis_results and 'hawkes' in self.analysis_results[timeframe]:
                hawkes_results = self.analysis_results[timeframe]['hawkes']

                if 'error' not in hawkes_results:
                    self.write_to_file("\nHAWKES PROCESS ANALYSIS:")
                    self.write_to_file(f"  Baseline Intensity (λ₀): {hawkes_results['baseline_intensity']:.4f} events/second")
                    self.write_to_file(f"  Jump Size (α): {hawkes_results['jump_size']:.4f}")
                    self.write_to_file(f"  Decay Rate (β): {hawkes_results['decay_rate']:.4f}")
                    self.write_to_file(f"  Branching Ratio (α/β): {hawkes_results['branching_ratio']:.4f}")
                    self.write_to_file(f"  CASCADE RISK: {hawkes_results['cascade_risk']}")
                    self.write_to_file(f"  Next Hour Event Probability: {hawkes_results['next_hour_probability']:.2%}")
                    self.write_to_file(f"  Expected Events Next Hour: {hawkes_results['expected_events_next_hour']:.1f}")

            # Traditional analysis
            total_long_liq = sum(d.get('long_value', 0) for d in liq_data)
            total_short_liq = sum(d.get('short_value', 0) for d in liq_data)
            total_liq = total_long_liq + total_short_liq

            if total_liq > 0:
                long_ratio = total_long_liq / total_liq

                self.write_to_file(f"\nLIQUIDATION SUMMARY:")
                self.write_to_file(f"  Total Liquidations: ${total_liq:,.0f}")
                self.write_to_file(f"  Long Liquidations: ${total_long_liq:,.0f} ({long_ratio:.1%})")
                self.write_to_file(f"  Short Liquidations: ${total_short_liq:,.0f} ({(1-long_ratio):.1%})")

                recent_data = liq_data[-20:] if len(liq_data) >= 20 else liq_data
                recent_long = sum(d.get('long_value', 0) for d in recent_data)
                recent_short = sum(d.get('short_value', 0) for d in recent_data)

                if recent_long > recent_short * 1.5:
                    self.write_to_file(f"  Recent Bias: LONG SQUEEZE (L/S Ratio: {self.safe_divide(recent_long, recent_short):.2f})")
                elif recent_short > recent_long * 1.5:
                    self.write_to_file(f"  Recent Bias: SHORT SQUEEZE (S/L Ratio: {self.safe_divide(recent_short, recent_long):.2f})")
                else:
                    self.write_to_file(f"  Recent Bias: BALANCED")

                if 'amount' in liq_data[0]:
                    large_liqs = [d for d in liq_data if d.get('order_type') == 'order' and d.get('amount', 0) > 100000]
                    if large_liqs:
                        self.write_to_file(f"\nLARGE LIQUIDATIONS (>$100k): {len(large_liqs)}")
                        for liq in large_liqs[-5:]:
                            self.write_to_file(f"  ${liq['amount']:,.0f} {liq.get('side', 'UNKNOWN')} "
                                             f"@ ${liq.get('price', 0):,.2f}")

    def generate_comprehensive_summary_enhanced(self):
        """Generate enhanced summary with projections"""
        print("\n[GENERATING ENHANCED COMPREHENSIVE SUMMARY WITH PROJECTIONS]")
        self.logger.info("Generating comprehensive summary")

        # Write detailed results
        self.write_detailed_results()

        # Generate summary
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            f.write(f"MATHEMATICAL ANALYSIS SUMMARY V45 - {self.symbol_config['name']}\n")
            f.write("="*100 + "\n")
            f.write(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC\n")
            f.write(f"Analyst: {self.analyst}\n")
            f.write("="*100 + "\n\n")

            # --- KEY TAKEAWAYS ---
            f.write("KEY TAKEAWAYS & ACTIONABLE INSIGHTS\n")
            f.write("-"*50 + "\n")

            bullish_signals, bearish_signals, total_signals, timeframe_signals = self._calculate_aggregate_bias()
            bullish_pct = self.calculate_weighted_bias(timeframe_signals['breakdown'], tf_weights={'15m': 1.5, '30m': 1.25, '1h': 1.0, '4h': 0.8, '6h': 0.7, '12h': 0.6, '1d': 0.5})
            overall_bias = self._get_bias_label(bullish_pct)

            f.write(f"1. OVERALL MARKET BIAS: {overall_bias} ({bullish_pct:.1f}% Bullish)\n")
            f.write(f"   - Short-term (15m-1h) bias appears {self._get_bias_label(timeframe_signals.get('short_term_bull_pct', 50))}.\n")
            f.write(f"   - Long-term (4h-1d) bias appears {self._get_bias_label(timeframe_signals.get('long_term_bull_pct', 50))}.\n")

            # FIX: Missing Data Handling in Key Takeaways
            cnn_pred_label = "NEUTRAL"
            if 'cnn_prediction' in self.analysis_results and 'error' not in self.safe_get_nested(self.analysis_results, ['cnn_prediction'], {}):
                cnn = self.analysis_results['cnn_prediction']
                cnn_pred_label = self.safe_get_nested(cnn, ['next_period_prediction'], 'UNKNOWN')
                probs = self.safe_get_nested(cnn, ['prediction_probabilities'], [])
                cnn_confidence = (max(probs) * 100) if probs else 0.0
                # FIX: Hardcoded Assumptions
                prediction_horizon = getattr(self.cnn_predictor, 'prediction_horizon', 5)
                horizon_hours = prediction_horizon * 15 / 60  # Assuming 15min bars
                f.write(f"2. CNN PREDICTION (Next ~{horizon_hours:.2f} hours): {self.format_prediction_with_confidence(cnn_pred_label, cnn_confidence)}\n")
            else:
                f.write(f"2. CNN PREDICTION: Analysis failed or not available.\n")

            if 'microstructure_bridge' in self.analysis_results and 'error' not in self.safe_get_nested(self.analysis_results, ['microstructure_bridge'], {}):
                bridge = self.analysis_results['microstructure_bridge']
                current_price = bridge.get('current_price', 0)
                pred_price = bridge.get('best_prediction', 0)
                if current_price > 0 and pred_price > 0:
                    direction = "UP" if pred_price > current_price else "DOWN"
                    f.write(f"3. MICROSTRUCTURE (Next 15 min): Pressure suggests a move {direction} towards ${pred_price:,.2f} ({self.format_prediction_with_confidence('', bridge.get('confidence', 0))}).\n")

            # FIX: Unsafe Dictionary Access
            cascade_risk = self.safe_get_nested(self.analysis_results, ['1h', 'hawkes', 'cascade_risk'], 'UNKNOWN')
            vol_regime = self.safe_get_nested(self.analysis_results, ['1h', 'ms_garch', 'current_regime'], 'Unknown')
            f.write(f"4. RISK ASSESSMENT: Liquidation cascade risk is {cascade_risk}. Volatility is in a {vol_regime} regime.\n")

            # FIX: Conditional Strategy Recommendations
            if overall_bias.endswith("BULLISH") and cnn_pred_label == 'Bull':
                f.write("5. STRATEGY: Consider long positions. Models show broad bullish alignment. Monitor for confirmation from microstructure data.\n")
            elif overall_bias.endswith("BEARISH") and cnn_pred_label == 'Bear':
                f.write("5. STRATEGY: Consider short positions. Models show broad bearish alignment. High cascade risk could accelerate downside.\n")
            else:
                f.write("5. STRATEGY: Market is in a mixed state. Range-trading or waiting for a clearer signal is advisable. Pay attention to short-term microstructure cues.\n")

            f.write("\n\n")

            # --- DETAILED SUMMARY ---
            f.write("DETAILED SUMMARY\n")
            f.write("="*50 + "\n\n")

            f.write("I. CURRENT MARKET STATE\n")
            f.write("-"*50 + "\n")
            # FIX: Price Display Without Validation
            if ('15m' in self.kline_data and not self.kline_data['15m'].empty and self.validate_dataframe(self.kline_data['15m'], ['close'])):
                f.write(f"Current Price: ${self.kline_data['15m']['close'].iloc[-1]:,.2f}\n")
            else:
                f.write(f"Current Price: Not available\n")
                
            f.write(f"Volatility Regime: {vol_regime}\n")
            if '1h' in self.analysis_results and 'dfa' in self.safe_get_nested(self.analysis_results, ['1h'], {}):
                f.write(f"Market Type: {self.safe_get_nested(self.analysis_results, ['1h', 'dfa', 'market_type'])} (Hurst: {self.safe_get_nested(self.analysis_results, ['1h', 'dfa', 'hurst'], 0.5):.3f})\n")
            if '1h' in self.analysis_results and 'hmm' in self.safe_get_nested(self.analysis_results, ['1h'], {}) and 'error' not in self.safe_get_nested(self.analysis_results, ['1h', 'hmm'], {}):
                f.write(f"HMM State: {self.safe_get_nested(self.analysis_results, ['1h', 'hmm', 'current_state'])}\n")
            f.write("\n")

            f.write("II. PREDICTIONS AND PROJECTIONS\n")
            f.write("-"*50 + "\n")
            if 'cnn_prediction' in self.analysis_results and 'error' not in self.safe_get_nested(self.analysis_results, ['cnn_prediction'], {}):
                cnn = self.analysis_results['cnn_prediction']
                f.write(f"CNN Market Prediction:\n")
                f.write(f"  - Next Period Direction: {self.safe_get_nested(cnn, ['next_period_prediction'], 'N/A')}\n")
                probs = self.safe_get_nested(cnn, ['prediction_probabilities'], [])
                cnn_confidence = (max(probs) * 100) if probs else 0.0
                f.write(f"  - Confidence: {cnn_confidence:.1f}%\n")
                if 'best_hyperparameters' in cnn:
                    f.write(f"  - Optimized with: lr={self.safe_get_nested(cnn, ['best_hyperparameters', 'lr'], 'N/A'):.5f}\n")
            if 'microstructure_bridge' in self.analysis_results and 'error' not in self.safe_get_nested(self.analysis_results, ['microstructure_bridge'], {}):
                bridge = self.analysis_results['microstructure_bridge']
                f.write(f"Microstructure Prediction (15m Bar):\n")
                f.write(f"  - Predicted Close: ${bridge.get('best_prediction', 0):,.2f} (Confidence: {bridge.get('confidence', 0):.1f}%)\n")
            if '1h' in self.analysis_results and 'ukf' in self.safe_get_nested(self.analysis_results, ['1h'], {}):
                ukf = self.analysis_results['1h']['ukf']
                f.write(f"UKF State Estimation (1h):\n")
                f.write(f"  - Future Price (1 step): ${self.safe_get_nested(ukf, ['future_price'], 0):,.2f} (±${self.safe_get_nested(ukf, ['price_uncertainty'], 0):,.2f})\n")
            f.write("\n")

            f.write("III. CROSS-TIMEFRAME SYNTHESIS\n")
            f.write("-"*50 + "\n")
            f.write(f"Overall Market Bias: {overall_bias} ({bullish_pct:.1f}% Bullish)\n")
            f.write("Timeframe Signal Breakdown:\n")
            # FIX: Percentage Calculation in Timeframe Breakdown
            if 'breakdown' in timeframe_signals:
                for tf, signals in timeframe_signals['breakdown'].items():
                    if isinstance(signals, dict) and signals.get('total', 0) > 0:
                        bull_pct_tf = self.safe_divide(signals.get('bullish', 0), signals.get('total', 1)) * 100
                        f.write(f"  - {tf:<4}: {self._get_bias_label(bull_pct_tf)} ({bull_pct_tf:.0f}% Bullish)\n")
            f.write("\n")

            if 'information_flow' in self.analysis_results and 'error' not in self.safe_get_nested(self.analysis_results, ['information_flow'], {}):
                f.write("IV. INFORMATION FLOW & COMPLEXITY\n")
                f.write("-"*50 + "\n")
                info_flow = self.analysis_results['information_flow']
                if 'top_sources' in info_flow:
                    f.write("Top Information Drivers:\n")
                    for source, contribution in self.safe_get_nested(info_flow, ['top_sources'], []):
                        f.write(f"  - {source}: {contribution:.4f}\n")
                network = self.safe_get_nested(info_flow, ['network_analysis'], {})
                if network and 'top_information_hubs' in network:
                    f.write("Key Information Hubs:\n")
                    for hub, score in self.safe_get_nested(network, ['top_information_hubs'], [])[:2]:
                        f.write(f"  - {hub}: {score:.3f}\n")
                cmi_results = self.analysis_results.get('conditional_mutual_information', {})
                if 'top_synergies' in cmi_results:
                    f.write("Top Synergistic Pairs:\n")
                    for pair, synergy in self.safe_get_nested(cmi_results, ['top_synergies'], [])[:2]:
                        f.write(f"  - {pair}: {synergy:.4f}\n")
                if '1h' in self.analysis_results and 'pe' in self.safe_get_nested(self.analysis_results, ['1h'], {}):
                    pe = self.analysis_results['1h']['pe']
                    f.write(f"Market Complexity: {pe.get('complexity', 'N/A')} (Predictability: {pe.get('predictability', 0)*100:.1f}%)\n")

            f.write("\n\nDATA COLLECTION & ANALYSIS STATISTICS\n")
            f.write("-"*50 + "\n")
            # FIX: Success Rate Calculation
            if self.stats['total_requests'] > 0:
                success_rate = self.safe_divide(self.stats['successful_requests'], self.stats['total_requests']) * 100
                f.write(f"Total API Requests: {self.stats['total_requests']} (Success Rate: {success_rate:.1f}%)\n")
            else:
                f.write(f"Total API Requests: 0 (No requests made)\n")
            f.write(f"Total Data Points: {self.stats['data_points_collected']:,}\n")
            analysis_end_time = datetime.now(timezone.utc)
            analysis_duration = (analysis_end_time - self.current_time).total_seconds()
            f.write(f"Analysis Duration: {analysis_duration:.1f} seconds\n")

        print(f"\nSummary saved to: {self.summary_file}")
        self.logger.info(f"Summary saved to: {self.summary_file}")

    def _get_bias_label(self, bullish_pct):
        if bullish_pct > 70: return "STRONGLY BULLISH"
        elif bullish_pct > 55: return "BULLISH"
        elif bullish_pct < 30: return "STRONGLY BEARISH"
        elif bullish_pct < 45: return "BEARISH"
        else: return "NEUTRAL"
    
    def calculate_weighted_bias(self, timeframe_signals, tf_weights):
        """Consistent weighting for bias calculation."""
        weighted_bullish = 0
        weighted_total = 0
        
        for tf, signals in timeframe_signals.items():
            weight = tf_weights.get(tf, 1.0)
            weighted_bullish += signals.get('bullish', 0) * weight
            weighted_total += signals.get('total', 0) * weight
        
        return self.safe_divide(weighted_bullish, weighted_total, 0.5) * 100

    def format_prediction_with_confidence(self, prediction, confidence):
        """Add confidence context to a prediction string."""
        if confidence > 80:
            conf_text = "HIGH CONFIDENCE ⬆⬆⬆"
        elif confidence > 60:
            conf_text = "MODERATE CONFIDENCE ⬆⬆"
        else:
            conf_text = "LOW CONFIDENCE ⬆"
        
        return f"{prediction} ({conf_text} - {confidence:.1f}%)"

    def _calculate_aggregate_bias(self):
        """Helper to calculate overall market bias from all timeframe signals."""
        bullish_signals = 0
        bearish_signals = 0

        timeframe_signals = {}
        tf_weights = {'15m': 1.5, '30m': 1.25, '1h': 1.0, '4h': 0.8, '6h': 0.7, '12h': 0.6, '1d': 0.5}

        for tf in self.timeframes:
            if tf not in self.analysis_results:
                continue

            tf_bullish, tf_bearish = 0, 0
            results_tf = self.analysis_results.get(tf, {})

            if 'dfa' in results_tf:
                if self.safe_get_nested(results_tf, ['dfa', 'market_type']) == 'TRENDING': tf_bullish += 1
                elif self.safe_get_nested(results_tf, ['dfa', 'market_type']) == 'MEAN_REVERTING': tf_bearish += 1

            if 'hmm' in results_tf and 'error' not in self.safe_get_nested(results_tf, ['hmm'], {}):
                if self.safe_get_nested(results_tf, ['hmm', 'current_state']) == 'BULL': tf_bullish += 1
                elif self.safe_get_nested(results_tf, ['hmm', 'current_state']) == 'BEAR': tf_bearish += 1

            if 'ha_ms_signal' in results_tf:
                signal = self.safe_get_nested(results_tf, ['ha_ms_signal', 'ms_signal'], 'NEUTRAL')
                if 'BUY' in signal: tf_bullish += 1
                elif 'SELL' in signal: tf_bearish += 1

            timeframe_signals[tf] = {'bullish': tf_bullish, 'bearish': tf_bearish, 'total': tf_bullish + tf_bearish}
            bullish_signals += tf_bullish * tf_weights.get(tf, 1.0)
            bearish_signals += tf_bearish * tf_weights.get(tf, 1.0)

        # FIX: Division by Zero and Weighting Logic
        total_weighted_signals = bullish_signals + bearish_signals
        
        st_bull = sum(timeframe_signals.get(tf, {}).get('bullish', 0) for tf in ['15m', '30m', '1h'])
        st_total = sum(timeframe_signals.get(tf, {}).get('total', 0) for tf in ['15m', '30m', '1h'])
        lt_bull = sum(timeframe_signals.get(tf, {}).get('bullish', 0) for tf in ['4h', '6h', '12h', '1d'])
        lt_total = sum(timeframe_signals.get(tf, {}).get('total', 0) for tf in ['4h', '6h', '12h', '1d'])

        short_term_bull_pct = self.safe_divide(st_bull, st_total, 50.0) * 100
        long_term_bull_pct = self.safe_divide(lt_bull, lt_total, 50.0) * 100

        # Return weighted signals for overall bias, but also return the normalized percentage
        return bullish_signals, bearish_signals, total_weighted_signals, {
            'breakdown': timeframe_signals,
            'short_term_bull_pct': short_term_bull_pct,
            'long_term_bull_pct': long_term_bull_pct
        }
    def write_detailed_results(self):
        """Write detailed analysis results"""
        self.write_to_file("\n\n" + "="*100)
        self.write_to_file("DETAILED MATHEMATICAL ANALYSIS RESULTS")
        self.write_to_file("="*100)

        for timeframe in self.timeframes:
            if timeframe not in self.analysis_results:
                continue

            self.write_to_file(f"\n\n{'='*80}")
            self.write_to_file(f"TIMEFRAME: {timeframe}")
            self.write_to_file(f"{'='*80}")

            results = self.analysis_results[timeframe]

            for analysis_name, analysis_data in results.items():
                self.write_to_file(f"\n{analysis_name.upper()}:")
                self.write_to_file("-" * 50)

                if isinstance(analysis_data, dict):
                    self._write_dict_results(analysis_data, indent=2)
                elif isinstance(analysis_data, (list, np.ndarray)):
                    self.write_to_file(f"  Length: {len(analysis_data)}")
                    if len(analysis_data) > 0:
                        self.write_to_file(f"  Sample: {analysis_data[:5]}")
                else:
                    self.write_to_file(f"  {analysis_data}")

        cross_tf_results = ['tensor_fusion', 'information_flow', 'conditional_mutual_information', 'microstructure_bridge', 'cnn_prediction']
        for result_name in cross_tf_results:
            if result_name in self.analysis_results:
                self.write_to_file(f"\n\n{'='*80}")
                self.write_to_file(f"{result_name.upper().replace('_', ' ')} ANALYSIS")
                self.write_to_file(f"{'='*80}")
                self._write_dict_results(self.analysis_results[result_name], indent=2)

    def _write_dict_results(self, data: Dict, indent: int = 0):
        """Helper to write dictionary results with proper formatting"""
        indent_str = " " * indent

        for key, value in data.items():
            if isinstance(value, dict):
                self.write_to_file(f"{indent_str}{key}:")
                self._write_dict_results(value, indent + 2)
            elif isinstance(value, (list, np.ndarray)):
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                if len(value) > 6:
                    display_val = f"[{', '.join(map(str, value[:3]))}, ..., {', '.join(map(str, value[-3:]))}]"
                else:
                    display_val = str(value)
                self.write_to_file(f"{indent_str}{key}: [Length: {len(value)}] {display_val}")
            elif isinstance(value, (int, float, np.integer, np.floating)):
                if isinstance(value, (float, np.floating)):
                    self.write_to_file(f"{indent_str}{key}: {value:.6f}")
                else:
                    self.write_to_file(f"{indent_str}{key}: {value}")
            else:
                str_val = str(value)
                if len(str_val) > 200:
                    str_val = str_val[:200] + "..."
                self.write_to_file(f"{indent_str}{key}: {str_val}")

    def write_to_file(self, content: str):
        """Write content to results file"""
        with open(self.results_file, 'a', encoding='utf-8') as f:
            f.write(content + "\n")

    def run_complete_analysis(self):
        """Main entry point for the complete analysis"""
        try:
            self.logger.info("PHASE 1: Fetching kline data")
            print("\nPHASE 1: FETCHING KLINE DATA")
            self.fetch_klines_concurrent()
            
            self.logger.info("PHASE 2: Collecting market microstructure data")
            print("\nPHASE 2: COLLECTING MARKET MICROSTRUCTURE DATA")
            self.collect_all_market_data_by_endpoint()
            
            # Train XGBoost model if not loaded
            if self.microstructure_bridge.xgb_model is None:
                print("\nPHASE 2.5: TRAINING MICROSTRUCTURE MODEL")
                self.train_microstructure_model()
            else:
                print("\nPHASE 2.5: USING PRE-LOADED XGBOOST MODEL")
            
            self.logger.info("PHASE 3: Performing enhanced mathematical analysis")
            print("\nPHASE 3: PERFORMING ENHANCED MATHEMATICAL ANALYSIS")
            self.perform_comprehensive_analysis_enhanced()
            
            # Save models after analysis
            print("\nPHASE 4: SAVING MODELS")
            self.save_models()
            
            print("\n" + "="*100)
            print("ANALYSIS COMPLETE!")
            print(f"Results saved to: {self.results_file}")
            print(f"Summary saved to: {self.summary_file}")
            print(f"Error log saved to: {self.error_log_file}")
            print(f"Models saved to: models/")
            print("="*100)
            
        except Exception as e:
            self.logger.exception("Critical error in analysis")
            print(f"\nCRITICAL ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
        
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        self.executor.shutdown(wait=True)
        self.logger.info("Resources cleaned up")
        print("\nResources cleaned up.")
    
    def train_microstructure_model(self, lookback_periods: int = 100):
        """Collect historical data and train XGBoost model for microstructure bridge"""
        print("\n[TRAINING XGBOOST MODEL FOR MICROSTRUCTURE BRIDGE]")
        self.logger.info("Starting XGBoost model training")
        
        try:
            # Prepare training data from collected data
            training_features = []
            training_targets = []
            
            # Use 15m timeframe for training (highest resolution)
            timeframe = '15m'
            
            if timeframe not in self.kline_data or self.kline_data[timeframe].empty:
                print("  ✗ No kline data available for training")
                return False
            
            kline_df = self.kline_data[timeframe]
            
            # Need at least lookback_periods + prediction horizon
            if len(kline_df) < lookback_periods + 10:
                print(f"  ✗ Insufficient data for training (need at least {lookback_periods + 10} candles)")
                return False
            
            # Prepare orderflow data
            orderflow_df = pd.DataFrame()
            if timeframe in self.collected_data and 'order_flow' in self.collected_data[timeframe]:
                of_data = self.collected_data[timeframe]['order_flow']
                if of_data:
                    orderflow_df = pd.DataFrame(of_data)
                    if 'timestamp' in orderflow_df.columns:
                        orderflow_df['timestamp'] = pd.to_datetime(orderflow_df['timestamp'], unit='ms')
                        orderflow_df = orderflow_df.set_index('timestamp')
                        # Calculate delta
                        if 'total_bid' in orderflow_df.columns and 'total_ask' in orderflow_df.columns:
                            orderflow_df['delta'] = orderflow_df['total_bid'] - orderflow_df['total_ask']
                            orderflow_df['volume'] = orderflow_df['total_bid'] + orderflow_df['total_ask']
            
            # Prepare orderbook data
            orderbook_df = pd.DataFrame()
            if timeframe in self.collected_data and 'orderbook' in self.collected_data[timeframe]:
                ob_data = self.collected_data[timeframe]['orderbook']
                if ob_data:
                    orderbook_df = pd.DataFrame(ob_data)
                    if 'timestamp' in orderbook_df.columns:
                        orderbook_df['timestamp'] = pd.to_datetime(orderbook_df['timestamp'], unit='ms')
                        orderbook_df = orderbook_df.set_index('timestamp')
            
            # Generate training samples
            print(f"  Generating training samples from {len(kline_df)} candles...")
            
            for i in range(lookback_periods, len(kline_df) - 5):
                # Current bar info
                current_bar = {
                    'open': float(kline_df.iloc[i]['open']),
                    'high': float(kline_df.iloc[i]['high']),
                    'low': float(kline_df.iloc[i]['low']),
                    'current': float(kline_df.iloc[i]['close']),
                    'time_remaining_ratio': 0.5  # Assume midway through bar
                }
                
                # Get relevant orderflow data (if available)
                current_time = kline_df.index[i]
                next_time = kline_df.index[i + 1] if i + 1 < len(kline_df) else current_time + pd.Timedelta(minutes=15)
                
                of_slice = pd.DataFrame()
                if not orderflow_df.empty:
                    mask = (orderflow_df.index >= current_time) & (orderflow_df.index < next_time)
                    of_slice = orderflow_df[mask].copy()
                    
                    # Add price column if missing
                    if not of_slice.empty and 'price' not in of_slice.columns:
                        of_slice['price'] = current_bar['current']
                
                # Get relevant orderbook data
                ob_slice = pd.DataFrame()
                if not orderbook_df.empty:
                    mask = (orderbook_df.index >= current_time) & (orderbook_df.index < next_time)
                    ob_slice = orderbook_df[mask].copy()
                    
                    if not ob_slice.empty:
                        # Create proper format for microstructure bridge
                        ob_slice['bid_size'] = ob_slice.get('buy_usd', 0)
                        ob_slice['ask_size'] = ob_slice.get('sell_usd', 0)
                        ob_slice['bid_price'] = current_bar['current'] * 0.9995
                        ob_slice['ask_price'] = current_bar['current'] * 1.0005
                        ob_slice['mid'] = current_bar['current']
                
                # Prepare features
                features = self.microstructure_bridge._prepare_xgb_features(
                    current_bar, of_slice, ob_slice
                )
                
                if not features.empty:
                    # Target is the actual close price of the bar
                    target = float(kline_df.iloc[i]['close'])
                    
                    training_features.append(features.iloc[0].to_dict())
                    training_targets.append(target)
            
            if len(training_features) < 50:
                print(f"  ✗ Insufficient training samples ({len(training_features)} < 50)")
                return False
            
            # Convert to DataFrame
            X_train = pd.DataFrame(training_features)
            y_train = np.array(training_targets)
            
            print(f"  Training on {len(X_train)} samples...")
            
            # Train the model
            self.microstructure_bridge.train_xgboost_model(X_train, y_train)
            
            # Evaluate on last 20% of data
            split_idx = int(0.8 * len(X_train))
            X_val = X_train.iloc[split_idx:]
            y_val = y_train[split_idx:]
            
            if len(X_val) > 0:
                predictions = self.microstructure_bridge.xgb_model.predict(X_val)
                mae = np.mean(np.abs(predictions - y_val))
                mape = np.mean(np.abs((predictions - y_val) / y_val)) * 100
                
                print(f"  ✓ Model trained successfully!")
                print(f"    - Training samples: {split_idx}")
                print(f"    - Validation MAE: ${mae:.2f}")
                print(f"    - Validation MAPE: {mape:.2f}%")
                
                # Feature importance
                if hasattr(self.microstructure_bridge.xgb_model, 'feature_importances_'):
                    importances = self.microstructure_bridge.xgb_model.feature_importances_
                    feature_names = X_train.columns.tolist()
                    importance_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance': importances
                    }).sort_values('importance', ascending=False)
                    
                    print("\n  Top 5 Features:")
                    for idx, row in importance_df.head(5).iterrows():
                        print(f"    - {row['feature']}: {row['importance']:.3f}")
            
            return True
            
        except Exception as e:
            self.logger.exception(f"Error training microstructure model: {e}")
            print(f"  ✗ Training failed: {str(e)}")
            return False

    def load_models(self):
        """Load pre-trained models from disk"""
        model_dir = "models"
        
        if not os.path.exists(model_dir):
            self.logger.warning(f"Model directory '{model_dir}' not found")
            return False
        
        models_loaded = []
        
        try:
            # Load CNN model
            cnn_model_path = os.path.join(model_dir, "cnn_market_predictor.pth")
            cnn_scaler_path = os.path.join(model_dir, "cnn_scaler.pkl")
            cnn_params_path = os.path.join(model_dir, "cnn_best_params.pkl")
            
            if os.path.exists(cnn_model_path):
                # Load model state
                checkpoint = torch.load(cnn_model_path, map_location=self.cnn_predictor.device)
                
                # Initialize model with saved dimensions
                self.cnn_predictor.model = MarketPredictionCNN(
                    n_features=checkpoint['n_features'],
                    n_timeframes=checkpoint['n_timeframes'],
                    sequence_length=checkpoint['sequence_length'],
                    n_classes=checkpoint['n_classes']
                ).to(self.cnn_predictor.device)
                
                # Load model weights
                self.cnn_predictor.model.load_state_dict(checkpoint['model_state_dict'])
                self.cnn_predictor.model.eval()
                self.cnn_predictor.is_trained = True
                
                # Load scaler
                if os.path.exists(cnn_scaler_path):
                    with open(cnn_scaler_path, 'rb') as f:
                        self.cnn_predictor.scaler = pickle.load(f)
                
                # Load best parameters
                if os.path.exists(cnn_params_path):
                    with open(cnn_params_path, 'rb') as f:
                        self.cnn_predictor.best_params = pickle.load(f)
                
                models_loaded.append("CNN Market Predictor")
                self.logger.info("CNN model loaded successfully")
            
            # Load XGBoost model
            xgb_model_path = os.path.join(model_dir, "xgb_microstructure_model.pkl")
            if os.path.exists(xgb_model_path):
                with open(xgb_model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    
                    # Check if it's versioned format
                    if isinstance(model_data, dict) and 'model' in model_data:
                        self.microstructure_bridge.xgb_model = model_data['model']
                        self.logger.info(f"XGBoost model loaded (version: {model_data.get('version', 'unknown')})")
                    else:
                        # Legacy format
                        self.microstructure_bridge.xgb_model = model_data
                        self.logger.info("XGBoost model loaded (legacy format)")
                
                models_loaded.append("XGBoost Microstructure Model")
            
            # Load MS-GARCH model
            msgarch_model_path = os.path.join(model_dir, "ms_garch_model.pkl")
            if os.path.exists(msgarch_model_path):
                with open(msgarch_model_path, 'rb') as f:
                    msgarch_data = pickle.load(f)
                    self.ms_garch.hmm_model = msgarch_data['hmm_model']
                    self.ms_garch.garch_models = msgarch_data['garch_models']
                    self.ms_garch.state_params = msgarch_data['state_params']
                    self.ms_garch.scaler = msgarch_data['scaler']
                
                models_loaded.append("MS-GARCH Model")
                self.logger.info("MS-GARCH model loaded successfully")
            
            # Load Hawkes Process parameters
            hawkes_params_path = os.path.join(model_dir, "hawkes_params.pkl")
            if os.path.exists(hawkes_params_path):
                with open(hawkes_params_path, 'rb') as f:
                    hawkes_params = pickle.load(f)
                    self.hawkes_process.mu = hawkes_params['mu']
                    self.hawkes_process.alpha = hawkes_params['alpha']
                    self.hawkes_process.beta = hawkes_params['beta']
                
                models_loaded.append("Hawkes Process Parameters")
                self.logger.info("Hawkes parameters loaded successfully")
            
            # Load UKF state
            ukf_state_path = os.path.join(model_dir, "ukf_state.pkl")
            if os.path.exists(ukf_state_path):
                with open(ukf_state_path, 'rb') as f:
                    ukf_state = pickle.load(f)
                    self.ukf_filter.x = ukf_state['x']
                    self.ukf_filter.P = ukf_state['P']
                    self.ukf_filter.Q = ukf_state['Q']
                    self.ukf_filter.R = ukf_state['R']
                
                models_loaded.append("UKF State")
                self.logger.info("UKF state loaded successfully")
            
            if models_loaded:
                print(f"\n✓ Successfully loaded {len(models_loaded)} models:")
                for model in models_loaded:
                    print(f"  - {model}")
                self.logger.info(f"Total models loaded: {len(models_loaded)}")
                return True
            else:
                print("\n⚠ No saved models found in 'models/' directory")
                self.logger.warning("No models found to load")
                return False
                
        except Exception as e:
            self.logger.exception(f"Error loading models: {e}")
            print(f"\n✗ Error loading models: {str(e)}")
            print("  Will train new models during analysis")
            return False

    def save_models(self):
        """Save trained models to disk"""
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        
        saved_models = []
        
        try:
            # Save CNN model
            if self.cnn_predictor.is_trained and self.cnn_predictor.model is not None:
                # Save model state
                torch.save({
                    'model_state_dict': self.cnn_predictor.model.state_dict(),
                    'n_features': self.cnn_predictor.model.fc1.in_features // 7,
                    'n_timeframes': 7,
                    'sequence_length': 50,  # You may want to store this properly
                    'n_classes': 3
                }, os.path.join(model_dir, "cnn_market_predictor.pth"))
                
                # Save scaler
                with open(os.path.join(model_dir, "cnn_scaler.pkl"), 'wb') as f:
                    pickle.dump(self.cnn_predictor.scaler, f)
                
                # Save best parameters if available
                if self.cnn_predictor.best_params:
                    with open(os.path.join(model_dir, "cnn_best_params.pkl"), 'wb') as f:
                        pickle.dump(self.cnn_predictor.best_params, f)
                
                saved_models.append("CNN Market Predictor")
            
            # Save XGBoost model
            if self.microstructure_bridge.xgb_model is not None:
                with open(os.path.join(model_dir, "xgb_microstructure_model.pkl"), 'wb') as f:
                    model_data = {
                        'version': '1.0',
                        'model': self.microstructure_bridge.xgb_model,
                        'created_at': datetime.now().isoformat()
                    }
                    pickle.dump(model_data, f)
                saved_models.append("XGBoost Microstructure Model")
            
            # Save MS-GARCH model
            if hasattr(self.ms_garch, 'hmm_model') and self.ms_garch.hmm_model is not None:
                with open(os.path.join(model_dir, "ms_garch_model.pkl"), 'wb') as f:
                    pickle.dump({
                        'hmm_model': self.ms_garch.hmm_model,
                        'garch_models': self.ms_garch.garch_models,
                        'state_params': self.ms_garch.state_params,
                        'scaler': self.ms_garch.scaler
                    }, f)
                saved_models.append("MS-GARCH Model")
            
            # Save Hawkes parameters
            with open(os.path.join(model_dir, "hawkes_params.pkl"), 'wb') as f:
                pickle.dump({
                    'mu': self.hawkes_process.mu,
                    'alpha': self.hawkes_process.alpha,
                    'beta': self.hawkes_process.beta
                }, f)
            saved_models.append("Hawkes Process Parameters")
            
            # Save UKF state
            with open(os.path.join(model_dir, "ukf_state.pkl"), 'wb') as f:
                pickle.dump({
                    'x': self.ukf_filter.x,
                    'P': self.ukf_filter.P,
                    'Q': self.ukf_filter.Q,
                    'R': self.ukf_filter.R
                }, f)
            saved_models.append("UKF State")
            
            if saved_models:
                print(f"\n✓ Successfully saved {len(saved_models)} models:")
                for model in saved_models:
                    print(f"  - {model}")
                self.logger.info(f"Models saved: {', '.join(saved_models)}")
            
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
            print(f"\n⚠ Error saving some models: {e}")


if __name__ == "__main__":
    print(f"\n{'='*100}")
    print("INTEGRATED MATHEMATICAL ANALYSIS SYSTEM V45")
    print(f"{'='*100}\n")
    
    # Check for model loading preference
    load_models = False
    if len(sys.argv) > 1 and sys.argv[1] == '--load-models':
        load_models = True
        print("Loading existing models...")
    
    # Create analyzer instance
    analyzer = IntegratedMathematicalAnalysisSystem(load_existing_models=load_models)
    
    # Run complete analysis
    analyzer.run_complete_analysis()