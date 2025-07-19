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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import talib
from typing import Tuple, Optional, Dict, List, Any, Union
import time
import os
import traceback
from multiprocessing import Pool, cpu_count
import optuna
from functools import partial, lru_cache
import logging
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from scipy.stats import gaussian_kde
from sklearn.feature_selection import mutual_info_regression
import networkx as nx
from itertools import product
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import pdist, squareform
import zlib
import xgboost as xgb
import pickle
import sqlite3
import hashlib
from pathlib import Path
import threading
from dataclasses import dataclass, field
from enum import Enum
import numba
from numba import jit, prange
import pyarrow as pa
import pyarrow.parquet as pq

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
numba.config.THREADING_LAYER = 'omp'


class OptimizedMarketDataCache:
    """Optimized caching system using Parquet for better compression and performance."""
    
    def __init__(self, cache_dir: str = "market_data_cache", 
                 db_name: str = "cache_metadata.db",
                 compression: str = 'snappy',
                 ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.db_path = self.cache_dir / db_name
        self.compression = compression
        self.ttl_hours = ttl_hours
        self.lock = threading.RLock()
        
        self._init_database()
        self._cleanup_expired_cache()
    
    def _init_database(self):
        """Initialize SQLite database with optimized indexes"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cache_metadata (
                    cache_key TEXT PRIMARY KEY,
                    endpoint TEXT NOT NULL,
                    params TEXT NOT NULL,
                    timestamp_created INTEGER NOT NULL,
                    timestamp_expires INTEGER NOT NULL,
                    last_data_timestamp INTEGER,
                    record_count INTEGER,
                    data_file TEXT NOT NULL,
                    checksum TEXT,
                    compressed BOOLEAN,
                    file_format TEXT DEFAULT 'pickle'
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_endpoint_expires 
                ON cache_metadata(endpoint, timestamp_expires)
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS latest_timestamps (
                    data_key TEXT PRIMARY KEY,
                    endpoint TEXT NOT NULL,
                    symbol TEXT,
                    timeframe TEXT,
                    exchange TEXT,
                    product_type TEXT,
                    latest_timestamp INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                )
            ''')
    
    @lru_cache(maxsize=1024)
    def _generate_cache_key(self, endpoint: str, params_str: str) -> str:
        """Generate unique cache key with memoization"""
        return hashlib.blake2b(f"{endpoint}:{params_str}".encode(), digest_size=32).hexdigest()
    
    def _generate_data_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """Generate key for latest timestamp tracking"""
        key_parts = [endpoint]
        for param in ['symbol', 'timeframe', 'interval', 'exchange', 'productType']:
            if param in params:
                key_parts.append(str(params[param]))
        return ":".join(key_parts)
    
    def get_latest_timestamp(self, endpoint: str, params: Dict[str, Any]) -> Optional[int]:
        """Get the latest timestamp for a specific data type"""
        data_key = self._generate_data_key(endpoint, params)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT latest_timestamp FROM latest_timestamps WHERE data_key = ?",
                (data_key,)
            )
            result = cursor.fetchone()
            return result[0] if result else None
    
    def update_latest_timestamp(self, endpoint: str, params: Dict[str, Any], 
                               timestamp: int):
        """Update the latest timestamp for a specific data type"""
        data_key = self._generate_data_key(endpoint, params)
        current_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO latest_timestamps 
                (data_key, endpoint, symbol, timeframe, exchange, product_type, 
                 latest_timestamp, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data_key, endpoint,
                params.get('symbol'), params.get('timeframe') or params.get('interval'),
                params.get('exchange'), params.get('productType'),
                timestamp, current_time
            ))
    
    def get_cached_data(self, endpoint: str, params: Dict[str, Any], 
                       merge_with_new: bool = True) -> Optional[Any]:
        """Retrieve cached data with optimized file reading"""
        params_str = json.dumps(sorted(params.items()))
        cache_key = self._generate_cache_key(endpoint, params_str)
        
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    '''SELECT data_file, compressed, timestamp_expires, 
                       last_data_timestamp, file_format 
                       FROM cache_metadata WHERE cache_key = ?''',
                    (cache_key,)
                )
                result = cursor.fetchone()
                
                if not result:
                    return None
                
                data_file, compressed, expires, last_timestamp, file_format = result
                current_time = int(datetime.now(timezone.utc).timestamp() * 1000)
                
                if current_time > expires:
                    return None
                
                file_path = self.cache_dir / data_file
                if not file_path.exists():
                    return None
                
                try:
                    if file_format == 'parquet':
                        cached_data = pq.read_table(file_path).to_pandas().to_dict('records')
                    else:
                        with open(file_path, 'rb') as f:
                            data = f.read()
                        if compressed and file_format == 'pickle':
                            data = zlib.decompress(data)
                        cached_data = pickle.loads(data)
                    
                    return {
                        'data': cached_data,
                        'last_timestamp': last_timestamp,
                        'cache_key': cache_key
                    }
                except Exception as e:
                    print(f"Error loading cached data: {e}")
                    return None
    
    def save_data(self, endpoint: str, params: Dict[str, Any], 
                  data: Any, last_timestamp: Optional[int] = None) -> str:
        """Save data with optimized compression"""
        params_str = json.dumps(sorted(params.items()))
        cache_key = self._generate_cache_key(endpoint, params_str)
        
        # Determine best format based on data type
        if isinstance(data, list) and len(data) > 100 and all(isinstance(x, dict) for x in data):
            # Use Parquet for large structured data
            data_file = f"{cache_key}.parquet"
            file_path = self.cache_dir / data_file
            file_format = 'parquet'
            
            df = pd.DataFrame(data)
            table = pa.Table.from_pandas(df)
            pq.write_table(table, file_path, compression=self.compression)
            
            checksum = hashlib.md5(open(file_path, 'rb').read()).hexdigest()
            compressed = True
            record_count = len(data)
        else:
            # Use pickle for other data types
            data_file = f"{cache_key}.pkl"
            file_path = self.cache_dir / data_file
            file_format = 'pickle'
            
            serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            
            if self.compression:
                serialized = zlib.compress(serialized, level=6)
                compressed = True
            else:
                compressed = False
            
            with open(file_path, 'wb') as f:
                f.write(serialized)
            
            checksum = hashlib.md5(serialized).hexdigest()
            record_count = len(data) if isinstance(data, (list, dict)) else 0
        
        with self.lock:
            current_time = int(datetime.now(timezone.utc).timestamp() * 1000)
            expires = current_time + (self.ttl_hours * 3600 * 1000)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO cache_metadata
                    (cache_key, endpoint, params, timestamp_created, timestamp_expires,
                     last_data_timestamp, record_count, data_file, checksum, compressed, file_format)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    cache_key, endpoint, params_str, current_time, expires,
                    last_timestamp, record_count, data_file, checksum, compressed, file_format
                ))
            
            if last_timestamp:
                self.update_latest_timestamp(endpoint, params, last_timestamp)
        
        return cache_key
    
    def merge_cached_with_new(self, cached_data: List[Dict], new_data: List[Dict], 
                             timestamp_field: str = 'timestamp') -> List[Dict]:
        """Merge cached data with new data, removing duplicates"""
        new_timestamps = {record.get(timestamp_field) for record in new_data 
                         if record.get(timestamp_field)}
        
        filtered_cached = [record for record in cached_data 
                          if record.get(timestamp_field) not in new_timestamps]
        
        merged = filtered_cached + new_data
        merged.sort(key=lambda x: x.get(timestamp_field, 0))
        
        return merged
    
    def _cleanup_expired_cache(self):
        """Remove expired cache entries and files"""
        current_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT cache_key, data_file FROM cache_metadata WHERE timestamp_expires < ?",
                (current_time,)
            )
            expired = cursor.fetchall()
            
            for cache_key, data_file in expired:
                file_path = self.cache_dir / data_file
                if file_path.exists():
                    try:
                        file_path.unlink()
                    except:
                        pass
            
            conn.execute(
                "DELETE FROM cache_metadata WHERE timestamp_expires < ?",
                (current_time,)
            )
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with sqlite3.connect(self.db_path) as conn:
            stats = {}
            
            cursor = conn.execute("SELECT COUNT(*) FROM cache_metadata")
            stats['total_entries'] = cursor.fetchone()[0]
            
            cursor = conn.execute(
                "SELECT endpoint, COUNT(*) FROM cache_metadata GROUP BY endpoint"
            )
            stats['by_endpoint'] = dict(cursor.fetchall())
            
            total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.pkl"))
            total_size += sum(f.stat().st_size for f in self.cache_dir.glob("*.parquet"))
            stats['total_size_mb'] = total_size / (1024 * 1024)
            
            cursor = conn.execute(
                "SELECT data_key, latest_timestamp FROM latest_timestamps ORDER BY updated_at DESC LIMIT 10"
            )
            stats['recent_updates'] = cursor.fetchall()
            
            return stats
    
    def clear_cache(self, endpoint: Optional[str] = None):
        """Clear cache completely or for specific endpoint"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                if endpoint:
                    cursor = conn.execute(
                        "SELECT data_file FROM cache_metadata WHERE endpoint = ?",
                        (endpoint,)
                    )
                    files = cursor.fetchall()
                    
                    for (data_file,) in files:
                        file_path = self.cache_dir / data_file
                        if file_path.exists():
                            file_path.unlink()
                    
                    conn.execute(
                        "DELETE FROM cache_metadata WHERE endpoint = ?",
                        (endpoint,)
                    )
                else:
                    for file_path in self.cache_dir.glob("*.pkl"):
                        file_path.unlink()
                    for file_path in self.cache_dir.glob("*.parquet"):
                        file_path.unlink()
                    
                    conn.execute("DELETE FROM cache_metadata")
                    conn.execute("DELETE FROM latest_timestamps")


class OptimizedPureNumpyTensorDecomposition:
    """Optimized tensor decomposition using Numba JIT compilation"""
    
    def __init__(self, rank: int = 10, max_iter: int = 100, tol: float = 1e-6):
        self.rank = rank
        self.max_iter = max_iter
        self.tol = tol
    
    @staticmethod
    @jit(nopython=True, parallel=True, cache=True)
    def _compute_khatri_rao_numba(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Numba-optimized Khatri-Rao product"""
        I, R1 = A.shape
        J, R2 = B.shape
        
        if R1 != R2:
            raise ValueError("Matrices must have same number of columns")
        
        result = np.zeros((I * J, R1))
        
        for r in prange(R1):
            for i in prange(I):
                for j in prange(J):
                    result[i * J + j, r] = A[i, r] * B[j, r]
        
        return result
    
    def cp_als(self, tensor: np.ndarray) -> Dict[str, Any]:
        """Optimized CP decomposition using ALS with Numba acceleration"""
        shape = tensor.shape
        ndims = len(shape)
        
        # Initialize factor matrices with SVD
        factors = []
        for i in range(ndims):
            unfolded = self._unfold_tensor(tensor, i)
            try:
                U, S, Vt = svds(unfolded, k=min(self.rank, min(unfolded.shape) - 1))
                factor = U @ np.diag(S)
                factor = factor[:, :self.rank]
                
                if factor.shape[1] < self.rank:
                    padding = np.random.randn(shape[i], self.rank - factor.shape[1])
                    factor = np.hstack([factor, padding])
                
            except:
                factor = np.random.randn(shape[i], self.rank)
            
            factor = factor / (np.linalg.norm(factor, axis=0) + 1e-10)
            factors.append(factor)
        
        errors = []
        
        for iteration in range(self.max_iter):
            old_factors = [f.copy() for f in factors]
            
            # Update each factor matrix
            for mode in range(ndims):
                unfolded = self._unfold_tensor(tensor, mode)
                kr_product = self._khatri_rao_product_optimized(factors, skip_index=mode)
                
                try:
                    # Cholesky decomposition for faster solve
                    gramian = kr_product.T @ kr_product
                    gramian += 1e-6 * np.eye(self.rank)
                    L = np.linalg.cholesky(gramian)
                    y = np.linalg.solve(L, kr_product.T @ unfolded.T)
                    factors[mode] = np.linalg.solve(L.T, y).T
                except:
                    factors[mode] = unfolded @ kr_product @ np.linalg.pinv(
                        kr_product.T @ kr_product + 1e-6 * np.eye(self.rank)
                    )
                
                # Normalize
                norms = np.linalg.norm(factors[mode], axis=0)
                factors[mode] = factors[mode] / (norms + 1e-10)
            
            # Check convergence
            factor_change = sum(
                np.linalg.norm(factors[i] - old_factors[i]) 
                for i in range(ndims)
            )
            
            if factor_change < self.tol:
                break
            
            # Calculate error every 10 iterations
            if iteration % 10 == 0:
                reconstructed = self._reconstruct_tensor_optimized(factors)
                error = np.linalg.norm(tensor - reconstructed) / (np.linalg.norm(tensor) + 1e-10)
                errors.append(error)
        
        # Final reconstruction error
        reconstructed = self._reconstruct_tensor_optimized(factors)
        final_error = np.linalg.norm(tensor - reconstructed) / (np.linalg.norm(tensor) + 1e-10)
        
        return {
            'factors': factors,
            'weights': np.ones(self.rank),
            'reconstruction_error': final_error,
            'iterations': iteration + 1,
            'convergence_history': errors
        }
    
    def _unfold_tensor(self, tensor: np.ndarray, mode: int) -> np.ndarray:
        """Optimized tensor unfolding"""
        n_modes = len(tensor.shape)
        new_order = [mode] + [i for i in range(n_modes) if i != mode]
        unfolded = np.transpose(tensor, new_order)
        return unfolded.reshape(tensor.shape[mode], -1)
    
    def _khatri_rao_product_optimized(self, matrices: List[np.ndarray], 
                                     skip_index: int = None) -> np.ndarray:
        """Optimized Khatri-Rao product computation"""
        if skip_index is not None:
            matrices = [m for i, m in enumerate(matrices) if i != skip_index]
        
        if not matrices:
            return np.array([[1.0]])
        
        result = matrices[-1]
        for matrix in reversed(matrices[:-1]):
            result = self._compute_khatri_rao_numba(matrix, result)
        
        return result
    
    def _reconstruct_tensor_optimized(self, factors: List[np.ndarray]) -> np.ndarray:
        """Optimized tensor reconstruction using einsum"""
        rank = factors[0].shape[1]
        ndims = len(factors)
        
        # Limit indices to available letters (max 10 dimensions)
        if ndims > 10:
            # Fall back to slower method for high dimensions
            return self._reconstruct_tensor_fallback(factors)
        
        indices = 'abcdefghij'[:ndims]
        
        reconstructed = np.zeros([f.shape[0] for f in factors])
        
        for r in range(rank):
            factor_vecs = [f[:, r] for f in factors]
            
            # Build einsum string dynamically
            if ndims == 2:
                reconstructed += np.einsum('i,j->ij', factor_vecs[0], factor_vecs[1])
            elif ndims == 3:
                reconstructed += np.einsum('i,j,k->ijk', factor_vecs[0], factor_vecs[1], factor_vecs[2])
            elif ndims == 4:
                reconstructed += np.einsum('i,j,k,l->ijkl', *factor_vecs)
            else:
                # General case - build string dynamically
                einsum_str = ','.join([indices[i] for i in range(ndims)]) + '->' + indices
                reconstructed += np.einsum(einsum_str, *factor_vecs)
        
        return reconstructed
    
    def _reconstruct_tensor_fallback(self, factors: List[np.ndarray]) -> np.ndarray:
        """Fallback reconstruction for high-dimensional tensors"""
        rank = factors[0].shape[1]
        reconstructed = np.zeros([f.shape[0] for f in factors])
        
        for r in range(rank):
            rank_1_tensor = factors[0][:, r]
            for factor in factors[1:]:
                rank_1_tensor = np.multiply.outer(rank_1_tensor, factor[:, r])
            reconstructed += rank_1_tensor
        
        return reconstructed

    def _reconstruct_tensor(self, factors: List[np.ndarray]) -> np.ndarray:
        """Backward compatibility wrapper"""
        return self._reconstruct_tensor_optimized(factors)

class OptimizedSimplifiedTensorAnalysis:
    """Optimized tensor analysis with parallel processing"""
    
    def __init__(self, lookback_size: int = 200):
        self.lookback_size = lookback_size
        self.decomposer = OptimizedPureNumpyTensorDecomposition(rank=10)
        self.tensor_dimensions = {
            'time': lookback_size,
            'features': 10,
            'timeframes': 7
        }
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def _calculate_rsi_numba(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Numba-optimized RSI calculation"""
        n = len(prices)
        rsi = np.zeros(n)
        
        if n < period + 1:
            return rsi
        
        deltas = np.diff(prices)
        seed = deltas[:period]
        
        up = 0.0
        down = 0.0
        for i in range(period):
            if seed[i] > 0:
                up += seed[i]
            else:
                down -= seed[i]
        
        up /= period
        down /= period
        
        if down == 0:
            rsi[period] = 100.0
        else:
            rs = up / down
            rsi[period] = 100.0 - 100.0 / (1.0 + rs)
        
        rsi[:period] = 50.0
        
        for i in range(period + 1, n):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.0
            else:
                upval = 0.0
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            
            if down == 0:
                rsi[i] = 100.0
            else:
                rs = up / down
                rsi[i] = 100.0 - 100.0 / (1.0 + rs)
        
        return rsi
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def _calculate_momentum_numba(prices: np.ndarray, period: int = 10) -> np.ndarray:
        """Numba-optimized momentum calculation"""
        n = len(prices)
        momentum = np.zeros(n)
        
        if n > period:
            for i in range(period, n):
                momentum[i] = prices[i] - prices[i - period]
        
        return momentum
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def _calculate_volatility_numba(prices: np.ndarray, period: int = 20) -> np.ndarray:
        """Numba-optimized volatility calculation"""
        n = len(prices)
        volatility = np.zeros(n)
        
        if n < 2:
            return volatility
        
        log_prices = np.log(prices + 1e-10)
        returns = np.diff(log_prices)
        
        for i in range(period, n):
            window_returns = returns[max(0, i-period):i]
            if len(window_returns) > 0:
                volatility[i] = np.std(window_returns)
        
        return volatility
    
    def create_simplified_tensor(self, data: Dict[str, pd.DataFrame]) -> np.ndarray:
        """Create tensor with optimized feature extraction"""
        n_features = self.tensor_dimensions['features']
        timeframes = ['15m', '30m', '1h', '4h', '6h', '12h', '1d']
        tensor = np.zeros((self.lookback_size, n_features, len(timeframes)))
        
        # Process timeframes in parallel
        with ThreadPoolExecutor(max_workers=len(timeframes)) as executor:
            futures = {}
            for tf_idx, timeframe in enumerate(timeframes):
                if timeframe in data:
                    futures[executor.submit(self._process_timeframe_features, 
                                          data[timeframe], n_features)] = tf_idx
            
            for future in as_completed(futures):
                tf_idx = futures[future]
                try:
                    features = future.result()
                    if features is not None:
                        n_samples = min(features.shape[0], self.lookback_size)
                        tensor[:n_samples, :, tf_idx] = features[:n_samples]
                except Exception as e:
                    print(f"Error processing timeframe {timeframes[tf_idx]}: {e}")
        
        return tensor
    
    def _process_timeframe_features(self, df: pd.DataFrame, n_features: int) -> Optional[np.ndarray]:
        """Extract features for a single timeframe with optimization"""
        if df.empty:
            return None
        
        n_samples = min(len(df), self.lookback_size)
        features = np.zeros((n_samples, n_features))
        
        # Price features
        features[:, 0] = df['close'].iloc[:n_samples].values
        features[:, 1] = df['high'].iloc[:n_samples].values
        features[:, 2] = df['low'].iloc[:n_samples].values
        features[:, 3] = df['volume'].iloc[:n_samples].values
        
        # Technical features using Numba
        if n_samples > 20:
            features[:, 4] = self._calculate_rsi_numba(df['close'].iloc[:n_samples].values)
            features[:, 5] = self._calculate_momentum_numba(df['close'].iloc[:n_samples].values)
            features[:, 6] = self._calculate_volatility_numba(df['close'].iloc[:n_samples].values)
            
            # Additional features
            features[:, 7] = (df['high'].iloc[:n_samples] - df['low'].iloc[:n_samples]).values
            features[:, 8] = (df['close'].iloc[:n_samples] - df['open'].iloc[:n_samples]).values
            features[:, 9] = df['volume'].iloc[:n_samples].rolling(10).mean().fillna(0).values
        
        # Normalize features
        means = np.mean(features, axis=0)
        stds = np.std(features, axis=0)
        mask = stds > 0
        features[:, mask] = (features[:, mask] - means[mask]) / stds[mask]
        
        return features
    
    def analyze_tensor(self, tensor: np.ndarray) -> Dict[str, Any]:
        """Perform tensor analysis using optimized methods"""
        # CP decomposition
        cp_result = self.decomposer.cp_als(tensor)
        
        # Extract temporal patterns
        temporal_factors = cp_result['factors'][0]
        
        # Calculate predictions
        predictions = {}
        for i in range(min(temporal_factors.shape[1], 5)):
            if len(temporal_factors[:, i]) > 20:
                x = np.arange(20)
                y = temporal_factors[-20:, i]
                try:
                    trend = np.polyfit(x, y, 1)[0]
                    predictions[f'factor_{i}_trend'] = trend
                except:
                    predictions[f'factor_{i}_trend'] = 0.0
        
        # Energy conservation check
        reconstructed_energy = np.sum(self.decomposer._reconstruct_tensor_optimized(cp_result['factors']) ** 2)
        original_energy = np.sum(tensor ** 2)
        energy_conservation_ratio = reconstructed_energy / original_energy if original_energy > 0 else 1.0
        
        return {
            'decomposition': cp_result,
            'temporal_patterns': temporal_factors,
            'predictions': predictions,
            'reconstruction_error': cp_result['reconstruction_error'],
            'energy_conservation_ratio': energy_conservation_ratio
        }
class OptimizedAdvancedMathematicalTools:
    """Optimized mathematical tools with parallel processing and caching"""
    
    def __init__(self):
        self._cache = {}
    
    def cointegration_and_vecm(self, data: pd.DataFrame, lag: int = 1) -> Dict[str, Any]:
        """Optimized cointegration test and VECM fitting"""
        if len(data) < 50:
            return {'error': 'Insufficient data for VECM analysis.'}
        
        # Ensure stationarity
        stationary_data = data.copy()
        
        # Parallel stationarity testing
        with ThreadPoolExecutor(max_workers=len(data.columns)) as executor:
            futures = {}
            for col in data.columns:
                futures[executor.submit(self._test_and_difference, 
                                      data[col])] = col
            
            for future in as_completed(futures):
                col = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        stationary_data[col] = result
                except Exception as e:
                    return {'error': f'Stationarity test failed for {col}: {str(e)}'}
        
        stationary_data = stationary_data.dropna()
        
        if len(stationary_data) < 20:
            return {'error': 'Insufficient data after differencing.'}
        
        try:
            # Cointegration test
            coint_result = coint(stationary_data.iloc[:, 0], stationary_data.iloc[:, 1])
            
            # Fit VECM
            model = VECM(stationary_data, k_ar_diff=lag, coint_rank=1, deterministic='c')
            vecm_res = model.fit()
            
            return {
                'coint_t_stat': float(coint_result[0]),
                'coint_p_value': float(coint_result[1]),
                'is_cointegrated': coint_result[1] < 0.05,
                'vecm_summary': str(vecm_res.summary()),
                'alpha': vecm_res.alpha.tolist(),
                'beta': vecm_res.beta.tolist(),
                'gamma': vecm_res.gamma.tolist() if hasattr(vecm_res, 'gamma') else None
            }
        except Exception as e:
            return {'error': f'VECM fitting failed: {str(e)}'}
    
    def _test_and_difference(self, series: pd.Series) -> Optional[pd.Series]:
        """Test for stationarity and difference if needed"""
        if series.nunique() <= 1:
            return None
        
        try:
            adf_result = adfuller(series.dropna())
            if adf_result[1] > 0.05:
                return series.diff()
            return series
        except:
            return None
    
    @staticmethod
    @jit(nopython=True, parallel=True, cache=True)
    def _mfdfa_fluctuation_numba(y: np.ndarray, scales: np.ndarray, 
                                 q_range: np.ndarray) -> np.ndarray:
        """Numba-optimized MF-DFA fluctuation calculation"""
        n_scales = len(scales)
        n_q = len(q_range)
        Fq = np.zeros((n_q, n_scales))
        
        for scale_idx in prange(n_scales):
            scale = scales[scale_idx]
            n_segments = len(y) // scale
            
            if n_segments == 0:
                continue
            
            fluctuations = np.zeros(n_segments * 2)
            fluct_count = 0
            
            # Forward direction
            for seg in range(n_segments):
                start = seg * scale
                end = (seg + 1) * scale
                segment = y[start:end]
                
                # Linear detrending
                x = np.arange(scale, dtype=np.float64)
                x_mean = np.mean(x)
                y_mean = np.mean(segment)
                
                xy_cov = 0.0
                xx_var = 0.0
                for i in range(scale):
                    xy_cov += (x[i] - x_mean) * (segment[i] - y_mean)
                    xx_var += (x[i] - x_mean) ** 2
                
                if xx_var > 0:
                    slope = xy_cov / xx_var
                    intercept = y_mean - slope * x_mean
                    
                    fluct = 0.0
                    for i in range(scale):
                        residual = segment[i] - (slope * x[i] + intercept)
                        fluct += residual ** 2
                    
                    fluctuations[fluct_count] = np.sqrt(fluct / scale)
                    fluct_count += 1
            
            # Backward direction
            for seg in range(n_segments):
                start = len(y) - (seg + 1) * scale
                end = len(y) - seg * scale
                
                if start >= 0:
                    segment = y[start:end]
                    
                    x = np.arange(scale, dtype=np.float64)
                    x_mean = np.mean(x)
                    y_mean = np.mean(segment)
                    
                    xy_cov = 0.0
                    xx_var = 0.0
                    for i in range(scale):
                        xy_cov += (x[i] - x_mean) * (segment[i] - y_mean)
                        xx_var += (x[i] - x_mean) ** 2
                    
                    if xx_var > 0:
                        slope = xy_cov / xx_var
                        intercept = y_mean - slope * x_mean
                        
                        fluct = 0.0
                        for i in range(scale):
                            residual = segment[i] - (slope * x[i] + intercept)
                            fluct += residual ** 2
                        
                        fluctuations[fluct_count] = np.sqrt(fluct / scale)
                        fluct_count += 1
            
            # q-order fluctuation function
            if fluct_count > 0:
                active_flucts = fluctuations[:fluct_count]
                
                for q_idx in range(n_q):
                    q = q_range[q_idx]
                    if q == 0:
                        log_sum = 0.0
                        for i in range(fluct_count):
                            if active_flucts[i] > 0:
                                log_sum += np.log(active_flucts[i] ** 2)
                        Fq[q_idx, scale_idx] = np.exp(0.5 * log_sum / fluct_count)
                    else:
                        power_sum = 0.0
                        for i in range(fluct_count):
                            power_sum += active_flucts[i] ** q
                        Fq[q_idx, scale_idx] = (power_sum / fluct_count) ** (1.0 / q)
        
        return Fq
    
    def multifractal_spectrum_analysis(self, signal: np.ndarray, 
                                     q_range: np.ndarray = None,
                                     min_scale: int = 4,
                                     max_scale: int = None) -> Dict[str, Any]:
        """Optimized multifractal detrended fluctuation analysis"""
        if q_range is None:
            q_range = np.linspace(-5, 5, 21)
        
        if max_scale is None:
            max_scale = len(signal) // 4
        
        # Cumulative sum
        y = np.cumsum(signal - np.mean(signal))
        
        # Scales
        scales = np.unique(np.logspace(np.log10(min_scale), 
                                      np.log10(max_scale), 20).astype(int))
        
        # Use Numba-optimized fluctuation calculation
        Fq = self._mfdfa_fluctuation_numba(y, scales, q_range)
        
        # Calculate Hurst exponents
        hurst_q = np.zeros(len(q_range))
        for q_idx in range(len(q_range)):
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
        
        # Width of spectrum
        alpha_range = np.max(alpha) - np.min(alpha)
        
        return {
            'q_range': q_range.tolist(),
            'hurst_q': hurst_q.tolist(),
            'tau_q': tau_q.tolist(),
            'alpha': alpha.tolist(),
            'f_alpha': f_alpha.tolist(),
            'multifractality': float(alpha_range),
            'dominant_hurst': float(hurst_q[len(q_range) // 2]),
            'scales': scales.tolist(),
            'Fq': Fq.tolist()
        }
    
    def kyle_lambda_estimation(self, price_changes: np.ndarray, 
                             signed_volume: np.ndarray,
                             method: str = 'robust') -> Dict[str, float]:
        """Optimized Kyle Lambda estimation"""
        # Remove NaN values
        mask = ~(np.isnan(price_changes) | np.isnan(signed_volume))
        price_changes = price_changes[mask]
        signed_volume = signed_volume[mask]
        
        if len(price_changes) < 10:
            return {
                'lambda': 0.0, 'r_squared': 0.0, 'market_depth': np.inf,
                'avg_trade_size': 0.0, 'price_volatility': 0.0,
                'amihud_illiquidity': 0.0, 'price_impact_per_unit': 0.0
            }
        
        if method == 'regression':
            # OLS regression
            X = signed_volume.reshape(-1, 1)
            y = price_changes
            
            XtX = X.T @ X + 1e-10
            Xty = X.T @ y
            lambda_estimate = float(Xty / XtX)
            
            # R-squared
            y_pred = lambda_estimate * signed_volume
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / (ss_tot + 1e-10))
            
        elif method == 'robust':
            # Robust estimation using Huber
            from sklearn.linear_model import HuberRegressor
            
            try:
                huber = HuberRegressor(epsilon=1.35, max_iter=100)
                X = signed_volume.reshape(-1, 1)
                huber.fit(X, price_changes)
                lambda_estimate = float(huber.coef_[0])
                r_squared = huber.score(X, price_changes)
            except:
                # Fallback to median
                ratios = price_changes / (signed_volume + 1e-10)
                finite_ratios = ratios[np.isfinite(ratios)]
                lambda_estimate = np.median(finite_ratios) if len(finite_ratios) > 0 else 0.0
                
                y_pred = lambda_estimate * signed_volume
                r_squared = np.corrcoef(price_changes, y_pred)[0, 1] ** 2 if len(price_changes) > 1 else 0.0
        
        # Market metrics
        market_depth = 1 / (abs(lambda_estimate) + 1e-10)
        avg_trade_size = np.mean(np.abs(signed_volume))
        price_volatility = np.std(price_changes)
        amihud = np.mean(np.abs(price_changes) / (np.abs(signed_volume) + 1e-10))
        
        return {
            'lambda': lambda_estimate,
            'r_squared': float(r_squared),
            'market_depth': float(market_depth),
            'avg_trade_size': float(avg_trade_size),
            'price_volatility': float(price_volatility),
            'amihud_illiquidity': float(amihud),
            'price_impact_per_unit': float(abs(lambda_estimate))
        }
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def _count_diagonal_lines_numba(recurrence_matrix: np.ndarray, 
                                   min_length: int = 2) -> Tuple[List[int], int]:
        """Numba-optimized diagonal line counting for RQA"""
        n = recurrence_matrix.shape[0]
        diagonals = []
        max_line = 0
        
        # Check all diagonals
        for k in range(1 - n, n):
            if k < 0:
                i_start = -k
                j_start = 0
                length = min(n + k, n)
            else:
                i_start = 0
                j_start = k
                length = min(n - k, n)
            
            # Find consecutive 1s
            line_length = 0
            for idx in range(length):
                i = i_start + idx
                j = j_start + idx
                
                if recurrence_matrix[i, j] == 1:
                    line_length += 1
                else:
                    if line_length >= min_length:
                        diagonals.append(line_length)
                        max_line = max(max_line, line_length)
                    line_length = 0
            
            if line_length >= min_length:
                diagonals.append(line_length)
                max_line = max(max_line, line_length)
        
        return diagonals, max_line
    
    def recurrence_quantification_analysis(self, signal: np.ndarray,
                                         embedding_dim: int = 3,
                                         time_delay: int = 1,
                                         threshold: float = None) -> Dict[str, float]:
        """Optimized RQA with Numba acceleration"""
        n = len(signal)
        m = embedding_dim
        tau = time_delay
        
        N = n - (m - 1) * tau
        if N <= 0:
            return {
                'recurrence_rate': 0.0, 'determinism': 0.0,
                'laminarity': 0.0, 'max_line': 0,
                'entropy': 0.0, 'trapping_time': 0.0
            }
        
        # Phase space reconstruction
        idx = np.arange(N)[:, None] + np.arange(m)[None, :] * tau
        embedded = signal[idx]
        
        # Distance matrix
        from scipy.spatial.distance import cdist
        dist_matrix = cdist(embedded, embedded, metric='euclidean')
        
        # Threshold selection
        if threshold is None:
            threshold = 0.1 * np.max(dist_matrix)
        
        # Recurrence matrix
        recurrence_matrix = (dist_matrix < threshold).astype(np.int32)
        
        # RQA measures
        total_points = N * N
        recurrence_points = np.sum(recurrence_matrix)
        RR = recurrence_points / total_points
        
        # Diagonal lines
        diagonals, max_line = self._count_diagonal_lines_numba(recurrence_matrix)
        
        if diagonals:
            total_diagonal_points = sum(diagonals)
            DET = total_diagonal_points / recurrence_points if recurrence_points > 0 else 0.0
            
            # Entropy
            diagonal_hist, _ = np.histogram(diagonals, 
                                          bins=range(2, max(diagonals) + 2))
            p = diagonal_hist / np.sum(diagonal_hist)
            p = p[p > 0]
            entropy_diag = -np.sum(p * np.log(p)) if len(p) > 0 else 0.0
        else:
            DET = 0.0
            max_line = 0
            entropy_diag = 0.0
        
        # Laminarity (vertical lines)
        verticals = []
        for i in range(N):
            column = recurrence_matrix[:, i]
            changes = np.diff(np.concatenate([[0], column, [0]]))
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]
            lengths = ends - starts
            verticals.extend(lengths[lengths >= 2])
        
        if verticals:
            LAM = sum(verticals) / recurrence_points if recurrence_points > 0 else 0.0
            trapping_time = np.mean(verticals)
        else:
            LAM = 0.0
            trapping_time = 0.0
        
        return {
            'recurrence_rate': float(RR),
            'determinism': float(DET),
            'laminarity': float(LAM),
            'max_line': int(max_line),
            'entropy': float(entropy_diag),
            'trapping_time': float(trapping_time),
            'threshold': float(threshold)
        }
    
    def transfer_entropy_spectrum(self, source: np.ndarray, 
                                target: np.ndarray,
                                lag_range: range = None,
                                bins: int = 5) -> Dict[str, Any]:
        """Optimized transfer entropy calculation"""
        if lag_range is None:
            lag_range = range(1, min(30, len(source) // 10))
        
        # Discretize signals
        source_disc = pd.qcut(source, q=bins, labels=False, duplicates='drop')
        target_disc = pd.qcut(target, q=bins, labels=False, duplicates='drop')
        
        # Parallel computation
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            for lag in lag_range:
                if lag >= len(source) - 1:
                    continue
                
                futures[executor.submit(self._calculate_te_for_lag, 
                                      source_disc, target_disc, lag)] = lag
            
            te_values = []
            lags_computed = []
            
            for future in as_completed(futures):
                lag = futures[future]
                try:
                    te = future.result()
                    te_values.append(te)
                    lags_computed.append(lag)
                except:
                    te_values.append(0.0)
                    lags_computed.append(lag)
        
        # Sort by lag
        sorted_pairs = sorted(zip(lags_computed, te_values))
        lags_computed, te_values = zip(*sorted_pairs) if sorted_pairs else ([], [])
        
        te_array = np.array(te_values)
        
        # Find optimal lag
        if len(te_array) > 0 and np.max(te_array) > 0:
            optimal_lag = lags_computed[np.argmax(te_array)]
            max_te = np.max(te_array)
            
            threshold = np.mean(te_array) + np.std(te_array)
            significant_lags = [lag for lag, te in zip(lags_computed, te_array) 
                              if te > threshold]
        else:
            optimal_lag = 1
            max_te = 0.0
            significant_lags = []
        
        cumulative_te = np.cumsum(te_array)
        
        return {
            'lag_range': list(lags_computed),
            'te_values': te_array.tolist(),
            'optimal_lag': int(optimal_lag),
            'max_te': float(max_te),
            'significant_lags': significant_lags,
            'cumulative_te': cumulative_te.tolist(),
            'mean_te': float(np.mean(te_array)),
            'total_information_transfer': float(np.sum(te_array))
        }
    
    def _calculate_te_for_lag(self, source_disc: np.ndarray, 
                            target_disc: np.ndarray, lag: int) -> float:
        """Calculate transfer entropy for a specific lag"""
        min_len = min(len(source_disc) - lag, len(target_disc) - lag)
        if min_len < 10:
            return 0.0
        
        y_future = target_disc[lag:lag + min_len]
        y_past = target_disc[:min_len]
        x_past = source_disc[:min_len]
        
        return self._calculate_transfer_entropy(x_past, y_past, y_future)
    
    def _calculate_transfer_entropy(self, x_past: np.ndarray, 
                                  y_past: np.ndarray, 
                                  y_future: np.ndarray) -> float:
        """Optimized transfer entropy calculation"""
        # Create joint arrays
        xyz = np.column_stack([y_future, y_past, x_past])
        yz = np.column_stack([y_future, y_past])
        yx = np.column_stack([y_past, x_past])
        y = y_past
        
        # Calculate probabilities
        def get_probs(data):
            if data.ndim == 1:
                counts = pd.Series(data).value_counts(normalize=True)
            else:
                tuples = [tuple(row) for row in data]
                counts = pd.Series(tuples).value_counts(normalize=True)
            return counts.values
        
        p_xyz = get_probs(xyz)
        p_yz = get_probs(yz)
        p_yx = get_probs(yx)
        p_y = get_probs(y)
        
        # Calculate entropies
        h_xyz = -np.sum(p_xyz * np.log2(p_xyz + 1e-12))
        h_yz = -np.sum(p_yz * np.log2(p_yz + 1e-12))
        h_yx = -np.sum(p_yx * np.log2(p_yx + 1e-12))
        h_y = -np.sum(p_y * np.log2(p_y + 1e-12))
        
        # Transfer entropy
        te = h_yz + h_yx - h_y - h_xyz
        
        return max(0, te)
    

    def _get_probs(self, data: np.ndarray) -> np.ndarray:
        """Helper to get probabilities of unique rows/elements."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Use pandas for efficient unique row counting
        df = pd.DataFrame(data)
        counts = df.value_counts(normalize=True)
        return counts.values
    @staticmethod
    @jit(nopython=True, cache=True)
    def _sample_entropy_numba(signal: np.ndarray, m: int, r: float) -> Tuple[float, float]:
        """Numba-optimized sample entropy calculation"""
        N = len(signal)
        
        if N < m + 1:
            return 0.0, 0.0
        
        # Count template matches for length m
        B = 0
        for i in range(N - m):
            template_i = signal[i:i + m]
            for j in range(N - m):
                if i != j:
                    template_j = signal[j:j + m]
                    max_dist = 0.0
                    for k in range(m):
                        dist = abs(template_i[k] - template_j[k])
                        if dist > max_dist:
                            max_dist = dist
                    
                    if max_dist <= r:
                        B += 1
        
        # Count template matches for length m+1
        A = 0
        for i in range(N - m - 1):
            template_i = signal[i:i + m + 1]
            for j in range(N - m - 1):
                if i != j:
                    template_j = signal[j:j + m + 1]
                    max_dist = 0.0
                    for k in range(m + 1):
                        dist = abs(template_i[k] - template_j[k])
                        if dist > max_dist:
                            max_dist = dist
                    
                    if max_dist <= r:
                        A += 1
        
        # Normalize
        B = B / ((N - m) * (N - m - 1)) if (N - m - 1) > 0 else 0
        A = A / ((N - m) * (N - m - 1)) if (N - m - 1) > 0 else 0
        
        return A, B
    
    def sample_entropy(self, signal: np.ndarray, 
                      m: int = 2, 
                      r: float = None,
                      normalize: bool = True) -> Dict[str, float]:
        """Optimized sample entropy calculation"""
        N = len(signal)
        
        if N < m + 1:
            return {
                'sample_entropy': 0.0,
                'threshold': 0.0,
                'A_matches': 0,
                'B_matches': 0,
                'embedding_dimension': m
            }
        
        # Normalize signal
        if normalize:
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
        
        # Set threshold
        if r is None:
            r = 0.2 * np.std(signal, ddof=0)
        
        # Use Numba-optimized calculation
        A, B = self._sample_entropy_numba(signal, m, r)
        
        # Sample entropy
        if A > 0 and B > 0:
            sampen = -np.log(A / B)
        else:
            sampen = np.inf
        
        return {
            'sample_entropy': float(sampen),
            'threshold': float(r),
            'A_matches': float(A),
            'B_matches': float(B),
            'embedding_dimension': int(m)
        }


class OptimizedTopologicalDataAnalysis:
    """Optimized TDA with parallel processing"""
    
    def __init__(self, n_intervals: int = 10, overlap_pct: float = 0.3, 
                 eps: float = 0.5, min_samples: int = 5):
        self.n_intervals = n_intervals
        self.overlap_pct = overlap_pct
        self.clustering_algo = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        self.scaler = MinMaxScaler()
    
    def analyze_market_shape(self, data: pd.DataFrame, filter_col: str, 
                           feature_cols: List[str]) -> Dict[str, Any]:
        """Optimized market shape analysis"""
        if data.empty or len(data) < 50 or filter_col not in data.columns or \
           not all(c in data.columns for c in feature_cols):
            return self._default_tda_results("Insufficient or invalid data for TDA.")
        
        # Scale features
        features = self.scaler.fit_transform(data[feature_cols].values)
        filter_values = data[filter_col].values
        
        # Create intervals
        interval_endpoints = np.linspace(np.min(filter_values), 
                                       np.max(filter_values), 
                                       self.n_intervals + 1)
        interval_length = interval_endpoints[1] - interval_endpoints[0]
        overlap = interval_length * self.overlap_pct
        
        G = nx.Graph()
        node_id_counter = 0
        nodes_by_interval = []
        
        # Parallel clustering
        with ThreadPoolExecutor(max_workers=min(self.n_intervals, 4)) as executor:
            futures = {}
            
            for i in range(self.n_intervals):
                min_val = interval_endpoints[i] - (overlap / 2 if i > 0 else 0)
                max_val = interval_endpoints[i+1] + (overlap / 2 if i < self.n_intervals - 1 else 0)
                
                bin_mask = (filter_values >= min_val) & (filter_values < max_val)
                if np.sum(bin_mask) >= self.clustering_algo.min_samples:
                    futures[executor.submit(self._cluster_interval, 
                                          features[bin_mask], 
                                          filter_values[bin_mask],
                                          data.index[bin_mask],
                                          i)] = i
            
            # Process results
            interval_results = {}
            for future in as_completed(futures):
                i = futures[future]
                try:
                    interval_results[i] = future.result()
                except Exception as e:
                    print(f"Clustering failed in interval {i}: {e}")
                    interval_results[i] = []
        
        # Build graph
        for i in range(self.n_intervals):
            interval_nodes = []
            
            if i in interval_results:
                for cluster_data in interval_results[i]:
                    G.add_node(node_id_counter, **cluster_data)
                    interval_nodes.append(node_id_counter)
                    node_id_counter += 1
            
            nodes_by_interval.append(interval_nodes)
        
        # Connect nodes
        for i in range(self.n_intervals - 1):
            for node1 in nodes_by_interval[i]:
                for node2 in nodes_by_interval[i+1]:
                    points1 = set(G.nodes[node1]['points'])
                    points2 = set(G.nodes[node2]['points'])
                    intersection = points1.intersection(points2)
                    
                    if intersection:
                        G.add_edge(node1, node2, weight=len(intersection))
        
        return self._compute_tda_metrics(G)
    
    def _cluster_interval(self, features: np.ndarray, filter_values: np.ndarray,
                         indices: pd.Index, interval: int) -> List[Dict]:
        """Cluster data within an interval"""
        clusters = self.clustering_algo.fit_predict(features)
        
        results = []
        for cluster_id in np.unique(clusters):
            if cluster_id == -1:  # Skip noise
                continue
            
            cluster_mask = (clusters == cluster_id)
            cluster_indices = indices[cluster_mask]
            
            results.append({
                'points': cluster_indices.tolist(),
                'size': len(cluster_indices),
                'interval': interval,
                'avg_filter_val': float(np.mean(filter_values[cluster_mask]))
            })
        
        return results
    
    def _compute_tda_metrics(self, G: nx.Graph) -> Dict[str, Any]:
        """Compute topological metrics from graph"""
        if G.number_of_nodes() == 0:
            return self._default_tda_results("No clusters found, graph is empty.")
        
        try:
            # Basic metrics
            num_nodes = G.number_of_nodes()
            num_edges = G.number_of_edges()
            
            # Components
            components = list(nx.connected_components(G))
            num_components = len(components)
            
            # Cycles
            try:
                cycles = nx.cycle_basis(G)
                num_cycles = len(cycles)
                avg_cycle_len = np.mean([len(c) for c in cycles]) if cycles else 0
            except:
                cycles = []
                num_cycles = 0
                avg_cycle_len = 0
            
            # Flares
            degrees = [d for n, d in G.degree()]
            flares = [n for n, d in G.degree() if d > 2]
            num_flares = len(flares)
            
            # Centrality for small graphs
            if num_nodes < 100:
                centrality = nx.betweenness_centrality(G)
                top_central_nodes = sorted(centrality.items(), 
                                         key=lambda x: x[1], 
                                         reverse=True)[:5]
            else:
                top_central_nodes = []
            
            # Shape interpretation
            shape_interpretation = self._interpret_shape(num_components, 
                                                        num_cycles, 
                                                        num_flares, 
                                                        num_nodes)
            
            return {
                'graph': G,
                'num_nodes': num_nodes,
                'num_edges': num_edges,
                'num_components': num_components,
                'num_cycles': num_cycles,
                'avg_cycle_len': float(avg_cycle_len),
                'num_flares': num_flares,
                'shape_interpretation': shape_interpretation,
                'top_central_nodes': top_central_nodes,
                'avg_degree': float(np.mean(degrees)) if degrees else 0,
                'error': None
            }
        except Exception as e:
            return self._default_tda_results(f"Error computing TDA metrics: {e}")
    
    def _interpret_shape(self, num_components: int, num_cycles: int, 
                        num_flares: int, num_nodes: int) -> str:
        """Interpret topological shape"""
        if num_nodes == 0:
            return "No Structure"
        
        flare_ratio = num_flares / num_nodes
        
        if num_cycles > 0 and num_components == 1:
            if num_cycles > 3:
                return "Complex Cyclic Structure (Multiple Feedback Loops)"
            else:
                return "Simple Cyclic Pattern (Ranging/Consolidation)"
        elif flare_ratio > 0.15:
            return "Branching Structure (High Volatility/Multiple Scenarios)"
        elif num_components > 1:
            if num_components > 3:
                return "Highly Fragmented (Multiple Disconnected Regimes)"
            else:
                return "Bifurcated Market (Regime Transition)"
        elif num_edges > num_nodes * 1.5:
            return "Dense Network (High Correlation/Herding)"
        else:
            return "Linear Progression (Trending Market)"
    
    def _default_tda_results(self, error_msg: str) -> Dict[str, Any]:
        """Default results for failed analysis"""
        return {
            'graph': nx.Graph(),
            'num_nodes': 0,
            'num_edges': 0,
            'num_components': 0,
            'num_cycles': 0,
            'avg_cycle_len': 0,
            'num_flares': 0,
            'shape_interpretation': "Analysis Failed",
            'top_central_nodes': [],
            'avg_degree': 0,
            'error': error_msg
        }


class OptimizedMultiModalTensorFusion:
    """Optimized tensor fusion with GPU support awareness"""
    
    def __init__(self, lookback_size: int = 200):
        self.lookback_size = lookback_size
        self.simplified_analysis = OptimizedSimplifiedTensorAnalysis(lookback_size)
    
    def create_unified_market_tensor(self, data: Dict[str, pd.DataFrame]) -> np.ndarray:
        """Create unified tensor with parallel processing"""
        return self.simplified_analysis.create_simplified_tensor(data)
    
    def tensor_decomposition_prediction(self, tensor: np.ndarray) -> Dict[str, Any]:
        """Perform optimized tensor decomposition"""
        results = self.simplified_analysis.analyze_tensor(tensor)
        
        # Add predictive components
        if 'decomposition' in results and 'temporal_patterns' in results:
            predictions = self._generate_predictions(results['temporal_patterns'])
            results['predictions'] = predictions
        
        return results
    
    def _generate_predictions(self, temporal_patterns: np.ndarray) -> Dict[str, float]:
        """Generate predictions from temporal patterns"""
        predictions = {}
        
        # Analyze each temporal factor
        for i in range(min(temporal_patterns.shape[1], 5)):
            if temporal_patterns.shape[0] > 20:
                # Fit trend using robust regression
                x = np.arange(20)
                y = temporal_patterns[-20:, i]
                
                # Use Theil-Sen estimator
                from sklearn.linear_model import TheilSenRegressor
                
                model = TheilSenRegressor(random_state=42)
                model.fit(x.reshape(-1, 1), y)
                
                # Predict next value
                trend = model.predict([[20]])[0] - model.predict([[19]])[0]
                predictions[f'factor_{i}_trend'] = float(trend)
                
                # Calculate momentum
                momentum = np.mean(np.diff(y[-5:]))
                predictions[f'factor_{i}_momentum'] = float(momentum)
        
        return predictions


class OptimizedInformationFlowAnalyzer:
    """Optimized information flow analysis"""
    
    def __init__(self):
        self.endpoints = [
            'kline_price', 'orderflow', 'orderbook_pressure',
            'large_orders', 'liquidations', 'funding_rate',
            'open_interest', 'cvd', 'market_orders'
        ]
        self.max_lag = 20
        self.advanced_tools = OptimizedAdvancedMathematicalTools()
    
    def calculate_transfer_entropy_matrix(self, data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Calculate transfer entropy with parallel processing"""
        active_endpoints = [e for e in self.endpoints if e in data]
        n_endpoints = len(active_endpoints)
        
        te_matrix = np.zeros((n_endpoints, n_endpoints))
        lag_matrix = np.zeros((n_endpoints, n_endpoints), dtype=int)
        
        # Prepare pairs for parallel processing
        pairs = [(i, j, source, target) 
                for i, source in enumerate(active_endpoints)
                for j, target in enumerate(active_endpoints)
                if i != j]
        
        # Process in parallel
        with ProcessPoolExecutor(max_workers=min(len(pairs), cpu_count())) as executor:
            futures = {}
            
            for i, j, source, target in pairs:
                futures[executor.submit(
                    self._calculate_te_pair,
                    data[source].values,
                    data[target].values
                )] = (i, j)
            
            for future in as_completed(futures):
                i, j = futures[future]
                try:
                    te_result = future.result()
                    te_matrix[i, j] = te_result['max_te']
                    lag_matrix[i, j] = te_result['optimal_lag']
                except Exception as e:
                    print(f"Error calculating TE for pair ({i},{j}): {e}")
        
        # Create DataFrames
        te_df = pd.DataFrame(te_matrix, index=active_endpoints, columns=active_endpoints)
        lag_df = pd.DataFrame(lag_matrix, index=active_endpoints, columns=active_endpoints)
        
        # Create flow network
        flow_network = self._create_flow_network(te_df, lag_df, active_endpoints)
        
        # Calculate network metrics
        network_metrics = self._calculate_network_metrics(flow_network)
        
        return {
            'transfer_entropy': te_df,
            'optimal_lags': lag_df,
            'information_flow_network': flow_network,
            'network_metrics': network_metrics
        }
    
    def _calculate_te_pair(self, source: np.ndarray, target: np.ndarray) -> Dict[str, Any]:
        """Calculate TE for a single pair"""
        return self.advanced_tools.transfer_entropy_spectrum(
            source, target,
            lag_range=range(1, min(self.max_lag + 1, len(source) // 2))
        )
    
    def _create_flow_network(self, te_df: pd.DataFrame, lag_df: pd.DataFrame, 
                           active_endpoints: List[str]) -> nx.DiGraph:
        """Create directed graph of information flow"""
        G = nx.DiGraph()
        
        # Add nodes
        for endpoint in active_endpoints:
            G.add_node(endpoint)
        
        # Add edges where transfer entropy is significant
        threshold = te_df.values.mean() + te_df.values.std()
        
        for i, source in enumerate(active_endpoints):
            for j, target in enumerate(active_endpoints):
                if i != j and te_df.iloc[i, j] > threshold:
                    G.add_edge(
                        source,
                        target,
                        weight=te_df.iloc[i, j],
                        lag=lag_df.iloc[i, j]
                    )
        
        return G
    
    def _calculate_network_metrics(self, G: nx.DiGraph) -> Dict[str, Any]:
        """Calculate network topology metrics"""
        if G.number_of_nodes() == 0:
            return {}
        
        metrics = {
            'density': nx.density(G),
            'avg_clustering': nx.average_clustering(G.to_undirected()),
            'num_strong_components': nx.number_strongly_connected_components(G)
        }
        
        # Centrality measures for small networks
        if G.number_of_nodes() < 50:
            metrics['betweenness_centrality'] = nx.betweenness_centrality(G)
            metrics['eigenvector_centrality'] = nx.eigenvector_centrality(G, max_iter=100)
            
            # Identify information hubs
            bc = metrics['betweenness_centrality']
            metrics['information_hubs'] = sorted(bc.items(), 
                                               key=lambda x: x[1], 
                                               reverse=True)[:3]
        
        return metrics
    
    def partial_information_decomposition(self, data: Dict[str, pd.Series]) -> Dict[str, float]:
        """Decompose information contributions from each endpoint"""
        if 'kline_price' not in data:
            return {}
        
        target = data['kline_price']
        contributions = {}
        
        # Calculate mutual information for each source
        for source_name, source_data in data.items():
            if source_name == 'kline_price':
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
            except:
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
        
        sources = [k for k in data.keys() if k != 'kline_price']
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
                            correlations.append(abs(corr))
                        except:
                            pass
        
        return np.mean(correlations) if correlations else 0.0
    
    def _estimate_synergy(self, data: Dict[str, pd.Series]) -> float:
        """Estimate synergistic information potential"""
        if len(data) < 3:
            return 0.0
        
        correlations = []
        for key1, key2 in product(data.keys(), repeat=2):
            if key1 < key2:
                min_len = min(len(data[key1]), len(data[key2]))
                if min_len > 10:
                    try:
                        corr = np.corrcoef(data[key1][:min_len], data[key2][:min_len])[0, 1]
                        correlations.append(corr)
                    except:
                        pass
        
        return np.std(correlations) if correlations else 0.0


class OptimizedMicrostructureKlineBridge:
    """Optimized microstructure to kline bridge with ML integration"""
    
    def __init__(self):
        self.aggregation_methods = ['time', 'volume', 'dollar', 'information']
        self.microprice_weights = {'bid': 0.3, 'ask': 0.3, 'mid': 0.4}
        self.advanced_tools = OptimizedAdvancedMathematicalTools()
        self.xgb_model = None
        self.model_version = "1.0"
        self.feature_importance = {}
    
    def train_xgboost_model(self, features: pd.DataFrame, target: pd.Series):
        """Train XGBoost with optimized parameters"""
        params = {
            'objective': 'reg:squarederror',
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist'
        }
        
        self.xgb_model = xgb.XGBRegressor(**params)
        
        # Train with early stopping
        X_train, X_val = features[:-100], features[-100:]
        y_train, y_val = target[:-100], target[-100:]
        
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=10,
            verbose=False
        )
        
        # Store feature importance
        self.feature_importance = dict(zip(features.columns, 
                                         self.xgb_model.feature_importances_))
    
    def predict_current_bar_close(self, 
                                  current_bar_data: Dict[str, Any],
                                  orderflow_data: pd.DataFrame,
                                  book_data: pd.DataFrame) -> Dict[str, float]:
        """Predict current kline close"""
        predictions = {}
        
        # 1. Microprice estimation
        microprice = self.calculate_microprice(book_data)
        predictions['microprice'] = microprice
        
        # 2. VWAP projection
        if not orderflow_data.empty:
            vwap_projection = self.project_vwap(orderflow_data, current_bar_data)
            predictions['vwap_projection'] = vwap_projection
        
        # 3. Order flow imbalance projection
        if 'delta' in orderflow_data.columns:
            ofi_projection = self.project_from_order_flow(orderflow_data)
            predictions['ofi_projection'] = ofi_projection
        
        # 4. Kyle Lambda projection
        if not orderflow_data.empty and 'price' in orderflow_data.columns and 'volume' in orderflow_data.columns:
            kyle_projection = self.kyle_lambda_projection(orderflow_data)
            predictions['kyle_lambda_projection'] = kyle_projection
        
        # 5. Momentum projection
        momentum_projection = self.momentum_projection(current_bar_data)
        predictions['momentum_projection'] = momentum_projection
        
        # 6. XGBoost prediction
        if self.xgb_model:
            features = self._prepare_xgb_features(current_bar_data, orderflow_data, book_data)
            xgb_pred = self.predict_with_xgboost(features)
            if xgb_pred:
                predictions['xgboost_prediction'] = xgb_pred
        
        # 7. Weighted ensemble
        weights = {
            'microprice': 0.15,
            'vwap_projection': 0.15,
            'ofi_projection': 0.15,
            'kyle_lambda_projection': 0.15,
            'momentum_projection': 0.10,
            'xgboost_prediction': 0.30
        }
        
        valid_predictions = {k: v for k, v in predictions.items() if v > 0}
        if valid_predictions:
            total_weight = sum(weights.get(k, 0.15) for k in valid_predictions)
            ensemble_prediction = sum(
                valid_predictions[k] * weights.get(k, 0.15) / total_weight
                for k in valid_predictions
            )
        else:
            ensemble_prediction = current_bar_data.get('current', 0)
        
        predictions['ensemble'] = ensemble_prediction
        predictions['confidence'] = self.calculate_prediction_confidence(
            predictions, current_bar_data
        )
        
        return predictions
    
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
        
        # Price projection
        price_base = float(orderflow_data['price'].iloc[-1]) if 'price' in orderflow_data.columns else 0
        
        if price_base == 0:
            return 0.0
        
        # Normalize delta impact
        avg_price_change = orderflow_data['price'].diff().abs().mean() if 'price' in orderflow_data.columns else 1
        delta_impact = (cum_delta / (abs(cum_delta) + 1000)) * avg_price_change
        
        projection = price_base + delta_impact + (delta_acceleration * 0.1)
        
        return float(projection)
    
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
            signed_volume[1:]
        )
        
        # Project price
        recent_signed_volume = np.sum(signed_volume[-5:])
        lambda_estimate = kyle_result['lambda']
        
        projected_change = lambda_estimate * recent_signed_volume
        projection = float(prices[-1] + projected_change)
        
        return projection
    
    def momentum_projection(self, current_bar_data: Dict[str, Any]) -> float:
        """Simple momentum-based projection"""
        open_price = current_bar_data.get('open', 0)
        current_price = current_bar_data.get('current', open_price)
        high = current_bar_data.get('high', current_price)
        low = current_bar_data.get('low', current_price)
        
        if open_price == 0:
            return current_price
        
        # Current momentum
        momentum = (current_price - open_price) / open_price
        
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
    
    def predict_with_xgboost(self, features: pd.DataFrame) -> Optional[float]:
        """Predict using the trained XGBoost model"""
        if self.xgb_model:
            return self.xgb_model.predict(features)[0]
        return None
    
    def _prepare_xgb_features(self, current_bar_data, orderflow_data, book_data) -> pd.DataFrame:
        """Prepare features for the XGBoost model"""
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
            features['vwap'] = (orderflow_data['price'] * orderflow_data['volume']).sum() / features['total_volume'] if features['total_volume'] > 0 else 0
        else:
            features['cum_delta'] = 0
            features['total_volume'] = 0
            features['vwap'] = 0

        # Book features
        if not book_data.empty:
            latest_book = book_data.iloc[-1]
            features['book_imbalance'] = (latest_book['bid_size'] - latest_book['ask_size']) / (latest_book['bid_size'] + latest_book['ask_size'] + 1e-9) if 'bid_size' in latest_book else 0
            features['microprice'] = self.calculate_microprice(book_data)
        else:
            features['book_imbalance'] = 0
            features['microprice'] = 0
            
        return pd.DataFrame([features])
    
    def calculate_prediction_confidence(self, predictions: Dict[str, float],
                                      current_bar_data: Dict[str, Any]) -> float:
        """Calculate confidence in prediction"""
        pred_values = [v for k, v in predictions.items() 
                      if k not in ['ensemble', 'confidence'] and v > 0]
        
        if not pred_values:
            return 0.0
        
        pred_std = np.std(pred_values)
        pred_mean = np.mean(pred_values)
        
        # Coefficient of variation
        cv = pred_std / pred_mean if pred_mean != 0 else 1.0
        
        # Time factor
        time_confidence = 1 - current_bar_data.get('time_remaining_ratio', 0.5)
        
        # Combined confidence
        confidence = (1 - min(cv, 1)) * time_confidence * 100
        
        return float(confidence)


class OptimizedHeikinAshiAnalysis:
    """Optimized Heikin Ashi analysis with vectorized operations"""
    
    def __init__(self, df: pd.DataFrame, z_lookback_period: int = 20):
        self.df = df.copy()
        self.z_lookback_period = z_lookback_period
        self.ha_df = pd.DataFrame(index=df.index)
    
    def calculate_all_signals(self) -> pd.DataFrame:
        """Calculate all signals with optimized vectorization"""
        self._calculate_heikin_ashi_vectorized()
        
        results = pd.DataFrame(index=self.df.index)
        
        # Copy HA values
        for col in ['open', 'high', 'low', 'close']:
            results[f'ha_{col}'] = self.ha_df[col]
        
        # Z-Score Heikin Ashi
        z_ha_results = self._calculate_z_score_ha_vectorized()
        results = pd.concat([results, z_ha_results], axis=1)
        
        # MS-Signal indicators
        ms_signal_results = self._calculate_ms_signal_vectorized()
        results = pd.concat([results, ms_signal_results], axis=1)
        
        return results
    
    def _calculate_heikin_ashi_vectorized(self):
        """Vectorized Heikin Ashi calculation"""
        # HA Close
        self.ha_df['close'] = (self.df['open'] + self.df['high'] + 
                               self.df['low'] + self.df['close']) / 4
        
        # HA Open - vectorized
        ha_open = pd.Series(index=self.df.index, dtype=float)
        ha_open.iloc[0] = (self.df['open'].iloc[0] + self.df['close'].iloc[0]) / 2
        
        # Use exponential moving average for efficiency
        alpha = 0.5
        ha_open = ha_open.fillna(0).ewm(alpha=alpha, adjust=False).mean()
        ha_open = (ha_open.shift(1) + self.ha_df['close'].shift(1)) / 2
        ha_open.iloc[0] = (self.df['open'].iloc[0] + self.df['close'].iloc[0]) / 2
        
        self.ha_df['open'] = ha_open
        
        # HA High and Low
        self.ha_df['high'] = pd.concat([
            self.df['high'], 
            self.ha_df['open'], 
            self.ha_df['close']
        ], axis=1).max(axis=1)
        
        self.ha_df['low'] = pd.concat([
            self.df['low'], 
            self.ha_df['open'], 
            self.ha_df['close']
        ], axis=1).min(axis=1)
    
    def _calculate_z_score_ha_vectorized(self) -> pd.DataFrame:
        """Vectorized Z-Score transformation"""
        z_ha_df = pd.DataFrame(index=self.df.index)
        
        for col in ['open', 'high', 'low', 'close']:
            rolling = self.ha_df[col].rolling(window=self.z_lookback_period)
            mean = rolling.mean()
            std = rolling.std()
            z_ha_df[f'z_ha_{col}'] = (self.ha_df[col] - mean) / (std + 1e-10)
        
        return z_ha_df.fillna(0)
    
    def _calculate_ms_signal_vectorized(self) -> pd.DataFrame:
        """Vectorized MS-Signal calculation"""
        ms_df = pd.DataFrame(index=self.df.index)
        
        # RSI
        try:
            ms_df['ha_rsi'] = talib.RSI(self.ha_df['close'].values, timeperiod=14)
        except:
            delta = self.ha_df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / (loss + 1e-10)
            ms_df['ha_rsi'] = 100 - (100 / (1 + rs))
        
        # High/Low signals
        c = self.ha_df['close']
        o = self.ha_df['open']
        rsi = ms_df['ha_rsi']
        
        # Bearish reversal
        bearish_cond = (c.shift(2) > o.shift(2)) & (c.shift(1) > o.shift(1)) & \
                      (c < o) & (rsi.shift(2) >= 70)
        ms_df['ha_high_signal'] = o.shift(2).where(bearish_cond).ffill()
        
        # Bullish reversal
        bullish_cond = (c.shift(2) < o.shift(2)) & (c.shift(1) < o.shift(1)) & \
                      (c > o) & (rsi.shift(2) <= 30)
        ms_df['ha_low_signal'] = o.shift(2).where(bullish_cond).ffill()
        
        # Stochastic RSI
        stoch_len = 7
        k_len = 3
        d_len = 3
        
        rsi_min = rsi.rolling(window=stoch_len).min()
        rsi_max = rsi.rolling(window=stoch_len).max()
        stoch = 100 * (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10)
        k = stoch.rolling(window=k_len).mean()
        d = k.rolling(window=d_len).mean()
        ms_df['stoch_rsi'] = ((k + d) / 2).fillna(50)
        
        # OBV signals
        obv = (np.sign(self.df['close'].diff()) * self.df['volume']).fillna(0).cumsum()
        obv_high = obv.rolling(window=18).max()
        obv_low = obv.rolling(window=18).min()
        
        h1 = obv_high.where(obv_high == obv_high.shift(1)).ffill()
        l1 = obv_low.where(obv_low == obv_low.shift(1)).ffill()
        
        obv_check = pd.Series(0, index=self.df.index)
        obv_check[(obv.shift(1) <= h1) & (obv > h1)] = 20
        obv_check[(obv.shift(1) >= l1) & (obv < l1)] = -20
        
        ms_df['obv'] = obv
        ms_df['obv_check'] = obv_check.replace(0, method='ffill')
        
        # DMI signals
        try:
            plus_di = talib.PLUS_DI(self.df['high'].values, 
                                   self.df['low'].values, 
                                   self.df['close'].values, timeperiod=14)
            minus_di = talib.MINUS_DI(self.df['high'].values, 
                                     self.df['low'].values, 
                                     self.df['close'].values, timeperiod=14)
            adx = talib.ADX(self.df['high'].values, 
                           self.df['low'].values, 
                           self.df['close'].values, timeperiod=14)
        except:
            plus_di = pd.Series(50, index=self.df.index)
            minus_di = pd.Series(50, index=self.df.index)
            adx = pd.Series(25, index=self.df.index)
        
        # DMI signal calculation
        adx_signals = pd.Series(0, index=self.df.index)
        
        strong_trend = (adx > 25) & (adx >= adx.shift(1))
        weak_trend = (adx < 25) & (adx >= adx.shift(1))
        
        adx_signals[strong_trend & (plus_di >= minus_di)] = 20
        adx_signals[strong_trend & (plus_di < minus_di)] = -20
        adx_signals[weak_trend & (plus_di >= minus_di)] = 10
        adx_signals[weak_trend & (plus_di < minus_di)] = -10
        
        ms_df['dmi_signal'] = adx_signals
        
        # Momentum signals
        period = 10
        mom = self.df['close'].diff(period)
        
        try:
            signal = talib.EMA(mom.values, timeperiod=9)
        except:
            signal = mom.ewm(span=9, adjust=False).mean()
        
        ms_df['mom0'] = np.where(mom > 0, 10, -10)
        ms_df['mom1'] = np.where(mom > signal, 10, -10)
        
        # Sum signal
        ms_df['sum_signal'] = (ms_df['dmi_signal'] + ms_df['obv_check'] + 
                              ms_df['mom0'] + ms_df['mom1'])
        
        return ms_df
class IntegratedMathematicalAnalysisSystem:
    def __init__(self):
        # Configuration
        self.api_key = "1d1e9478ca4c4c36a1b33b2e008bc4a5"
        self.base_url = "https://open-api.coinank.com"
        self.analyst = "ASJFOJ1"
        self.symbol = 'CRYPTO_ASSET'
        self.timeframes = ['15m', '30m', '1h', '4h', '6h', '12h', '1d']
        self.limit = 1000
        self.lookback_size = 200
        
        # Multiprocessing optimization
        self.n_processes = max(1, int(cpu_count() * 0.8))
        self.executor = ThreadPoolExecutor(max_workers=self.n_processes)
        # Removed ProcessPoolExecutor to avoid pickle issues
                
        # Timestamps
        self.current_time = datetime.now(timezone.utc)
        self.current_timestamp = int(self.current_time.timestamp() * 1000)
        timestamp_str = self.current_time.strftime('%Y%m%d_%H%M%S')
        self.version = "v45_optimized"
        
        # Create results directory
        self.results_dir = Path(f"analysis_results_{self.version}")
        self.results_dir.mkdir(exist_ok=True)
        self.run_dir = self.results_dir / timestamp_str
        self.run_dir.mkdir(exist_ok=True)
        
        # Output files
        self.results_file = self.run_dir / f"mathematical_analysis_{self.version}_{timestamp_str}_LATEST.json"
        self.summary_file = self.run_dir / f"mathematical_summary_{self.version}_{timestamp_str}_LATEST.txt"
        self.error_log_file = self.run_dir / f"error_log_{self.version}_{timestamp_str}_LATEST.txt"
        
        self._initialize_output_files()
        
        # Optimized caching
        self.cache = OptimizedMarketDataCache(cache_dir=f"market_data_cache_{self.version}")
        
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
                     "OKX": "BTC-USDT", "Bitget": "BTCUSDT", "Huobi": "BTCUSDT"},
            "FUTURES": {"Binance": "BTCUSDT", "OKX": "BTC-USD-FUTURES"}
        }
        
        self.stats = {
            "total_requests": 0, "successful_requests": 0, "failed_requests": 0,
            "data_points_collected": 0, "sections_completed": [],
            "cache_hits": 0, "cache_misses": 0
        }
        
        # Initialize optimized components
        self.tensor_fusion = OptimizedMultiModalTensorFusion(self.lookback_size)
        self.info_flow_analyzer = OptimizedInformationFlowAnalyzer()
        self.microstructure_bridge = OptimizedMicrostructureKlineBridge()
        self.advanced_tools = OptimizedAdvancedMathematicalTools()
        self.tda_analyzer = OptimizedTopologicalDataAnalysis()
        
        self.collected_data = {}
        self.analysis_results = {}
        self.kline_data = {}
        self.previous_analysis = self.load_previous_analysis()
        
        # Performance monitoring
        self.performance_metrics = {
            'function_timings': {},
            'memory_usage': {}
        }
    
# Backward compatibility method aliases
    def eemd(self, signal_data):
        return self._eemd_optimized(signal_data)

    def kalman_filter(self, observations):
        return self._kalman_filter_optimized(observations)

    def dfa_hurst(self, signal_data, min_scale=4, max_scale=None):
        return self._dfa_hurst_optimized(signal_data, min_scale, max_scale)

    def permutation_entropy(self, signal_data, order=3, delay=1):
        return self._permutation_entropy_optimized(signal_data, order, delay)

    def hilbert_homodyne(self, signal_data):
        return self._hilbert_homodyne_optimized(signal_data)

    def matrix_profile(self, signal_data, window_size=10):
        return self._matrix_profile_optimized(signal_data, window_size)

    def egarch_volatility(self, returns):
        return self._egarch_volatility_optimized(returns)

    def hidden_markov_model(self, returns, n_states=3):
        return self._hidden_markov_model_optimized(returns, n_states)

    def _initialize_output_files(self):
        """Initialize output files with headers"""
        with open(self.error_log_file, 'w', encoding='utf-8') as f:
            f.write(f"Error Log {self.version}\n{'='*80}\n")
            f.write(f"Started: {self.current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC\n\n")
    
    def log_error(self, endpoint: str, params: Dict, error_msg: str, 
                  description: str = "", exc_info: bool = False):
        """Enhanced error logging with context"""
        with open(self.error_log_file, 'a', encoding='utf-8') as f:
            timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            log_entry = (
                f"{timestamp} | DESC: {description} | "
                f"ENDPOINT: {endpoint} | PARAMS: {json.dumps(params)} | "
                f"MSG: {error_msg}\n"
            )
            f.write(log_entry)
            
            if exc_info:
                traceback.print_exc(file=f)
                f.write("\n")
        
        print(f"  ✗ ERROR: {description} - {error_msg[:100]}...")
    
    def validate_timestamp(self, timestamp_ms: int):
        """Check if the requested timestamp is in the future"""
        current_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        if timestamp_ms > current_ms:
            raise ValueError(f"Requesting future data: {timestamp_ms} > {current_ms}")
    
    def make_request(self, endpoint, params, description="", retries=3, retry_delay=2, use_cache=True):
        """Make API request with error handling, retry mechanism, and caching"""
        if use_cache:
            cached_result = self.cache.get_cached_data(endpoint, params)
            if cached_result:
                cached_data = cached_result['data']
                last_timestamp = cached_result['last_timestamp']
                
                if last_timestamp:
                    params_new = params.copy()
                    params_new['startTime'] = str(last_timestamp + 1)
                    
                    print(f"  → {description}: Found cache, fetching updates...")
                    new_response = self._direct_request(endpoint, params_new, description, retries, retry_delay)
                    
                    if new_response and new_response.get('data'):
                        self.stats['cache_hits'] += 1
                        merged_data = self._merge_responses(cached_data, new_response, endpoint)
                        new_last_ts = self._extract_last_timestamp(merged_data, endpoint)
                        self.cache.save_data(endpoint, params, merged_data, new_last_ts)
                        return merged_data
                    else:
                        print(f"  → {description}: Using cached data (no new updates).")
                        self.stats['cache_hits'] += 1
                        return cached_data
                else:
                    print(f"  → {description}: Using fully cached data.")
                    self.stats['cache_hits'] += 1
                    return cached_data
        
        self.stats['cache_misses'] += 1
        response = self._direct_request(endpoint, params, description, retries, retry_delay)
        
        if response and use_cache:
            last_timestamp = self._extract_last_timestamp(response, endpoint)
            self.cache.save_data(endpoint, params, response, last_timestamp)
        
        return response
    
    def _direct_request(self, endpoint, params, description, retries, retry_delay):
        """The actual HTTP request part"""
        time.sleep(1.5)  # Global throttle
        
        timeout = 60 if endpoint == "/api/liquidation/orders" else 20
        
        # Validate timestamp
        if 'endTime' in params:
            try:
                end_time_ms = int(params['endTime'])
                self.validate_timestamp(end_time_ms)
            except (ValueError, TypeError) as e:
                self.log_error(endpoint, params, f"Invalid endTime: {e}", description)
                params['endTime'] = str(int((datetime.now(timezone.utc) - timedelta(minutes=15)).timestamp() * 1000))
        
        for attempt in range(retries):
            self.stats["total_requests"] += 1
            try:
                response = requests.get(
                    f"{self.base_url}{endpoint}",
                    headers={'apikey': self.api_key},
                    params=params,
                    timeout=timeout
                )
                
                if response.status_code >= 500:
                    response.raise_for_status()
                
                data = response.json()
                if data.get('success') and str(data.get('code')) == '1':
                    self.stats["successful_requests"] += 1
                    return data
                else:
                    error_msg = f"API Error: {data.get('msg', 'Unknown error')}"
                    if "system error" in error_msg.lower() and attempt < retries - 1:
                        print(f"  ? {description}: {error_msg}. Retrying ({attempt+1}/{retries})...")
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    
                    self.stats["failed_requests"] += 1
                    self.log_error(endpoint, params, error_msg, description)
                    return None
            
            except requests.exceptions.RequestException as e:
                error_msg = f"RequestException: {str(e)}"
                if attempt < retries - 1:
                    print(f"  ? {description}: {error_msg}. Retrying ({attempt+1}/{retries})...")
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    self.stats["failed_requests"] += 1
                    self.log_error(endpoint, params, error_msg, description, exc_info=True)
                    return None
        return None
    
    def _extract_last_timestamp(self, response: Dict[str, Any], endpoint: str) -> Optional[int]:
        """Extract the latest timestamp from various response data structures"""
        if not response or not response.get('data'):
            return None
        
        data_root = response.get('data')
        
        def get_ts_from_record(rec):
            """Helper to extract timestamp from a single data record"""
            if isinstance(rec, dict):
                for field in ['ts', 'timestamp', 'begin', 'openTime', 'lastUpdateTime']:
                    if field in rec and rec[field] is not None:
                        try:
                            return int(rec[field])
                        except (ValueError, TypeError):
                            continue
            elif isinstance(rec, list) and len(rec) > 0:
                try:
                    return int(rec[0])
                except (ValueError, TypeError):
                    pass
            return None
        
        data_list = None
        if isinstance(data_root, list):
            data_list = data_root
        elif isinstance(data_root, dict):
            for key in ['list', 'data', 'details']:
                if key in data_root and isinstance(data_root[key], list):
                    data_list = data_root[key]
                    break
            
            if not data_list and 'data' in data_root and isinstance(data_root['data'], dict):
                if 'data' in data_root['data'] and isinstance(data_root['data']['data'], list):
                    data_list = data_root['data']['data']
            
            if not data_list and 'tss' in data_root and isinstance(data_root['tss'], list):
                if data_root['tss']:
                    return int(data_root['tss'][-1])
        
        if data_list and len(data_list) > 0:
            return get_ts_from_record(data_list[-1])
        
        return None
    
    def _merge_responses(self, cached_response: Dict[str, Any], new_response: Dict[str, Any], 
                        endpoint: str) -> Dict[str, Any]:
        """Merge cached and new response data"""
        cached_data_root = cached_response.get('data')
        new_data_root = new_response.get('data')
        
        if not new_data_root:
            return cached_response
        if not cached_data_root:
            return new_response
        
        def merge_lists(list1, list2):
            if not list1 or not list2:
                return list1 or list2
            
            ts_field = None
            first_record = list2[0]
            if isinstance(first_record, dict):
                for field in ['ts', 'timestamp', 'begin', 'openTime', 'lastUpdateTime']:
                    if field in first_record:
                        ts_field = field
                        break
            elif isinstance(first_record, list):
                ts_field = 0
            
            if ts_field is None:
                return list1 + list2
            
            merged_dict = {}
            for item in list1 + list2:
                key = item[ts_field] if isinstance(item, dict) else item[ts_field]
                merged_dict[key] = item
            
            sorted_items = sorted(merged_dict.values(), 
                                key=lambda x: x[ts_field] if isinstance(x, dict) else x[ts_field])
            return sorted_items
        
        # Handle different response structures
        if isinstance(cached_data_root, list) and isinstance(new_data_root, list):
            merged_response = cached_response.copy()
            merged_response['data'] = merge_lists(cached_data_root, new_data_root)
            return merged_response
        
        if isinstance(cached_data_root, dict) and isinstance(new_data_root, dict):
            merged_response = new_response.copy()
            
            for key in ['list', 'data', 'details']:
                if key in cached_data_root and key in new_data_root and \
                   isinstance(cached_data_root[key], list) and isinstance(new_data_root[key], list):
                    merged_response['data'][key] = merge_lists(cached_data_root[key], new_data_root[key])
                    return merged_response
            
            # Handle tss format
            if 'tss' in cached_data_root and 'tss' in new_data_root:
                def reconstruct(data_dict):
                    records = []
                    value_key = next((k for k in data_dict if k != 'tss'), None)
                    if not value_key: return []
                    for ts, val in zip(data_dict['tss'], data_dict[value_key]):
                        records.append({'timestamp': ts, 'value': val})
                    return records
                
                cached_records = reconstruct(cached_data_root)
                new_records = reconstruct(new_data_root)
                merged_records = merge_lists(cached_records, new_records)
                
                value_key = next((k for k in new_data_root if k != 'tss'), 'longShortRatio')
                merged_response['data']['tss'] = [rec['timestamp'] for rec in merged_records]
                merged_response['data'][value_key] = [rec['value'] for rec in merged_records]
                return merged_response
        
        return new_response
    
    def fetch_klines_concurrent(self):
        """Fetch kline data for all timeframes concurrently"""
        print("\n[FETCHING KLINE DATA - CONCURRENT]")
        url = "https://api.binance.com/api/v3/klines"
        
        def fetch_single_timeframe(interval):
            params = {
                'symbol': 'BTCUSDT',
                'interval': interval,
                'limit': self.limit
            }
            
            try:
                print(f"  → Fetching {interval} data...")
                response = requests.get(url, params=params, timeout=30)
                
                if response.status_code != 200:
                    print(f"  ✗ Error fetching {interval}: {response.text}")
                    return interval, None
                
                data = response.json()
                print(f"  ✓ Received {len(data)} candles for {interval}")
                
                if not data:
                    return interval, None
                
                # Convert to DataFrame
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])
                
                # Convert to numeric
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                for col in ['close', 'high', 'low', 'open', 'volume', 'taker_buy_base', 'taker_buy_quote']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                return interval, df
                
            except Exception as e:
                self.log_error("Binance Klines", params, str(e), f"Kline fetch for {interval}", exc_info=True)
                return interval, None
        
        # Fetch all timeframes concurrently
        with ThreadPoolExecutor(max_workers=len(self.timeframes)) as executor:
            futures = {executor.submit(fetch_single_timeframe, tf): tf 
                      for tf in self.timeframes}
            
            for future in as_completed(futures):
                try:
                    interval, df = future.result()
                    if df is not None:
                        self.kline_data[interval] = df
                except Exception as e:
                    self.log_error("Kline Processing", {}, str(e), "Processing future for klines", exc_info=True)
        
        print(f"\nSuccessfully fetched data for: {list(self.kline_data.keys())}")
    
    def collect_order_flow_data(self, timeframe='1h'):
        """Collect order flow data"""
        order_flow_data = []
        
        for product_type in ["SWAP", "SPOT"]:
            for exchange in self.exchange_groups["orderflow"][:2]:
                for tick_count in [1, 5]:
                    symbol = self.best_symbols[product_type].get(exchange, "BTCUSDT")
                    params = {
                        'exchange': exchange,
                        'symbol': symbol,
                        'interval': timeframe,
                        'endTime': str(self.current_timestamp),
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
        
        if timeframe not in self.collected_data:
            self.collected_data[timeframe] = {}
        self.collected_data[timeframe]['order_flow'] = order_flow_data
    
    def collect_cvd_data(self, timeframe='1h'):
        """Collect Cumulative Volume Delta data"""
        cvd_data = []
        
        for product_type in ["SWAP", "SPOT"]:
            # Individual exchange CVD
            for exchange in self.exchange_groups["cvd"][:2]:
                symbol = self.best_symbols[product_type].get(exchange, "BTCUSDT")
                params = {
                    'exchange': exchange,
                    'symbol': symbol,
                    'interval': timeframe,
                    'endTime': str(self.current_timestamp),
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
            
            # Aggregated CVD
            params = {
                'baseCoin': 'BTC',
                'interval': timeframe,
                'endTime': str(self.current_timestamp),
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
        
        if timeframe not in self.collected_data:
            self.collected_data[timeframe] = {}
        self.collected_data[timeframe]['cvd'] = cvd_data
    
    def collect_market_order_data(self, timeframe='1h'):
        """Collect market order metrics"""
        market_order_data = []
        
        for product_type in ["SWAP", "SPOT"]:
            # Per-exchange data
            for exchange in self.exchange_groups["market_order"][:2]:
                symbol = self.best_symbols[product_type].get(exchange, "BTCUSDT")
                params = {
                    'exchange': exchange,
                    'symbol': symbol,
                    'interval': timeframe,
                    'endTime': str(self.current_timestamp),
                    'size': '300',
                    'productType': product_type
                }
                
                # Buy/sell volume
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
                
                # Buy/sell count
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
        
        if timeframe not in self.collected_data:
            self.collected_data[timeframe] = {}
        self.collected_data[timeframe]['market_orders'] = market_order_data
    
    def collect_large_order_data(self, timeframe='1h'):
        """Collect large order data"""
        large_order_data = []
        
        thresholds = ['1000000', '5000000']
        for product_type in ["SWAP", "SPOT"]:
            for exchange in self.exchange_groups["large_orders"][:1]:
                symbol = self.best_symbols[product_type].get(exchange, "BTCUSDT")
                for threshold in thresholds:
                    params = {
                        'exchange': exchange,
                        'symbol': symbol,
                        'productType': product_type,
                        'amount': threshold,
                        'size': '300',
                        'endTime': str(self.current_timestamp)
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
                                    'timeframe': timeframe,
                                    'order_type': 'large_market_trade'
                                })
                                self.stats["data_points_collected"] += 1
        
        if timeframe not in self.collected_data:
            self.collected_data[timeframe] = {}
        self.collected_data[timeframe]['large_orders'] = large_order_data
    
    def collect_orderbook_data(self, timeframe='1h'):
        """Collect orderbook data"""
        orderbook_data = []
        
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
                        'endTime': str(self.current_timestamp),
                        'size': '300'
                    }
                    data = self.make_request("/api/orderBook/v2/bySymbol", params,
                                           f"{exchange} {product_type} OrderBook {timeframe}")
                    if data and data.get('data'):
                        for record in data.get('data', []):
                            if isinstance(record, list) and len(record) >= 5:
                                buy_usd, sell_usd = float(record[1]), float(record[3])
                                imbalance = (buy_usd - sell_usd) / (buy_usd + sell_usd + 1e-10)
                                orderbook_data.append({
                                    'exchange': exchange,
                                    'product_type': product_type,
                                    'timestamp': record[0],
                                    'rate': rate,
                                    'buy_usd': buy_usd,
                                    'sell_usd': sell_usd,
                                    'imbalance': imbalance,
                                    'timeframe': timeframe
                                })
                                self.stats["data_points_collected"] += 1
        
        if timeframe not in self.collected_data:
            self.collected_data[timeframe] = {}
        self.collected_data[timeframe]['orderbook'] = orderbook_data
    
    def collect_open_interest_data(self, timeframe='1h'):
        """Collect open interest data"""
        oi_data = []
        
        # Aggregated OI
        params = {
            'baseCoin': 'BTC',
            'interval': timeframe,
            'endTime': self.current_timestamp,
            'size': '300'
        }
        
        data = self.make_request("/api/openInterest/aggKline", params, f"Aggregated OI {timeframe}")
        if data and data.get('data'):
            prev_oi = None
            for record in data.get('data', []):
                if isinstance(record, dict):
                    curr_oi = float(record.get('close', 0))
                    oi_change = curr_oi - prev_oi if prev_oi is not None else 0
                    change_pct = (oi_change / prev_oi * 100) if prev_oi and prev_oi > 0 else 0
                    
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
                    'endTime': self.current_timestamp,
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
                            change_pct = (oi_change / prev_oi * 100) if prev_oi and prev_oi > 0 else 0
                            
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
        
        if timeframe not in self.collected_data:
            self.collected_data[timeframe] = {}
        self.collected_data[timeframe]['open_interest'] = oi_data
    
    def collect_positioning_data(self, timeframe='1h'):
        """Collect positioning data"""
        positioning_data = []
        
        for exchange in self.exchange_groups["longshort"][:1]:
            symbol = self.best_symbols["SWAP"].get(exchange, "BTCUSDT")
            
            # Global Account Ratio
            params = {
                'exchange': exchange,
                'symbol': symbol,
                'interval': timeframe,
                'endTime': str(self.current_timestamp),
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
        
        if timeframe not in self.collected_data:
            self.collected_data[timeframe] = {}
        self.collected_data[timeframe]['positioning'] = positioning_data
    
    def collect_net_positions_data(self, timeframe='1h'):
        """Collect net positions data"""
        net_positions_data = []
        
        for exchange in self.exchange_groups["net_positions"][:2]:
            symbol = self.best_symbols["SWAP"].get(exchange, "BTCUSDT")
            params = {
                'exchange': exchange,
                'symbol': symbol,
                'interval': timeframe,
                'endTime': str(self.current_timestamp),
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
        
        if timeframe not in self.collected_data:
            self.collected_data[timeframe] = {}
        self.collected_data[timeframe]['net_positions'] = net_positions_data
    
    def collect_liquidation_data(self, timeframe='1h'):
        """Collect liquidation data"""
        liquidation_data = []
        
        # All Exchange liquidation
        params = {
            'baseCoin': 'BTC',
            'interval': timeframe,
            'endTime': self.current_timestamp,
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
                    ls_ratio = long_val / short_val if short_val > 0 else 0
                    
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
        
        if timeframe not in self.collected_data:
            self.collected_data[timeframe] = {}
        self.collected_data[timeframe]['liquidation'] = liquidation_data
    
    def collect_funding_data(self, timeframe='1h'):
        """Collect funding rate data"""
        funding_data = []
        
        # Aggregated funding history
        params = {
            'baseCoin': 'BTC',
            'exchangeType': 'USDT',
            'endTime': str(self.current_timestamp),
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
        
        # Weighted funding rates
        params = {
            'baseCoin': 'BTC',
            'interval': timeframe,
            'endTime': self.current_timestamp,
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
        
        if timeframe not in self.collected_data:
            self.collected_data[timeframe] = {}
        self.collected_data[timeframe]['funding'] = funding_data
    
    def collect_fund_flow_data(self, timeframe='1h'):
        """Collect fund flow data"""
        fund_flow_data = []
        
        # Map timeframe to API interval
        if timeframe in ['15m', '30m']:
            api_interval = '15m'
        elif timeframe in ['4h', '6h', '12h']:
            api_interval = '1h'
        elif timeframe == '1d':
            api_interval = '1d'
        else:
            api_interval = timeframe
        
        # Historical fund flow
        for product_type in ["SWAP", "SPOT"]:
            params = {
                'baseCoin': 'BTC',
                'endTime': str(self.current_timestamp),
                'productType': product_type,
                'size': '300',
                'interval': api_interval
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
        
        if timeframe not in self.collected_data:
            self.collected_data[timeframe] = {}
        self.collected_data[timeframe]['fund_flow'] = fund_flow_data
    
    def collect_all_market_data_by_endpoint(self):
        """Collect market data by iterating through endpoints first"""
        print("\n[COLLECTING MARKET DATA - BY ENDPOINT]")
        
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
        
        for func, name in tasks_to_run:
            print(f"\n--- Collecting {name} Data ---")
            
            with ThreadPoolExecutor(max_workers=self.n_processes) as executor:
                futures = {
                    executor.submit(func, timeframe=tf): tf
                    for tf in self.timeframes
                }
                
                for future in as_completed(futures):
                    try:
                        future.result(timeout=180)
                    except Exception as e:
                        timeframe = futures[future]
                        self.log_error(name, {'timeframe': timeframe}, str(e), 
                                     f"Data collection for {name}", exc_info=True)
        
        print(f"\nTotal data points collected: {self.stats['data_points_collected']:,}")
        print(f"Cache Hits: {self.stats['cache_hits']}, Cache Misses: {self.stats['cache_misses']}")
    
    def perform_comprehensive_analysis_optimized(self):
        """Optimized analysis with parallel processing"""
        print("\n[PERFORMING OPTIMIZED MATHEMATICAL ANALYSIS]")
        
        # Parallel timeframe analysis
        with ThreadPoolExecutor(max_workers=self.n_processes) as executor:
            futures = {}
            
            for timeframe in self.timeframes:
                if timeframe not in self.kline_data:
                    continue
                
                futures[executor.submit(self._analyze_timeframe_optimized, 
                                      timeframe)] = timeframe
            
            for future in as_completed(futures):
                timeframe = futures[future]
                try:
                    self.analysis_results[timeframe] = future.result()
                    print(f"  ✓ Completed analysis for {timeframe}")
                except Exception as e:
                    print(f"  ✗ Failed analysis for {timeframe}: {e}")
                    self.log_error("Analysis", {'timeframe': timeframe}, 
                                 str(e), "Timeframe analysis", exc_info=True)

        # Cross-timeframe analysis
        print("\n  → Performing cross-timeframe analysis...")
        
        print("    • Multi-Modal Tensor Fusion...")
        self.perform_tensor_fusion_analysis()
        
        print("    • Information Flow Dynamics...")
        self.perform_information_flow_analysis()
        
        print("    • Microstructure-Kline Bridge...")
        self.perform_microstructure_bridge_analysis()
        
        print("    • Topological Data Analysis...")
        self.perform_tda_analysis()
        
        print("    • Market Structure Analysis...")
        self.analyze_order_flow_patterns_enhanced()
        self.analyze_market_positioning_enhanced()
        self.analyze_liquidation_cascades_enhanced()
        
        print("\n  ✓ Analysis complete!")
    
    def _analyze_timeframe_optimized(self, timeframe: str) -> Dict[str, Any]:
        """Optimized analysis for a single timeframe"""
        df = self.kline_data[timeframe]
        
        if df.empty or len(df) < 20:
            return {'error': 'Insufficient data'}
        
        prices = df['close'].values
        returns = np.diff(np.log(prices + 1e-10))
        returns = np.concatenate([[0], returns])
        
        results = {}
        
        # Advanced Mathematical Tools
        results['multifractal'] = self.advanced_tools.multifractal_spectrum_analysis(returns)
        results['sample_entropy'] = self.advanced_tools.sample_entropy(returns)
        results['rqa'] = self.advanced_tools.recurrence_quantification_analysis(returns)
        
        # Kyle Lambda
        if timeframe in self.collected_data and 'order_flow' in self.collected_data[timeframe]:
            order_flow_df = pd.DataFrame(self.collected_data[timeframe]['order_flow'])
            if not order_flow_df.empty and 'total_bid' in order_flow_df.columns:
                signed_volume = order_flow_df['total_bid'] - order_flow_df['total_ask']
                if len(signed_volume) <= len(prices):
                    price_subset = prices[:len(signed_volume)]
                    price_changes = np.diff(price_subset)
                    if len(price_changes) > 0:
                        results['kyle_lambda'] = self.advanced_tools.kyle_lambda_estimation(
                            price_changes, signed_volume[1:].values, method='robust'
                        )
        
        # Traditional Analysis
        results['eemd'] = self._eemd_optimized(prices)
        results['kalman'] = self._kalman_filter_optimized(prices)
        results['dfa'] = self._dfa_hurst_optimized(returns)
        results['pe'] = self._permutation_entropy_optimized(returns)
        results['hilbert'] = self._hilbert_homodyne_optimized(prices)
        results['matrix_profile'] = self._matrix_profile_optimized(prices)
        results['egarch'] = self._egarch_volatility_optimized(returns)
        results['hmm'] = self._hidden_markov_model_optimized(returns)
        
        # Heikin Ashi
        ha_analyzer = OptimizedHeikinAshiAnalysis(df)
        results['ha_ms_signal'] = ha_analyzer.calculate_all_signals()
        
        # VECM Analysis
        try:
            vecm_data = self._prepare_vecm_data(timeframe)
            if vecm_data is not None and len(vecm_data) > 20:
                results['vecm'] = self.advanced_tools.cointegration_and_vecm(vecm_data)
            else:
                results['vecm'] = {'error': 'Insufficient data for VECM'}
        except Exception as e:
            results['vecm'] = {'error': f'VECM failed: {str(e)}'}
        
        return results
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def _eemd_core_numba(signal: np.ndarray, num_modes: int = 5) -> List[np.ndarray]:
        """Numba-optimized EEMD core computation"""
        imfs = []
        residue = signal.copy()
        
        for i in range(num_modes):
            if len(residue) < 10:
                break
            
            noise = np.random.normal(0, 0.1 * np.std(residue), len(residue))
            noisy_signal = residue + noise
            
            h = noisy_signal.copy()
            
            for sift in range(10):
                peaks = []
                troughs = []
                
                for j in range(1, len(h) - 1):
                    if h[j] > h[j-1] and h[j] > h[j+1]:
                        peaks.append(j)
                    elif h[j] < h[j-1] and h[j] < h[j+1]:
                        troughs.append(j)
                
                if len(peaks) < 2 or len(troughs) < 2:
                    break
                
                upper = np.zeros(len(h))
                lower = np.zeros(len(h))
                
                # Linear interpolation for envelopes
                for k in range(len(h)):
                    if k <= peaks[0]:
                        upper[k] = h[peaks[0]]
                    elif k >= peaks[-1]:
                        upper[k] = h[peaks[-1]]
                    else:
                        for p in range(len(peaks) - 1):
                            if peaks[p] <= k <= peaks[p+1]:
                                t = (k - peaks[p]) / (peaks[p+1] - peaks[p])
                                upper[k] = h[peaks[p]] * (1 - t) + h[peaks[p+1]] * t
                                break
                
                for k in range(len(h)):
                    if k <= troughs[0]:
                        lower[k] = h[troughs[0]]
                    elif k >= troughs[-1]:
                        lower[k] = h[troughs[-1]]
                    else:
                        for t in range(len(troughs) - 1):
                            if troughs[t] <= k <= troughs[t+1]:
                                frac = (k - troughs[t]) / (troughs[t+1] - troughs[t])
                                lower[k] = h[troughs[t]] * (1 - frac) + h[troughs[t+1]] * frac
                                break
                
                mean_env = (upper + lower) / 2
                h = h - mean_env
            
            imfs.append(h)
            residue = residue - h
            
            if np.std(residue) < 0.001 * np.std(signal):
                break
        
        return imfs
    
    def _eemd_optimized(self, signal_data: np.ndarray) -> Dict[str, Any]:
        """Optimized EEMD implementation"""
        try:
            imfs = IntegratedMathematicalAnalysisSystem._eemd_core_numba(signal_data.astype(np.float64))
            
            imf_energies = [np.sum(imf ** 2) for imf in imfs]
            total_energy = sum(imf_energies) if imf_energies else 1
            
            dominant_imf = int(np.argmax(imf_energies)) if imf_energies else 0
            
            return {
                'num_imfs': len(imfs),
                'energy_distribution': imf_energies,
                'dominant_imf': dominant_imf,
                'total_energy': float(total_energy),
                'imfs': imfs
            }
        except Exception as e:
            return {
                'num_imfs': 0,
                'energy_distribution': [],
                'dominant_imf': 0,
                'total_energy': 0,
                'imfs': [],
                'error': str(e)
            }
    def _kalman_filter_optimized(self, observations: np.ndarray) -> Dict[str, Any]:
        """Optimized Kalman Filter implementation"""
        try:
            observations = np.asarray(observations, dtype=np.float64)
            n = len(observations)
            filtered = np.zeros(n)
            innovations = np.zeros(n)
            variance = np.zeros(n)
            
            # Initialize
            x = observations[0]
            P = 1.0
            Q = 0.01  # Process noise
            R = 1.0   # Measurement noise
            
            filtered[0] = x
            variance[0] = np.sqrt(P)
            
            for i in range(1, n):
                # Predict
                x_pred = x
                P_pred = P + Q
                
                # Update
                K = P_pred / (P_pred + R)
                innovations[i] = observations[i] - x_pred
                x = x_pred + K * innovations[i]
                P = (1 - K) * P_pred
                
                filtered[i] = x
                variance[i] = np.sqrt(P)
            
            return {
                'filtered': filtered,
                'innovations': innovations,
                'variance': variance,
                'kalman_gain': float(K)
            }
        except Exception as e:
            return {
                'filtered': observations,
                'innovations': np.zeros_like(observations),
                'variance': np.zeros_like(observations),
                'kalman_gain': 0,
                'error': str(e)
            }
    
    def _dfa_hurst_optimized(self, signal_data: np.ndarray, min_scale: int = 4, 
                             max_scale: int = None) -> Dict[str, Any]:
        """Optimized DFA and Hurst Exponent calculation"""
        try:
            if len(signal_data) < 20:
                return {'hurst': 0.5, 'local_hurst': np.full_like(signal_data, 0.5), 
                       'persistence': 'neutral', 'optimal_window': 20, 'fluctuations': []}
            
            if max_scale is None:
                max_scale = len(signal_data) // 4
            
            # Cumulative sum
            y = np.cumsum(signal_data - np.mean(signal_data))
            
            # Calculate scales
            scales = np.unique(np.logspace(np.log10(min_scale), np.log10(max_scale), 20).astype(int))
            fluctuations = []
            
            for scale in scales:
                n_segments = len(y) // scale
                if n_segments == 0:
                    continue
                
                seg_flucts = []
                for i in range(n_segments):
                    segment = y[i * scale:(i + 1) * scale]
                    x = np.arange(len(segment))
                    
                    try:
                        coeffs = np.polyfit(x, segment, 1)
                        fit = np.polyval(coeffs, x)
                        fluct = np.sqrt(np.mean((segment - fit) ** 2))
                        seg_flucts.append(fluct)
                    except:
                        pass
                
                if seg_flucts:
                    fluctuations.append((scale, np.mean(seg_flucts)))
            
            # Calculate Hurst exponent
            if len(fluctuations) > 1:
                scales_log = np.log([f[0] for f in fluctuations])
                flucts_log = np.log([f[1] for f in fluctuations])
                try:
                    hurst, _ = np.polyfit(scales_log, flucts_log, 1)
                except:
                    hurst = 0.5
            else:
                hurst = 0.5
            
            # Local Hurst using sliding window
            window_size = 20
            local_hurst = []
            
            for i in range(len(signal_data)):
                start = max(0, i - window_size)
                end = min(len(signal_data), i + window_size)
                if end - start > 10:
                    local_data = signal_data[start:end]
                    local_h = 0.5 + 0.1 * np.tanh(np.std(local_data) / 
                                                 (np.mean(np.abs(local_data)) + 1e-10))
                else:
                    local_h = 0.5
                local_hurst.append(local_h)
            
            return {
                'hurst': float(hurst),
                'local_hurst': np.array(local_hurst),
                'persistence': 'persistent' if hurst > 0.5 else 'anti-persistent',
                'optimal_window': window_size,
                'fluctuations': fluctuations
            }
        except Exception as e:
            return {'hurst': 0.5, 'local_hurst': np.full_like(signal_data, 0.5), 
                   'persistence': 'neutral', 'optimal_window': 20, 'fluctuations': [],
                   'error': str(e)}
                       
    def _permutation_entropy_optimized(self, signal_data: np.ndarray, order: int = 3, 
                                      delay: int = 1) -> Dict[str, float]:
        """Optimized permutation entropy calculation"""
        try:
            def _embed(x, order, delay):
                N = len(x)
                if N < (order - 1) * delay + 1:
                    return np.array([])
                Y = np.zeros((order, N - (order - 1) * delay))
                for i in range(order):
                    Y[i] = x[i * delay:i * delay + Y.shape[1]]
                return Y.T
            
            window_size = 20
            pe_values = []
            
            for i in range(len(signal_data)):
                start = max(0, i - window_size)
                end = min(len(signal_data), i + window_size)
                
                if end - start >= order * delay + 1:
                    window_data = signal_data[start:end]
                    
                    mat = _embed(window_data, order, delay)
                    if len(mat) > 0:
                        sorted_indices = np.argsort(mat, axis=1)
                        patterns = []
                        for row in sorted_indices:
                            pattern = tuple(row)
                            patterns.append(pattern)
                        
                        unique, counts = np.unique(patterns, return_counts=True, axis=0)
                        probs = counts / len(patterns)
                        pe = entropy(probs + 1e-10)
                        
                        pe_norm = pe / np.log(math.factorial(order))
                    else:
                        pe_norm = 0.5
                else:
                    pe_norm = 0.5
                
                pe_values.append(pe_norm)
            
            return {
                'pe': np.array(pe_values),
                'complexity': float(np.mean(pe_values)),
                'predictability': float(1 - np.mean(pe_values)),
                'entropy_efficiency': float(np.std(pe_values) / (np.mean(pe_values) + 1e-10))
            }
        except Exception as e:
            return {'pe': np.full_like(signal_data, 0.5), 'complexity': 0.5, 
                   'predictability': 0.5, 'entropy_efficiency': 0, 'error': str(e)}
    
    def _hilbert_homodyne_optimized(self, signal_data: np.ndarray) -> Dict[str, Any]:
        """Optimized Hilbert Transform and Homodyne Discriminator"""
        try:
            # Hilbert transform
            analytic_signal = hilbert(signal_data)
            
            # Extract amplitude and phase
            amplitude = np.abs(analytic_signal)
            phase = np.unwrap(np.angle(analytic_signal))
            
            # Instantaneous frequency
            inst_freq = np.diff(phase) / (2.0 * np.pi)
            inst_freq = np.concatenate([[0], inst_freq])
            
            # Phase coherence
            phase_diff = np.diff(phase)
            phase_coherence = 1 - np.abs(np.sin(phase_diff))
            phase_coherence = np.concatenate([[1], phase_coherence])
            
            # Additional metrics
            amplitude_envelope = np.abs(hilbert(amplitude))
            phase_velocity = np.gradient(phase)
            
            return {
                'amplitude': amplitude,
                'inst_freq': inst_freq,
                'phase': phase,
                'phase_coherence': phase_coherence,
                'amplitude_envelope': amplitude_envelope,
                'phase_velocity': phase_velocity
            }
        except Exception as e:
            return {
                'amplitude': np.abs(signal_data),
                'inst_freq': np.zeros_like(signal_data),
                'phase': np.zeros_like(signal_data),
                'phase_coherence': np.ones_like(signal_data),
                'amplitude_envelope': np.abs(signal_data),
                'phase_velocity': np.zeros_like(signal_data),
                'error': str(e)
            }
    
    def _matrix_profile_optimized(self, signal_data: np.ndarray, window_size: int = 10) -> Dict[str, Any]:
        """Optimized Matrix Profile calculation"""
        try:
            n = len(signal_data)
            if n < window_size * 2:
                mp = np.full(max(1, n - window_size + 1), np.inf)
                mp_idx = np.zeros(len(mp), dtype=int)
            else:
                # Create subsequences using sliding window
                shape = (n - window_size + 1, window_size)
                strides = (signal_data.strides[0], signal_data.strides[0])
                subsequences = np.lib.stride_tricks.as_strided(signal_data, shape=shape, strides=strides)
                
                # Initialize matrix profile
                mp = np.full(n - window_size + 1, np.inf)
                mp_idx = np.zeros(n - window_size + 1, dtype=int)
                
                # Calculate distances
                for i in range(n - window_size + 1):
                    query = subsequences[i]
                    
                    # Vectorized distance calculation
                    distances = np.sqrt(np.sum((subsequences - query) ** 2, axis=1))
                    
                    # Exclude trivial matches
                    exclusion_zone_start = max(0, i - window_size // 2)
                    exclusion_zone_end = min(n - window_size + 1, i + window_size // 2 + 1)
                    distances[exclusion_zone_start:exclusion_zone_end] = np.inf
                    
                    # Find minimum
                    min_dist = np.min(distances)
                    if np.isfinite(min_dist):
                        mp[i] = min_dist
                        mp_idx[i] = np.argmin(distances)
            
            # Pad for consistent length
            mp_padded = np.concatenate([mp, np.full(window_size - 1, mp[-1] if len(mp) > 0 else 0)])
            
            # Normalize for anomaly score
            if np.max(mp_padded) > np.min(mp_padded):
                anomaly_score = (mp_padded - np.min(mp_padded)) / (np.max(mp_padded) - np.min(mp_padded))
            else:
                anomaly_score = np.zeros_like(mp_padded)
            
            # Discord (anomaly) detection
            discord_idx = np.argmax(mp) if len(mp) > 0 and np.any(np.isfinite(mp)) else 0
            
            return {
                'matrix_profile': mp_padded,
                'anomaly_score': anomaly_score,
                'motif_idx': mp_idx,
                'discord_idx': int(discord_idx),
                'discord_value': float(mp[discord_idx]) if len(mp) > discord_idx and np.isfinite(mp[discord_idx]) else 0
            }
        except Exception as e:
            return {
                'matrix_profile': np.zeros_like(signal_data),
                'anomaly_score': np.zeros_like(signal_data),
                'motif_idx': np.zeros(max(1, len(signal_data) - window_size + 1), dtype=int),
                'discord_idx': 0,
                'discord_value': 0,
                'error': str(e)
            }
    
    def _egarch_volatility_optimized(self, returns: np.ndarray) -> Dict[str, Any]:
        """Optimized EGARCH Volatility Modeling"""
        try:
            returns_clean = returns[np.isfinite(returns)] * 100
            if len(returns_clean) < 50:
                vol = np.full_like(returns, np.std(returns))
                return {
                    'conditional_volatility': vol,
                    'leverage_effect': 0.0,
                }
            
            # Fit EGARCH model
            model = arch_model(returns_clean, vol='EGARCH', p=1, o=1, q=1, rescale=False)
            res = model.fit(disp='off', show_warning=False)
            
            conditional_vol = res.conditional_volatility / 100
            
            # Pad if necessary
            if len(conditional_vol) < len(returns):
                padding = np.full(len(returns) - len(conditional_vol), conditional_vol[0])
                conditional_vol = np.concatenate([padding, conditional_vol])
            
            return {
                'conditional_volatility': conditional_vol[:len(returns)],
                'leverage_effect': float(res.params.get('gamma[1]', 0.0)),
                'summary': str(res.summary())
            }
        except Exception as e:
            vol = np.full_like(returns, np.std(returns))
            return {
                'conditional_volatility': vol,
                'leverage_effect': 0.0,
                'error': str(e)
            }
    
    def _hidden_markov_model_optimized(self, returns: np.ndarray, n_states: int = 3) -> Dict[str, Any]:
        """Optimized Hidden Markov Model implementation"""
        try:
            returns_clean = returns[np.isfinite(returns)]
            
            if len(returns_clean) < 100:
                return self._hmm_fallback(returns, n_states)
            
            # Scale returns
            returns_std = np.std(returns_clean)
            if returns_std == 0 or returns_std < 1e-8:
                return self._hmm_fallback(returns, n_states)
            
            # Normalize returns
            returns_normalized = returns_clean / returns_std
            X = returns_normalized.reshape(-1, 1)
            
            # Try multiple initialization strategies
            best_model = None
            best_score = -np.inf
            
            for init_method in ['kmeans', 'random']:
                try:
                    hmm_model = hmm.GaussianHMM(
                        n_components=n_states,
                        covariance_type="diag",
                        random_state=42,
                        n_iter=200,
                        tol=1e-4,
                        init_params=init_method,
                        params='stmc',
                        implementation='log'
                    )
                    
                    # Set initial parameters for kmeans
                    if init_method == 'kmeans':
                        percentiles = np.linspace(0, 100, n_states + 1)
                        means = []
                        for i in range(n_states):
                            low = np.percentile(returns_normalized, percentiles[i])
                            high = np.percentile(returns_normalized, percentiles[i + 1])
                            means.append((low + high) / 2)
                        
                        hmm_model.means_ = np.array(means).reshape(-1, 1)
                        hmm_model.covars_ = np.array([[[np.var(returns_normalized) / n_states]] for _ in range(n_states)])
                        
                        # Transition matrix
                        trans_prob = 0.3 / (n_states - 1)
                        hmm_model.transmat_ = np.full((n_states, n_states), trans_prob)
                        np.fill_diagonal(hmm_model.transmat_, 0.7)
                        
                        hmm_model.startprob_ = np.ones(n_states) / n_states
                    
                    # Fit model
                    hmm_model.fit(X)
                    
                    if hmm_model.monitor_.converged:
                        score = hmm_model.score(X)
                        if score > best_score:
                            best_score = score
                            best_model = hmm_model
                    
                except Exception as e:
                    continue
            
            if best_model is None:
                return self._hmm_fallback(returns, n_states)
            
            # Use best model for predictions
            hmm_model = best_model
            
            # Predict states
            X_full = returns.reshape(-1, 1) / returns_std
            states = hmm_model.predict(X_full)
            state_probs = hmm_model.predict_proba(X_full)
            
            # Rescale means and variances
            means = hmm_model.means_.flatten() * returns_std
            
            if hmm_model.covariance_type == 'diag':
                variances = hmm_model.covars_.reshape(n_states, -1).flatten() * (returns_std ** 2)
            else:
                variances = np.array([np.diag(cov)[0] for cov in hmm_model.covars_]) * (returns_std ** 2)
            
            # Calculate stationary distribution
            try:
                eigenvalues, eigenvectors = np.linalg.eig(hmm_model.transmat_.T)
                stationary_idx = np.argmax(np.real(eigenvalues))
                stationary = np.real(eigenvectors[:, stationary_idx])
                stationary = np.abs(stationary) / np.sum(np.abs(stationary))
            except:
                stationary = np.ones(n_states) / n_states
            
            # Sort states by mean return
            sort_idx = np.argsort(means)
            means = means[sort_idx]
            variances = variances[sort_idx]
            
            # Remap states
            state_map = {old: new for new, old in enumerate(sort_idx)}
            states = np.array([state_map[s] for s in states])
            state_probs = state_probs[:, sort_idx]
            
            # Label states
            if n_states == 3:
                state_labels = ['Bear', 'Neutral', 'Bull']
            else:
                state_labels = []
                for i in range(n_states):
                    if i < n_states // 3:
                        state_labels.append(f'Bear_{i}')
                    elif i > 2 * n_states // 3:
                        state_labels.append(f'Bull_{i}')
                    else:
                        state_labels.append(f'Neutral_{i}')
            
            return {
                'states': states,
                'state_probabilities': state_probs,
                'transition_matrix': hmm_model.transmat_[sort_idx][:, sort_idx],
                'means': means,
                'variances': variances,
                'stationary_distribution': stationary[sort_idx],
                'state_labels': state_labels,
                'converged': True,
                'log_likelihood': float(best_score)
            }
            
        except Exception as e:
            return self._hmm_fallback(returns, n_states)
    
    def _hmm_fallback(self, returns: np.ndarray, n_states: int = 3) -> Dict[str, Any]:
        """Fallback HMM results when model fails"""
        n = len(returns)
        
        # Simple regime detection based on quantiles
        if n_states == 3:
            thresholds = [np.percentile(returns, 33), np.percentile(returns, 67)]
            states = np.zeros(n, dtype=int)
            states[returns < thresholds[0]] = 0
            states[(returns >= thresholds[0]) & (returns < thresholds[1])] = 1
            states[returns >= thresholds[1]] = 2
            state_labels = ['Bear', 'Neutral', 'Bull']
        else:
            percentiles = np.linspace(0, 100, n_states + 1)
            states = np.zeros(n, dtype=int)
            state_labels = []
            
            for i in range(n_states):
                low = np.percentile(returns, percentiles[i])
                high = np.percentile(returns, percentiles[i + 1])
                mask = (returns >= low) & (returns < high)
                if i == n_states - 1:
                    mask = returns >= low
                states[mask] = i
                
                if i < n_states // 3:
                    state_labels.append(f'Bear_{i}')
                elif i > 2 * n_states // 3:
                    state_labels.append(f'Bull_{i}')
                else:
                    state_labels.append(f'Neutral_{i}')
        
        # Calculate means and variances
        means = np.zeros(n_states)
        variances = np.zeros(n_states)
        
        for i in range(n_states):
            state_returns = returns[states == i]
            if len(state_returns) > 0:
                means[i] = np.mean(state_returns)
                variances[i] = np.var(state_returns)
            else:
                means[i] = 0
                variances[i] = np.var(returns)
        
        # Simple transition matrix
        transition_matrix = np.zeros((n_states, n_states))
        for i in range(len(states) - 1):
            transition_matrix[states[i], states[i + 1]] += 1
        
        # Normalize rows
        for i in range(n_states):
            row_sum = np.sum(transition_matrix[i])
            if row_sum > 0:
                transition_matrix[i] /= row_sum
            else:
                transition_matrix[i] = np.ones(n_states) / n_states
        
        # State probabilities
        state_probs = np.zeros((n, n_states))
        for i in range(n):
            state_probs[i, states[i]] = 1.0
        
        # Stationary distribution
        state_counts = np.bincount(states, minlength=n_states)
        stationary_dist = state_counts / n
        
        return {
            'states': states,
            'state_probabilities': state_probs,
            'transition_matrix': transition_matrix,
            'means': means,
            'variances': variances,
            'stationary_distribution': stationary_dist,
            'state_labels': state_labels,
            'converged': False,
            'log_likelihood': -np.inf
        }
    
    def _prepare_vecm_data(self, timeframe: str) -> Optional[pd.DataFrame]:
        """Prepare data for VECM analysis"""
        try:
            # Price data
            price_df = self.kline_data[timeframe][['timestamp', 'close']].set_index('timestamp')
            
            # Order flow data
            of_data = self.collected_data.get(timeframe, {}).get('order_flow', [])
            if of_data:
                of_df = pd.DataFrame(of_data)
                of_df['timestamp'] = pd.to_datetime(of_df['timestamp'], unit='ms')
                of_df['imbalance'] = of_df['total_bid'] - of_df['total_ask']
                of_df = of_df.groupby('timestamp')['imbalance'].mean().to_frame(name='orderflow')
            else:
                of_df = pd.DataFrame(columns=['orderflow'])
            
            # CVD data
            cvd_data = self.collected_data.get(timeframe, {}).get('cvd', [])
            if cvd_data:
                cvd_df = pd.DataFrame(cvd_data)
                cvd_df['timestamp'] = pd.to_datetime(cvd_df['timestamp'], unit='ms')
                cvd_df = cvd_df.groupby('timestamp')['cvd_close'].mean().to_frame(name='cvd')
            else:
                cvd_df = pd.DataFrame(columns=['cvd'])
            
            # Join dataframes
            vecm_data = price_df.join(of_df, how='inner').join(cvd_df, how='inner')
            vecm_data = vecm_data.dropna()
            
            return vecm_data if len(vecm_data) > 20 else None
            
        except Exception as e:
            self.log_error("VECM Prep", {"timeframe": timeframe}, str(e), 
                         "VECM data preparation", exc_info=True)
            return None
    
    def perform_tensor_fusion_analysis(self):
        """Perform multi-modal tensor fusion analysis"""
        try:
            tensor_data = {}
            
            # Combine kline data with market microstructure
            for timeframe in self.timeframes:
                if timeframe in self.kline_data:
                    df = self.kline_data[timeframe].copy()
                    
                    # Add order flow imbalance
                    if timeframe in self.collected_data and 'order_flow' in self.collected_data[timeframe]:
                        order_flow_df = pd.DataFrame(self.collected_data[timeframe]['order_flow'])
                        if not order_flow_df.empty and 'total_bid' in order_flow_df.columns:
                            imbalance = (order_flow_df['total_bid'] - order_flow_df['total_ask']) / \
                                       (order_flow_df['total_bid'] + order_flow_df['total_ask'] + 1e-10)
                            if len(imbalance) <= len(df):
                                df.loc[:len(imbalance)-1, 'orderflow_imbalance'] = imbalance.values
                    
                    # Add book pressure
                    if timeframe in self.collected_data and 'orderbook' in self.collected_data[timeframe]:
                        orderbook_df = pd.DataFrame(self.collected_data[timeframe]['orderbook'])
                        if not orderbook_df.empty and 'imbalance' in orderbook_df.columns:
                            if len(orderbook_df) <= len(df):
                                df.loc[:len(orderbook_df)-1, 'book_pressure'] = orderbook_df['imbalance'].values
                    
                    # Add CVD
                    if timeframe in self.collected_data and 'cvd' in self.collected_data[timeframe]:
                        cvd_df = pd.DataFrame(self.collected_data[timeframe]['cvd'])
                        if not cvd_df.empty and 'cvd_close' in cvd_df.columns:
                            if len(cvd_df) <= len(df):
                                df.loc[:len(cvd_df)-1, 'cvd'] = cvd_df['cvd_close'].values
                    
                    tensor_data[timeframe] = df
            
            # Create unified tensor
            if tensor_data:
                market_tensor = self.tensor_fusion.create_unified_market_tensor(tensor_data)
                
                # Perform tensor decomposition
                decomposition_results = self.tensor_fusion.tensor_decomposition_prediction(market_tensor)
                
                self.analysis_results['tensor_fusion'] = {
                    'tensor_shape': market_tensor.shape,
                    'decomposition_rank': 10,
                    'temporal_patterns': decomposition_results['temporal_patterns'].tolist(),
                    'factor_predictions': decomposition_results['predictions'],
                    'reconstruction_error': decomposition_results['reconstruction_error'],
                    'energy_conservation_ratio': decomposition_results.get('energy_conservation_ratio', 1.0)
                }
                
                print(f"  ✓ Tensor shape: {market_tensor.shape}")
                print(f"  ✓ Reconstruction error: {decomposition_results['reconstruction_error']:.4f}")
                
        except Exception as e:
            self.log_error("TensorFusion", {}, str(e), "Tensor Fusion Analysis", exc_info=True)
            self.analysis_results['tensor_fusion'] = {
                'error': str(e),
                'tensor_shape': (0, 0, 0),
                'reconstruction_error': 1.0
            }
    
    def perform_information_flow_analysis(self):
        """Analyze information flow between different data sources"""
        try:
            info_flow_data = {}
            
            # Use 1h as reference
            if '1h' in self.kline_data:
                ref_df = self.kline_data['1h']
                info_flow_data['kline_price'] = ref_df['close']
                
                # Add order flow
                if '1h' in self.collected_data and 'order_flow' in self.collected_data['1h']:
                    order_flow_df = pd.DataFrame(self.collected_data['1h']['order_flow'])
                    if not order_flow_df.empty and 'total_bid' in order_flow_df.columns:
                        imbalance = (order_flow_df['total_bid'] - order_flow_df['total_ask']) / \
                                   (order_flow_df['total_bid'] + order_flow_df['total_ask'] + 1e-10)
                        info_flow_data['orderflow'] = pd.Series(imbalance.values[:len(ref_df)], 
                                                               index=ref_df.index[:len(imbalance)])
                
                # Add orderbook pressure
                if '1h' in self.collected_data and 'orderbook' in self.collected_data['1h']:
                    orderbook_df = pd.DataFrame(self.collected_data['1h']['orderbook'])
                    if not orderbook_df.empty and 'imbalance' in orderbook_df.columns:
                        info_flow_data['orderbook_pressure'] = pd.Series(
                            orderbook_df['imbalance'].values[:len(ref_df)], 
                            index=ref_df.index[:len(orderbook_df)]
                        )
                
                # Add liquidations
                if '1h' in self.collected_data and 'liquidation' in self.collected_data['1h']:
                    liq_df = pd.DataFrame(self.collected_data['1h']['liquidation'])
                    if not liq_df.empty and 'total_value' in liq_df.columns:
                        info_flow_data['liquidations'] = pd.Series(
                            liq_df['total_value'].values[:len(ref_df)],
                            index=ref_df.index[:len(liq_df)]
                        )
                
                # Add CVD
                if '1h' in self.collected_data and 'cvd' in self.collected_data['1h']:
                    cvd_df = pd.DataFrame(self.collected_data['1h']['cvd'])
                    if not cvd_df.empty and 'cvd_close' in cvd_df.columns:
                        info_flow_data['cvd'] = pd.Series(
                            cvd_df['cvd_close'].values[:len(ref_df)],
                            index=ref_df.index[:len(cvd_df)]
                        )
                
                # Add market orders
                if '1h' in self.collected_data and 'market_orders' in self.collected_data['1h']:
                    mo_df = pd.DataFrame(self.collected_data['1h']['market_orders'])
                    if not mo_df.empty:
                        volume_data = mo_df[mo_df['metric_type'] == 'volume']
                        if not volume_data.empty and 'net_volume' in volume_data.columns:
                            info_flow_data['market_orders'] = pd.Series(
                                volume_data['net_volume'].values[:len(ref_df)],
                                index=ref_df.index[:len(volume_data)]
                            )
            
            # Calculate transfer entropy matrix
            if len(info_flow_data) > 1:
                te_results = self.info_flow_analyzer.calculate_transfer_entropy_matrix(info_flow_data)
                pid_results = self.info_flow_analyzer.partial_information_decomposition(info_flow_data)
                
                # Add transfer entropy spectrum for key pairs
                te_spectrums = {}
                if 'orderflow' in info_flow_data and 'kline_price' in info_flow_data:
                    te_spectrums['orderflow_to_price'] = self.advanced_tools.transfer_entropy_spectrum(
                        info_flow_data['orderflow'].values,
                        info_flow_data['kline_price'].values
                    )
                
                if 'cvd' in info_flow_data and 'kline_price' in info_flow_data:
                    te_spectrums['cvd_to_price'] = self.advanced_tools.transfer_entropy_spectrum(
                        info_flow_data['cvd'].values,
                        info_flow_data['kline_price'].values
                    )
                
                self.analysis_results['information_flow'] = {
                    'transfer_entropy': te_results['transfer_entropy'].to_dict(),
                    'optimal_lags': te_results['optimal_lags'].to_dict(),
                    'information_contributions': pid_results.get('individual_contributions', {}),
                    'total_information': pid_results.get('total_information', 0),
                    'redundancy_factor': pid_results.get('redundancy_factor', 0),
                    'synergy_potential': pid_results.get('synergy_potential', 0),
                    'transfer_entropy_spectrums': te_spectrums,
                    'network_metrics': te_results.get('network_metrics', {})
                }
                
                print(f"  ✓ Analyzed {len(info_flow_data)} data streams")
                print(f"  ✓ Total information content: {pid_results.get('total_information', 0):.4f}")
            else:
                print("  ✗ Insufficient data for information flow analysis")
                
        except Exception as e:
            self.log_error("InfoFlow", {}, str(e), "Information Flow Analysis", exc_info=True)
            self.analysis_results['information_flow'] = {'error': str(e)}
    
    def perform_microstructure_bridge_analysis(self):
        """Analyze microstructure to kline bridge"""
        try:
            for timeframe in self.timeframes:
                if timeframe not in self.kline_data:
                    continue
                
                df = self.kline_data[timeframe]
                if df.empty:
                    continue
                
                current_bar = {
                    'open': float(df['open'].iloc[-1]),
                    'high': float(df['high'].iloc[-1]),
                    'low': float(df['low'].iloc[-1]),
                    'current': float(df['close'].iloc[-1]),
                    'time_remaining_ratio': 0.3
                }
                
                # Get orderflow data
                orderflow_data = pd.DataFrame()
                if timeframe in self.collected_data and 'order_flow' in self.collected_data[timeframe]:
                    of_raw = pd.DataFrame(self.collected_data[timeframe]['order_flow'])
                    if not of_raw.empty:
                        of_raw['timestamp'] = pd.to_datetime(of_raw['timestamp'], unit='ms')
                        of_raw['delta'] = of_raw['total_bid'] - of_raw['total_ask']
                        of_raw['volume'] = of_raw['total_bid'] + of_raw['total_ask']
                        
                        # Align with kline
                        kline_interval = df['timestamp'].diff().min()
                        if pd.notna(kline_interval):
                           of_resampled = of_raw.set_index('timestamp').resample(kline_interval).agg({
                               'delta': 'sum', 'volume': 'sum'
                           }).reindex(df['timestamp'], method='ffill').fillna(0)
                           
                           orderflow_data = pd.concat([
                               df[['timestamp', 'close']].rename(columns={'close': 'price'}), 
                               of_resampled.reset_index(drop=True)
                           ], axis=1).set_index('timestamp')
                           orderflow_data = orderflow_data.dropna()
                
                # Get orderbook data
                book_data = pd.DataFrame()
                if timeframe in self.collected_data and 'orderbook' in self.collected_data[timeframe]:
                    book_data = pd.DataFrame(self.collected_data[timeframe]['orderbook'])
                    if not book_data.empty:
                        book_data['bid_size'] = book_data['buy_usd']
                        book_data['ask_size'] = book_data['sell_usd']
                        book_data['bid'] = df['close'].iloc[-1] * 0.999
                        book_data['ask'] = df['close'].iloc[-1] * 1.001
                
                # Make prediction
                prediction = self.microstructure_bridge.predict_current_bar_close(
                    current_bar, orderflow_data.reset_index(), book_data
                )
                
                if timeframe not in self.analysis_results:
                    self.analysis_results[timeframe] = {}
                
                self.analysis_results[timeframe]['microstructure_prediction'] = prediction
                
                print(f"  ✓ {timeframe} prediction: {prediction.get('ensemble', 0):.2f} "
                      f"(confidence: {prediction.get('confidence', 0):.1f}%)")
                      
        except Exception as e:
            self.log_error("MicrostructureBridge", {}, str(e), 
                         "Microstructure Bridge Analysis", exc_info=True)
    
    def perform_tda_analysis(self):
        """Perform Topological Data Analysis"""
        try:
            if '1h' in self.kline_data and not self.kline_data['1h'].empty:
                df_1h = self.kline_data['1h'].copy()
                
                # Prepare features
                df_1h['returns'] = df_1h['close'].pct_change().fillna(0)
                df_1h['volatility'] = df_1h['returns'].rolling(window=10).std().fillna(0)
                df_1h['momentum'] = df_1h['close'].diff(5).fillna(0)
                
                tda_data = df_1h[['close', 'returns', 'volatility', 'momentum']].dropna()
                
                if len(tda_data) > 50:
                    print("  ✓ Running TDA on 1h price, returns, and volatility...")
                    tda_results = self.tda_analyzer.analyze_market_shape(
                        data=tda_data,
                        filter_col='close',
                        feature_cols=['returns', 'volatility']
                    )
                    self.analysis_results['tda_analysis'] = tda_results
                    if tda_results.get('error'):
                         self.log_error("TDA", {}, tda_results['error'], "TDA Analysis")
                    else:
                        print(f"  ✓ TDA Market Shape: {tda_results.get('shape_interpretation')}")
                else:
                    self.analysis_results['tda_analysis'] = self.tda_analyzer._default_tda_results(
                        "Insufficient data after preparation.")
            else:
                self.analysis_results['tda_analysis'] = self.tda_analyzer._default_tda_results(
                    "No 1h kline data available.")
        except Exception as e:
            self.log_error("TDA", {}, str(e), "TDA Analysis", exc_info=True)
            self.analysis_results['tda_analysis'] = self.tda_analyzer._default_tda_results(str(e))
    
    def analyze_order_flow_patterns_enhanced(self):
        """Enhanced order flow analysis across timeframes"""
        combined_results = {}
        
        for timeframe in self.timeframes:
            if timeframe not in self.collected_data or 'order_flow' not in self.collected_data[timeframe]:
                continue
            
            order_flow_data = pd.DataFrame(self.collected_data[timeframe]['order_flow'])
            if order_flow_data.empty:
                continue
            
            # Calculate imbalance
            if 'total_bid' in order_flow_data.columns and 'total_ask' in order_flow_data.columns:
                order_flow_data['imbalance'] = (order_flow_data['total_bid'] - order_flow_data['total_ask']) / \
                                               (order_flow_data['total_bid'] + order_flow_data['total_ask'] + 1e-10)
                
                # Aggregate by timestamp
                numeric_cols = ['total_ask', 'total_bid', 'imbalance']
                agg_dict = {col: 'mean' for col in numeric_cols if col in order_flow_data.columns}
                
                if agg_dict:
                    agg_flow = order_flow_data.groupby('timestamp').agg(agg_dict)
                    
                    combined_results[timeframe] = {
                        'mean_imbalance': agg_flow['imbalance'].mean() if 'imbalance' in agg_flow else 0,
                        'imbalance_std': agg_flow['imbalance'].std() if 'imbalance' in agg_flow else 0,
                        'bid_ask_ratio': agg_flow['total_bid'].sum() / (agg_flow['total_ask'].sum() + 1e-10) 
                                       if 'total_bid' in agg_flow and 'total_ask' in agg_flow else 1,
                        'total_volume': (agg_flow.get('total_bid', 0).sum() + 
                                       agg_flow.get('total_ask', 0).sum()),
                        'imbalance_trend': np.polyfit(range(len(agg_flow)), agg_flow['imbalance'].values, 1)[0] 
                                         if len(agg_flow) > 1 and 'imbalance' in agg_flow else 0
                    }
        
        self.analysis_results['order_flow_analysis'] = combined_results
    
    def analyze_market_positioning_enhanced(self):
        """Enhanced positioning analysis across timeframes"""
        combined_results = {}
        
        for timeframe in self.timeframes:
            if timeframe not in self.collected_data or 'positioning' not in self.collected_data[timeframe]:
                continue
            
            positioning_data = pd.DataFrame(self.collected_data[timeframe]['positioning'])
            if positioning_data.empty or 'ls_ratio' not in positioning_data.columns:
                continue
            
            # Separate by type
            types = positioning_data['type'].unique() if 'type' in positioning_data else []
            
            timeframe_results = {}
            
            for pos_type in types:
                type_data = positioning_data[positioning_data['type'] == pos_type]
                if not type_data.empty:
                    ls_ratios = type_data['ls_ratio'].values
                    
                    # Calculate metrics
                    avg_ls_ratio = np.mean(ls_ratios)
                    
                    # Calculate trend
                    if len(ls_ratios) > 1:
                        ls_ratio_trend = np.polyfit(range(len(ls_ratios)), ls_ratios, 1)[0]
                    else:
                        ls_ratio_trend = 0
                    
                    timeframe_results[pos_type] = {
                        'avg_ratio': avg_ls_ratio,
                        'trend': ls_ratio_trend,
                        'volatility': np.std(ls_ratios),
                        'current_ratio': ls_ratios[-1] if len(ls_ratios) > 0 else 1
                    }
            
            # Overall metrics
            all_ratios = positioning_data['ls_ratio'].values
            avg_ls_ratio = np.mean(all_ratios)
            
            combined_results[timeframe] = {
                'avg_long_short_ratio': avg_ls_ratio,
                'ls_ratio_trend': np.polyfit(range(len(all_ratios)), all_ratios, 1)[0] if len(all_ratios) > 1 else 0,
                'market_sentiment': 'bullish' if avg_ls_ratio > 1.2 else 
                                  'bearish' if avg_ls_ratio < 0.8 else 'neutral',
                'positioning_volatility': np.std(all_ratios),
                'by_type': timeframe_results
            }
        
        self.analysis_results['positioning_analysis'] = combined_results
    
    def analyze_liquidation_cascades_enhanced(self):
        """Enhanced liquidation analysis across timeframes"""
        combined_results = {}
        
        for timeframe in self.timeframes:
            if timeframe not in self.collected_data or 'liquidation' not in self.collected_data[timeframe]:
                continue
            
            liquidation_data = pd.DataFrame(self.collected_data[timeframe]['liquidation'])
            if liquidation_data.empty or 'total_value' not in liquidation_data.columns:
                continue
            
            # Identify cascade events
            liquidation_data['total_change'] = liquidation_data['total_value'].pct_change()
            cascade_threshold = liquidation_data['total_change'].std() * 2
            
            cascades = liquidation_data[
                liquidation_data['total_change'].abs() > cascade_threshold
            ] if cascade_threshold > 0 else pd.DataFrame()
            
            # Calculate metrics
            timeframe_results = {
                'avg_liquidation_value': liquidation_data['total_value'].mean(),
                'liquidation_volatility': liquidation_data['total_value'].std(),
                'cascade_events': len(cascades),
                'avg_ls_liquidation_ratio': liquidation_data['ls_ratio'].mean() if 'ls_ratio' in liquidation_data else 1,
                'max_liquidation_spike': liquidation_data['total_value'].max(),
                'total_liquidation_volume': liquidation_data['total_value'].sum()
            }
            
            # Add long/short analysis
            if 'long_value' in liquidation_data and 'short_value' in liquidation_data:
                timeframe_results['long_liquidations'] = liquidation_data['long_value'].sum()
                timeframe_results['short_liquidations'] = liquidation_data['short_value'].sum()
                timeframe_results['liquidation_bias'] = 'long' if timeframe_results['long_liquidations'] > timeframe_results['short_liquidations'] else 'short'
            
            combined_results[timeframe] = timeframe_results
        
        self.analysis_results['liquidation_analysis'] = combined_results
    
    def load_previous_analysis(self) -> Optional[Dict[str, Any]]:
        """Load the most recent previous analysis file for comparison"""
        try:
            list_of_files = list(self.results_dir.glob('**/*.json'))
            if not list_of_files:
                print("  → No previous analysis found.")
                return None
            
            # Sort files by name to find the latest
            latest_file = sorted(list_of_files)[-1]
            
            # Ensure we don't load the current run's file
            if latest_file.resolve() == self.results_file.resolve():
                 if len(list_of_files) > 1:
                     latest_file = sorted(list_of_files)[-2]
                 else:
                     print("  → Only current analysis file exists.")
                     return None
            
            print(f"  → Loading previous analysis from: {latest_file.name}")
            with open(latest_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.log_error("LoadPrevious", {}, str(e), 
                         "Failed to load previous analysis file", exc_info=True)
            return None
    
    def compare_with_previous(self, prev_analysis: Dict[str, Any]) -> List[str]:
        """Compare with previous analysis"""
        report = []
        if not prev_analysis:
            return ["No previous analysis to compare against."]
        
        # Level 1: Statistical Changes
        report.append("--- Level 1: Statistical & Data Changes ---")
        
        # Compare API calls
        prev_stats = prev_analysis.get('stats', {})
        prev_reqs = prev_stats.get('successful_requests', 0)
        curr_reqs = self.stats.get('successful_requests', 0)
        if abs(prev_reqs - curr_reqs) > 5:
            report.append(f"Significant change in API response count: {prev_reqs} → {curr_reqs}")
        
        # Compare price stats
        prev_1h_prices = []
        if 'kline_data' in prev_analysis and '1h' in prev_analysis['kline_data']:
            prev_1h_data = prev_analysis['kline_data']['1h']
            if isinstance(prev_1h_data, list) and prev_1h_data:
                prev_1h_prices = [float(d.get('close', 0)) for d in prev_1h_data if 'close' in d]
        
        curr_1h_price = self.kline_data.get('1h', pd.DataFrame()).get('close', pd.Series()).tolist()
        
        if prev_1h_prices and curr_1h_price:
            prev_std = np.std(prev_1h_prices)
            curr_std = np.std(curr_1h_price)
            if curr_std > prev_std * 1.5:
                report.append("IMPORTANT: Price moved > 1.5 standard deviations from last analysis.")
        
        # Compare volatility regime
        prev_vol_regime = prev_analysis.get('analysis_results', {}).get('market_state', {}).get('state')
        curr_vol_regime = self._assess_market_state().get('state')
        if prev_vol_regime and curr_vol_regime and prev_vol_regime != curr_vol_regime:
            report.append(f"CRITICAL: Volatility regime switch: {prev_vol_regime} → {curr_vol_regime}")
        
        # Level 2: Pattern Changes
        report.append("\n--- Level 2: Pattern Changes ---")
        
        # Trend reversal
        prev_bias = prev_analysis.get('analysis_results', {}).get('directional_bias', {}).get('direction')
        curr_bias = self._calculate_directional_bias(self.analysis_results.get('1h', {})).get('direction')
        if prev_bias and curr_bias and prev_bias != curr_bias:
            report.append(f"Major Trend Reversal: {prev_bias} → {curr_bias}")
        
        # Level 3: Model Changes
        report.append("\n--- Level 3: Model Output Changes ---")
        
        # Confidence changes
        prev_conf = 100
        curr_conf = 100
        for tf in self.timeframes:
            if tf in self.analysis_results and 'microstructure_prediction' in self.analysis_results[tf]:
                curr_conf = min(curr_conf, self.analysis_results[tf]['microstructure_prediction'].get('confidence', 100))
        
        if prev_conf - curr_conf > 10:
            report.append(f"Model confidence dropped >10%: {prev_conf:.1f}% → {curr_conf:.1f}%")
        
        return report
    
    def generate_comprehensive_summary_enhanced(self):
        """Generate enhanced comprehensive summary"""
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            f.write(f"COMPREHENSIVE MATHEMATICAL ANALYSIS SUMMARY - {self.version.upper()}\n")
            f.write("="*80 + "\n\n")
            f.write(f"Analysis Date: {self.current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC\n")
            f.write(f"Analyst: {self.analyst}\n\n")
            
            all_targets = {}
            for timeframe in self.timeframes:
                if timeframe in self.analysis_results and timeframe in self.kline_data:
                    all_targets[timeframe] = self.calculate_exact_targets_timeframe(timeframe)
            
            # 1. Inter-Analysis Comparison
            if self.previous_analysis:
                f.write("1. INTER-ANALYSIS COMPARISON (CHANGES SINCE LAST RUN)\n")
                f.write("-" * 60 + "\n")
                comparison_report = self.compare_with_previous(self.previous_analysis)
                for line in comparison_report:
                    f.write(f"  - {line}\n")
            else:
                f.write("1. INTER-ANALYSIS COMPARISON\n")
                f.write("-" * 60 + "\n")
                f.write("  - No previous analysis found. This is the baseline run.\n")
            
            # 2. Market Vitals Dashboard
            f.write("\n\n2. MARKET VITALS DASHBOARD (1h Reference)\n")
            f.write("-" * 50 + "\n")
            self._write_market_vitals_dashboard(f)
            
            # 3. Per-Timeframe Projections
            f.write("\n\n3. MATHEMATICAL PROJECTIONS (PER TIMEFRAME)\n")
            f.write("-" * 50 + "\n")
            for timeframe in self.timeframes:
                f.write(f"\n========== {timeframe} ANALYSIS ==========\n")
                if timeframe in all_targets:
                    targets = all_targets[timeframe]
                    current_price = targets.get('current_price', 0)
                    consensus_target = targets.get('consensus_target', 0)
                    confidence = targets.get('confidence', 0)
                    direction = "BULLISH" if consensus_target > current_price else "BEARISH" if consensus_target < current_price else "NEUTRAL"
                    
                    f.write(f"  Overall Assessment:\n")
                    f.write(f"    - Direction: {direction}\n")
                    f.write(f"    - Consensus Target: {consensus_target:.4f}\n")
                    f.write(f"    - Confidence: {confidence:.2f}%\n")
                    f.write(f"    - Current Price: {current_price:.4f}\n\n")
                    
                    f.write(f"  Component Projections:\n")
                    f.write(f"    - Fibonacci Target:       {targets.get('fibonacci_target', 0):.4f}\n")
                    f.write(f"    - Harmonic Target:        {targets.get('harmonic_target', 0):.4f}\n")
                    f.write(f"    - Volatility Target:      {targets.get('volatility_target', 0):.4f}\n")
                    f.write(f"    - Wave Structure Target:  {targets.get('wave_target', 0):.4f}\n")
                    f.write(f"    - ML Forecast:            {targets.get('ml_forecast', 0):.4f}\n")
                    
                    if 'microstructure_prediction' in self.analysis_results.get(timeframe, {}):
                        pred = self.analysis_results[timeframe]['microstructure_prediction']
                        f.write(f"    - Microstructure Close:   {pred.get('ensemble', 0):.4f} (Conf: {pred.get('confidence', 0):.1f}%)\n")
                    
                    f.write("\n  Key Mathematical Indicators:\n")
                    if 'multifractal' in self.analysis_results.get(timeframe, {}):
                        mf = self.analysis_results[timeframe]['multifractal']
                        f.write(f"    - Multifractality: {mf['multifractality']:.4f} (Hurst: {mf['dominant_hurst']:.4f})\n")
                    
                    if 'rqa' in self.analysis_results.get(timeframe, {}):
                        rqa = self.analysis_results[timeframe]['rqa']
                        f.write(f"    - Market Determinism (RQA): {rqa['determinism']:.4f}\n")
                    
                    if 'hmm' in self.analysis_results.get(timeframe, {}):
                        hmm_data = self.analysis_results[timeframe]['hmm']
                        if hmm_data and hmm_data.get('states') is not None and len(hmm_data['states']) > 0:
                            current_state = hmm_data['states'][-1]
                            state_label = hmm_data['state_labels'][current_state]
                            f.write(f"    - Market Regime (HMM): {state_label}\n")
                else:
                    f.write("  No analysis results available for this timeframe.\n")
            
            # 4. Cross-Timeframe Synthesis
            f.write("\n\n4. CROSS-TIMEFRAME SYNTHESIS & CONCLUSION\n")
            f.write("="*50 + "\n")
            risk_metrics = self.calculate_risk_metrics()
            conclusion_text = self.generate_mathematical_conclusion(all_targets, risk_metrics)
            f.write(conclusion_text)
            
            # 5. Supporting Analyses
            f.write("\n\n5. SUPPORTING ANALYSES\n")
            f.write("="*50 + "\n")
            self._write_supporting_summary_sections(f, all_targets, risk_metrics)
            
            # Footer
            f.write(f"\n\nAnalysis Completed: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC\n")
            f.write(f"Total Processing Time: {time.time() - self.start_time:.2f} seconds\n")
            f.write(f"Data Points Collected: {self.stats['data_points_collected']:,}\n")
    
    def _write_market_vitals_dashboard(self, f):
        """Write market vitals dashboard section"""
        # Market State
        market_state = self._assess_market_state()
        f.write(f"  - Market State:         {market_state['state']} (Confidence: {market_state['confidence']:.1f}%)\n")
        
        # Directional Bias
        bias = self._calculate_directional_bias(self.analysis_results.get('1h', {}))
        f.write(f"  - Directional Bias:     {bias['direction']} (Strength: {bias['strength']:.1f}/10)\n")
        
        # Volatility
        vol = 0.0
        if '1h' in self.analysis_results and 'egarch' in self.analysis_results['1h']:
            vol = self.analysis_results['1h']['egarch']['conditional_volatility'][-1] * 100
        f.write(f"  - Conditional Vol (1h): {vol:.2f}%\n")
        
        # Liquidity
        liquidity = "N/A"
        if '1h' in self.analysis_results and 'kyle_lambda' in self.analysis_results['1h']:
            depth = self.analysis_results['1h']['kyle_lambda'].get('market_depth', np.inf)
            liquidity = f"~${depth/1e6:.2f}M per 1% slip" if np.isfinite(depth) else "Low"
        f.write(f"  - Market Depth:         {liquidity}\n")
        
        # Information Flow
        info_flow = "N/A"
        if 'information_flow' in self.analysis_results:
            total_info = self.analysis_results['information_flow'].get('total_information', 0)
            info_flow = f"{total_info:.2f} total bits"
        f.write(f"  - Information Flow:     {info_flow}\n")
        
        # Risk
        risk = self.calculate_risk_metrics()
        f.write(f"  - 1-day VaR (95%):      {risk['var_95']*100:.2f}%\n")
    
    def _write_supporting_summary_sections(self, f, all_targets, risk_metrics):
        """Write supporting analysis sections"""
        # Signal-to-Noise Ratio
        f.write("\nSignal-to-Noise Ratio:\n")
        f.write("-" * 40 + "\n")
        snr = self._calculate_signal_to_noise_ratio()
        snr_verdict = "High (Clear trends)" if snr > 0.7 else "Medium (Some chop)" if snr > 0.4 else "Low (Noisy)"
        f.write(f"  - Current Ratio: {snr:.2f} ({snr_verdict})\n")
        f.write("  - Implication: Measures the clarity of the underlying market trend.\n")
        
        # Leading vs. Lagging Indicators
        f.write("\nLeading vs. Lagging Indicators:\n")
        f.write("-" * 40 + "\n")
        leading_signal, lagging_signal = self._get_leading_lagging_signals()
        f.write(f"  - Leading Signal (Microstructure): {leading_signal}\n")
        f.write(f"  - Lagging Signal (Trend MA):       {lagging_signal}\n")
        if leading_signal != "NEUTRAL" and leading_signal == lagging_signal:
             f.write("  - Verdict: CONFIRMATION. Leading indicators align with the trend.\n")
        elif leading_signal != "NEUTRAL" and leading_signal != lagging_signal:
            f.write("  - Verdict: DIVERGENCE. Leading indicators suggest potential reversal.\n")
        else:
            f.write("  - Verdict: NEUTRAL. No clear divergence or confirmation.\n")
        
# Information Flow Analysis (continuing from where it cut off)
        f.write("\nInformation Flow Dynamics (1h Reference):\n")
        f.write("-" * 40 + "\n")
        if 'information_flow' in self.analysis_results and 'error' not in self.analysis_results['information_flow']:
            info_flow = self.analysis_results['information_flow']
            f.write("Top Information Flows to Price:\n")
            
            # Handle transfer entropy data
            te_data = info_flow.get('transfer_entropy', {})
            if isinstance(te_data, dict) and 'kline_price' in te_data:
                price_flows = te_data.get('kline_price', {})
                sorted_flows = sorted(price_flows.items(), key=lambda x: x[1], reverse=True)
                
                for source, te_value in sorted_flows:
                    if source != 'kline_price' and te_value > 0:
                        # Get lag from optimal_lags
                        lag_data = info_flow.get('optimal_lags', {})
                        if isinstance(lag_data, dict) and source in lag_data:
                            lag_value = lag_data[source].get('kline_price', 0)
                        else:
                            lag_value = 0
                        f.write(f"  - {source} → Price: TE={te_value:.4f} (leads by {lag_value} periods)\n")
            
            f.write(f"\nTotal Information Content: {info_flow.get('total_information', 0):.4f}\n")
            
            # Network metrics
            network_metrics = info_flow.get('network_metrics', {})
            if network_metrics:
                f.write(f"Network Density: {network_metrics.get('density', 0):.3f}\n")
                if 'information_hubs' in network_metrics:
                    f.write("Information Hubs: ")
                    hubs = network_metrics['information_hubs'][:3]
                    f.write(", ".join([f"{hub[0]} ({hub[1]:.3f})" for hub in hubs]) + "\n")
        else:
            f.write("  Analysis not available.\n")

        # Tensor Fusion Results
        f.write("\nMulti-Modal Tensor Fusion Analysis:\n")
        f.write("-" * 40 + "\n")
        if 'tensor_fusion' in self.analysis_results and 'error' not in self.analysis_results['tensor_fusion']:
            tf_results = self.analysis_results['tensor_fusion']
            f.write(f"  - Tensor Shape: {tf_results['tensor_shape']}\n")
            f.write(f"  - Pattern Clarity (1-Error): {(1 - tf_results['reconstruction_error'])*100:.2f}%\n")
            
            # Energy Conservation Check
            energy_ratio = tf_results.get('energy_conservation_ratio', 1.0)
            f.write(f"  - Energy Conservation: {energy_ratio*100:.2f}% ({'Conserved' if abs(1-energy_ratio) < 0.01 else 'Check Model'})\n")

            f.write("  - Strongest Factor Trends:\n")
            for factor, trend in sorted(tf_results.get('factor_predictions', {}).items(), 
                                       key=lambda item: abs(item[1]), reverse=True)[:3]:
                f.write(f"    - {factor}: {trend:+.4f}\n")
        else:
            f.write("  Analysis not available.\n")
        
        # TDA Results
        f.write("\nTopological Data Analysis (Market Shape):\n")
        f.write("-" * 40 + "\n")
        if 'tda_analysis' in self.analysis_results and not self.analysis_results['tda_analysis'].get('error'):
            tda = self.analysis_results['tda_analysis']
            f.write(f"  - Detected Shape: {tda['shape_interpretation']}\n")
            f.write(f"  - Graph Stats: {tda['num_nodes']} nodes, {tda['num_components']} components, ")
            f.write(f"{tda['num_cycles']} cycles, {tda['num_flares']} flares\n")
            f.write(f"  - Average Degree: {tda.get('avg_degree', 0):.2f}\n")
            
            # Top central nodes
            if tda.get('top_central_nodes'):
                f.write("  - Key Market States: ")
                top_nodes = tda['top_central_nodes'][:3]
                f.write(", ".join([f"Node {n[0]} (centrality: {n[1]:.3f})" for n in top_nodes]) + "\n")
        else:
            f.write(f"  Analysis failed: {self.analysis_results.get('tda_analysis', {}).get('error', 'Unknown')}\n")

        # Risk Assessment
        f.write("\nRisk Assessment (1h Reference):\n")
        f.write("-" * 40 + "\n")
        f.write(f"  - Value at Risk (95%): {risk_metrics['var_95']*100:.2f}%\n")
        f.write(f"  - Conditional VaR (95%): {risk_metrics['cvar_95']*100:.2f}%\n")
        f.write(f"  - Maximum Drawdown: {risk_metrics['max_drawdown']*100:.2f}%\n")
        f.write(f"  - Sharpe Ratio: {risk_metrics['sharpe_ratio']:.2f}\n")
        f.write(f"  - Information Ratio: {risk_metrics['info_ratio']:.2f}\n")

    def _calculate_signal_to_noise_ratio(self) -> float:
        """Calculate the Signal-to-Noise ratio for the market"""
        # Use EEMD results from 1h timeframe
        if '1h' not in self.analysis_results or 'eemd' not in self.analysis_results['1h']:
            return 0.5
        
        eemd_results = self.analysis_results['1h']['eemd']
        energies = eemd_results.get('energy_distribution', [])
        
        if not energies or len(energies) < 3:
            return 0.5

        # Signal is the energy in lower-frequency IMFs
        signal_energy = sum(energies[2:]) if len(energies) > 2 else 0
        # Noise is the energy in higher-frequency IMFs
        noise_energy = sum(energies[:2]) if len(energies) >= 2 else 1
        
        if noise_energy == 0:
            return 1.0
            
        snr = signal_energy / noise_energy
        # Normalize to 0-1 scale
        return 1 / (1 + 1/snr) if snr > 0 else 0

    def _get_leading_lagging_signals(self) -> Tuple[str, str]:
        """Compare leading and lagging indicators"""
        # Leading signal: Microstructure prediction
        leading_signal = "NEUTRAL"
        if '1h' in self.analysis_results and 'microstructure_prediction' in self.analysis_results['1h']:
            pred = self.analysis_results['1h']['microstructure_prediction'].get('ensemble', 0)
            if '1h' in self.kline_data and not self.kline_data['1h'].empty:
                current_price = self.kline_data['1h']['close'].iloc[-1]
                if pred > current_price * 1.001:
                    leading_signal = "BULLISH"
                elif pred < current_price * 0.999:
                    leading_signal = "BEARISH"

        # Lagging signal: 20/50 EMA cross on 4h
        lagging_signal = "NEUTRAL"
        if '4h' in self.kline_data:
            df = self.kline_data['4h']
            if len(df) > 50:
                ema20 = df['close'].ewm(span=20, adjust=False).mean().iloc[-1]
                ema50 = df['close'].ewm(span=50, adjust=False).mean().iloc[-1]
                if ema20 > ema50:
                    lagging_signal = "BULLISH"
                else:
                    lagging_signal = "BEARISH"
        
        return leading_signal, lagging_signal
    

    def _analyze_cvd_signal(self) -> str:
        """Analyze CVD signal for conclusion"""
        for timeframe in ['1h', '4h']:
            if timeframe in self.collected_data and 'cvd' in self.collected_data[timeframe]:
                cvd_data = pd.DataFrame(self.collected_data[timeframe]['cvd'])
                
                if not cvd_data.empty and 'cvd_close' in cvd_data.columns:
                    aggregate_cvd = cvd_data[cvd_data['exchange'] == 'AGGREGATE'] if 'exchange' in cvd_data.columns else cvd_data
                    
                    if not aggregate_cvd.empty:
                        cvd_values = aggregate_cvd['cvd_close'].values
                        if len(cvd_values) > 10:
                            cvd_trend = np.polyfit(range(len(cvd_values)), cvd_values, 1)[0]
                            latest_cvd = cvd_values[-1]
                            
                            if cvd_trend > 0 and latest_cvd > 0:
                                return "Positive CVD trend indicates net buying pressure"
                            elif cvd_trend < 0 and latest_cvd < 0:
                                return "Negative CVD trend indicates net selling pressure"
                            else:
                                return "Mixed CVD signals, market indecision"
        return None
    
    def _get_cvd_recommendation(self) -> str:
        """Generate CVD-based recommendation"""
        for timeframe in ['1h', '4h']:
            if timeframe in self.collected_data and 'cvd' in self.collected_data[timeframe]:
                cvd_data = pd.DataFrame(self.collected_data[timeframe]['cvd'])
                
                if not cvd_data.empty and 'cvd_close' in cvd_data.columns:
                    aggregate_cvd = cvd_data[cvd_data['exchange'] == 'AGGREGATE'] if 'exchange' in cvd_data.columns else cvd_data
                    
                    if not aggregate_cvd.empty:
                        cvd_values = aggregate_cvd['cvd_close'].values
                        if len(cvd_values) > 20:
                            recent_cvd = cvd_values[-10:].mean()
                            older_cvd = cvd_values[-20:-10].mean()
                            
                            if recent_cvd > older_cvd and recent_cvd > 0:
                                return "CVD shows accelerating buying pressure."
                            elif recent_cvd < older_cvd and recent_cvd < 0:
                                return "CVD shows accelerating selling pressure."
        return None
    
    def _calculate_key_levels(self, targets: Dict[str, float]) -> Dict[str, float]:
        """Calculate key mathematical levels"""
        
        current_price = targets.get('current_price', 0)
        if current_price == 0:
            return {}
        
        # Calculate standard levels
        levels = {
            'Immediate Support': current_price * 0.98,
            'Strong Support': current_price * 0.95,
            'Immediate Resistance': current_price * 1.02,
            'Strong Resistance': current_price * 1.05,
            'Pivot Point': current_price
        }
        
        # Add volatility-based levels (use 1h as reference)
        if '1h' in self.analysis_results and 'egarch' in self.analysis_results['1h']:
            vol_data = self.analysis_results['1h']['egarch']['conditional_volatility']
            if len(vol_data) >= 20:
                vol = np.mean(vol_data[-20:])
                levels['Vol-Adjusted Upper'] = current_price * (1 + 2 * vol)
                levels['Vol-Adjusted Lower'] = current_price * (1 - 2 * vol)
        
        # Add microstructure levels
        if '1h' in self.analysis_results and 'microstructure_prediction' in self.analysis_results['1h']:
            microprice = self.analysis_results['1h']['microstructure_prediction'].get('microprice', current_price)
            if microprice > 0:
                levels['Microprice Level'] = microprice
        
        # Add Kyle Lambda adjusted levels
        if '1h' in self.analysis_results and 'kyle_lambda' in self.analysis_results['1h']:
            kl = self.analysis_results['1h']['kyle_lambda']
            impact = abs(kl['lambda']) * 1000000  # Impact for $1M volume
            levels['Large Buy Impact'] = current_price * (1 + impact)
            levels['Large Sell Impact'] = current_price * (1 - impact)
        
        # Convert all to float
        return {k: float(v) for k, v in levels.items()}
    def calculate_exact_targets_timeframe(self, timeframe: str) -> Dict[str, float]:
        """Calculate exact targets for specific timeframe"""
        if timeframe not in self.kline_data:
            return self._default_targets()
        
        df = self.kline_data[timeframe]
        if df.empty:
            return self._default_targets()
            
        prices = df['close'].values
        current_price = float(prices[-1])
        
        # Adjust calculations based on timeframe
        tf_multiplier = {
            '15m': 0.25, '30m': 0.5, '1h': 1.0, '4h': 4.0, 
            '6h': 6.0, '12h': 12.0, '1d': 24.0
        }.get(timeframe, 1.0)
        
        # 1. Fibonacci Extension
        lookback = min(int(50 / tf_multiplier), len(prices) - 1)
        if lookback > 0:
            swing_high = np.max(prices[-lookback:])
            swing_low = np.min(prices[-lookback:])
            fib_range = swing_high - swing_low
            
            fib_target = swing_high + (fib_range * 0.618) if current_price > swing_high else swing_low - (fib_range * 0.618)
        else:
            fib_target = current_price * 1.05
        
        # 2. Harmonic Pattern Projection
        if timeframe in self.analysis_results and 'eemd' in self.analysis_results[timeframe]:
            imfs = self.analysis_results[timeframe]['eemd'].get('imfs', [])
            if imfs and len(imfs) > 0:
                dominant_cycle = np.mean([len(imf) for imf in imfs[:3] if len(imf) > 0])
                harmonic_projection = current_price * (1 + np.sin(2 * np.pi / max(dominant_cycle, 1)) * 0.1 * tf_multiplier)
            else:
                harmonic_projection = current_price * (1.05 * tf_multiplier)
        else:
            harmonic_projection = current_price * (1.05 * tf_multiplier)
        
        # 3. Volatility-Based Target
        if timeframe in self.analysis_results and 'egarch' in self.analysis_results[timeframe]:
            vol = self.analysis_results[timeframe]['egarch']['conditional_volatility']
            avg_vol = np.mean(vol[-20:]) if len(vol) >= 20 else np.std(prices[-20:]) / current_price
            vol_target = current_price * (1 + 2 * avg_vol * tf_multiplier)
        else:
            vol_target = current_price * (1.1 * tf_multiplier)
        
        # 4. Wave Structure Analysis
        if timeframe in self.analysis_results and 'multifractal' in self.analysis_results[timeframe]:
            mf = self.analysis_results[timeframe]['multifractal']
            multifractality = mf['multifractality']
            wave_amplitude = np.std(prices[-lookback:]) if lookback > 0 else current_price * 0.05
            wave_target = current_price + (wave_amplitude * (1 + multifractality) * tf_multiplier)
        else:
            wave_target = current_price * (1.08 * tf_multiplier)
        
        # 5. Machine Learning Forecast
        try:
            from sklearn.linear_model import LinearRegression
            
            features = []
            min_samples = min(len(prices), 20)
            for i in range(min_samples, len(prices)):
                features.append([
                    float(prices[i-1]),
                    float(np.mean(prices[max(0, i-5):i])),
                    float(np.mean(prices[max(0, i-10):i])),
                    float(np.std(prices[max(0, i-10):i]))
                ])
            
            targets_ml = prices[min_samples:]
            
            if len(features) > 50:
                model = LinearRegression()
                model.fit(features[:-1], targets_ml[1:])
                
                last_features = [[
                    float(prices[-1]),
                    float(np.mean(prices[-5:])),
                    float(np.mean(prices[-10:])),
                    float(np.std(prices[-10:]))
                ]]
                
                ml_forecast = float(model.predict(last_features)[0])
            else:
                ml_forecast = current_price * (1.06 * tf_multiplier)
        except:
            ml_forecast = current_price * (1.06 * tf_multiplier)
        
        # 6. Quantum State Projection
        if timeframe in self.analysis_results and 'hilbert' in self.analysis_results[timeframe]:
            phase_velocity = np.mean(self.analysis_results[timeframe]['hilbert']['phase_velocity'][-10:])
            quantum_projection = current_price * (1 + phase_velocity * 0.05 * tf_multiplier)
        else:
            quantum_projection = current_price * (1.07 * tf_multiplier)
        
        # Calculate consensus target
        weights = {
            'fibonacci': 0.20,
            'harmonic': 0.15,
            'volatility': 0.20,
            'wave': 0.15,
            'ml': 0.20,
            'quantum': 0.10
        }
        
        consensus_target = (
            fib_target * weights['fibonacci'] +
            harmonic_projection * weights['harmonic'] +
            vol_target * weights['volatility'] +
            wave_target * weights['wave'] +
            ml_forecast * weights['ml'] +
            quantum_projection * weights['quantum']
        )
        
        # Calculate confidence
        all_targets = [fib_target, harmonic_projection, vol_target, 
                      wave_target, ml_forecast, quantum_projection]
        target_std = np.std(all_targets)
        target_mean = np.mean(all_targets)
        confidence = max(0, min(100, 100 - (target_std / target_mean * 100))) if target_mean > 0 else 0
        
        # Estimate timeframe in periods
        timeframe_periods = {
            '15m': 96, '30m': 48, '1h': 24, '4h': 6,
            '6h': 4, '12h': 2, '1d': 1
        }.get(timeframe, 24)
        
        return {
            'fibonacci_target': float(fib_target),
            'harmonic_target': float(harmonic_projection),
            'volatility_target': float(vol_target),
            'wave_target': float(wave_target),
            'ml_forecast': float(ml_forecast),
            'quantum_projection': float(quantum_projection),
            'consensus_target': float(consensus_target),
            'confidence': float(confidence),
            'timeframe': timeframe_periods,
            'current_price': float(current_price)
        }
    
    def _default_targets(self) -> Dict[str, float]:
        """Return default targets if data is insufficient"""
        return {
            'fibonacci_target': 100000.0,
            'harmonic_target': 100000.0,
            'volatility_target': 100000.0,
            'wave_target': 100000.0,
            'ml_forecast': 100000.0,
            'quantum_projection': 100000.0,
            'consensus_target': 100000.0,
            'confidence': 0.0,
            'timeframe': 0,
            'current_price': 0.0
        }
    
    def calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        # Use 1h as primary timeframe
        if '1h' not in self.kline_data:
            return {
                'var_95': 0.0,
                'cvar_95': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'info_ratio': 0.0
            }
        
        df = self.kline_data['1h']
        if df.empty or len(df) < 2:
            return {
                'var_95': 0.0,
                'cvar_95': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'info_ratio': 0.0
            }
        
        prices = df['close'].values
        returns = np.diff(np.log(prices + 1e-10))
        
        # Value at Risk
        var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
        
        # Conditional VaR
        cvar_95 = np.mean(returns[returns <= var_95]) if len(returns[returns <= var_95]) > 0 else var_95
        
        # Maximum Drawdown
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / (running_max + 1e-10)
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
        
        # Sharpe Ratio
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252 * 24) if np.std(returns) > 0 else 0
        
        # Information Ratio
        try:
            ma_prices = df['close'].rolling(24).mean().dropna()
            if len(ma_prices) > 1:
                benchmark_returns = np.diff(np.log(ma_prices.values + 1e-10))
                if len(benchmark_returns) > 0 and len(returns) >= len(benchmark_returns):
                    active_returns = returns[-len(benchmark_returns):] - benchmark_returns
                    info_ratio = np.mean(active_returns) / (np.std(active_returns) + 1e-10) * np.sqrt(252 * 24)
                else:
                    info_ratio = 0.0
            else:
                info_ratio = 0.0
        except:
            info_ratio = 0.0
        
        return {
            'var_95': float(var_95),
            'cvar_95': float(cvar_95),
            'max_drawdown': float(max_drawdown),
            'sharpe_ratio': float(sharpe_ratio),
            'info_ratio': float(info_ratio)
        }
    
    def generate_mathematical_conclusion(self, all_targets: Dict[str, Dict[str, float]], 
                                       risk_metrics: Dict[str, float]) -> str:
        """Generate comprehensive mathematical conclusion"""
        if not all_targets:
            return "Insufficient data for a mathematical conclusion."

        conclusion = []

        # 1. Timeframe Alignment
        alignment_score = self._calculate_timeframe_alignment()
        conclusion.append("Timeframe Alignment & Divergence:\n")
        conclusion.append(f"  - Confluence Score: {alignment_score:.1f}%")
        if alignment_score > 70:
            conclusion.append("  - Verdict: STRONG CONFLUENCE. High conviction signal across timeframes.")
        elif alignment_score > 40:
            conclusion.append("  - Verdict: MODERATE ALIGNMENT. General agreement with minor noise.")
        else:
            conclusion.append("  - Verdict: DIVERGENCE. Significant disagreement warrants caution.")
        
        # Identify divergences
        short_term_dir = self._get_timeframe_group_direction(['15m', '30m', '1h'], all_targets)
        long_term_dir = self._get_timeframe_group_direction(['4h', '6h', '12h', '1d'], all_targets)
        if short_term_dir != 'NEUTRAL' and long_term_dir != 'NEUTRAL' and short_term_dir != long_term_dir:
            conclusion.append(f"  - Key Divergence: Short-term ({short_term_dir}) vs long-term ({long_term_dir}).")

        # 2. Overall Directional Bias
        directional_bias = self._calculate_directional_bias(all_targets.get('1h', {}))
        conclusion.append("\nOverall Directional Bias:\n")
        conclusion.append(f"  - Direction: {directional_bias['direction']}")
        conclusion.append(f"  - Conviction Strength: {directional_bias['strength']:.2f}/10.0")

        # 3. Synthesized Price Projection
        conclusion.append("\nSynthesized Price Projection:\n")
        valid_targets = [t['consensus_target'] for t in all_targets.values() if t['confidence'] > 10]
        valid_weights = [t['confidence'] for t in all_targets.values() if t['confidence'] > 10]
        if valid_targets:
            weighted_target = np.average(valid_targets, weights=valid_weights)
            avg_confidence = np.mean(valid_weights)
            primary_price = all_targets.get('1h', {}).get('current_price', weighted_target)
            move_pct = ((weighted_target - primary_price) / primary_price * 100) if primary_price > 0 else 0

            conclusion.append(f"  - Confidence-Weighted Target: {weighted_target:.4f}")
            conclusion.append(f"  - Implied Move: {move_pct:+.2f}%")
            conclusion.append(f"  - Average Model Confidence: {avg_confidence:.2f}%")
        else:
            conclusion.append("  - Projections lack sufficient confidence.")

        # 4. Market State & Recommendation
        market_state = self._assess_market_state()
        conclusion.append("\nDominant Market State & Recommendation:\n")
        conclusion.append(f"  - Assessed State: {market_state['state']} (Confidence: {market_state['confidence']:.1f}%)")
        recommendation = self._generate_recommendation(market_state, directional_bias, all_targets.get('1h', {}), risk_metrics)
        conclusion.append(f"  - Mathematical Strategy: {recommendation}")

        return "\n".join(conclusion)
    
    def _get_timeframe_group_direction(self, tfs: List[str], all_targets: Dict) -> str:
        """Get consensus direction for a group of timeframes"""
        directions = []
        for tf in tfs:
            if tf in all_targets:
                targets = all_targets[tf]
                if targets['confidence'] > 10:
                    direction = 1 if targets['consensus_target'] > targets['current_price'] else -1
                    directions.append(direction)
        
        if not directions:
            return "NEUTRAL"
        
        avg_direction = np.mean(directions)
        if avg_direction > 0.33:
            return "BULLISH"
        elif avg_direction < -0.33:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def _assess_market_state(self) -> Dict[str, Any]:
        """Assess current market state"""
        state_indicators = {
            'trending': 0,
            'ranging': 0,
            'volatile': 0
        }
        
        # Check across all timeframes
        for timeframe in self.timeframes:
            if timeframe not in self.analysis_results:
                continue
            
            results = self.analysis_results[timeframe]
            
            # Check Hurst exponent
            if 'dfa' in results:
                hurst = results['dfa']['hurst']
                if hurst > 0.6:
                    state_indicators['trending'] += 2
                elif hurst < 0.4:
                    state_indicators['volatile'] += 2
                else:
                    state_indicators['ranging'] += 2
            
            # Check multifractal spectrum
            if 'multifractal' in results:
                mf_width = results['multifractal']['multifractality']
                if mf_width > 0.5:
                    state_indicators['volatile'] += 1
            
            # Check RQA determinism
            if 'rqa' in results:
                det = results['rqa']['determinism']
                if det > 0.7:
                    state_indicators['trending'] += 1
                elif det < 0.3:
                    state_indicators['volatile'] += 1
            
            # Check volatility persistence
            if 'egarch' in results:
                persistence = results['egarch'].get('leverage_effect', 0)
                if abs(persistence) > 0.1:
                    state_indicators['volatile'] += 1
        
        # Check order flow
        if 'order_flow_analysis' in self.analysis_results and '1h' in self.analysis_results['order_flow_analysis']:
            imbalance = abs(self.analysis_results['order_flow_analysis']['1h']['mean_imbalance'])
            if imbalance > 0.1:
                state_indicators['trending'] += 1
        
        # Determine dominant state
        total = sum(state_indicators.values())
        if total == 0:
            return {'state': 'NEUTRAL', 'confidence': 0, 'indicators': state_indicators}
            
        max_state = max(state_indicators, key=state_indicators.get)
        confidence = (state_indicators[max_state] / total) * 100
        
        # State Transition Logic Check
        if 'hmm' in self.analysis_results.get('15m', {}):
             hmm15m = self.analysis_results['15m']['hmm']
             if hmm15m.get('converged') and len(hmm15m.get('states', [])) > 1:
                 last_state = hmm15m['states'][-2]
                 current_state = hmm15m['states'][-1]
                 if abs(last_state - current_state) > 1:
                     print(f"WARNING: Unrealistic state jump detected on 15m")
                     max_state = 'ranging'
                     confidence = 50.0

        return {
            'state': max_state.upper(),
            'confidence': confidence,
            'indicators': state_indicators
        }
    
    def _calculate_directional_bias(self, targets: Dict[str, float]) -> Dict[str, Any]:
        """Calculate market directional bias"""
        current_price = targets.get('current_price', 0)
        consensus_target = targets.get('consensus_target', 0)
        
        if current_price == 0:
            return {'direction': 'NEUTRAL', 'strength': 0}
        
        bias_score = 0
        
        # Price target bias
        if consensus_target > current_price * 1.02:
            bias_score += 3
            direction = 'BULLISH'
        elif consensus_target < current_price * 0.98:
            bias_score -= 3
            direction = 'BEARISH'
        else:
            direction = 'NEUTRAL'
        
        # Technical bias from all timeframes
        for timeframe in self.timeframes:
            if timeframe in self.analysis_results and 'ha_ms_signal' in self.analysis_results[timeframe]:
                ha_results = self.analysis_results[timeframe]['ha_ms_signal']
                if not ha_results.empty and 'sum_signal' in ha_results:
                    composite_signal = ha_results['sum_signal'].iloc[-1]
                    if composite_signal > 40:
                        bias_score += 1
                    elif composite_signal < -40:
                        bias_score -= 1
        
        # Market structure bias
        if 'positioning_analysis' in self.analysis_results:
            for tf_data in self.analysis_results['positioning_analysis'].values():
                sentiment = tf_data.get('market_sentiment', 'neutral')
                if sentiment == 'bullish':
                    bias_score += 1
                elif sentiment == 'bearish':
                    bias_score -= 1
        
        # CVD bias
        cvd_bias = self._get_cvd_bias()
        bias_score += cvd_bias
        
        # Normalize strength
        strength = min(10, abs(bias_score))
        
        if bias_score < 0:
            direction = 'BEARISH'
        elif bias_score > 0:
            direction = 'BULLISH'
        
        return {
            'direction': direction,
            'strength': float(strength),
            'raw_score': bias_score
        }
    
    def _get_cvd_bias(self) -> int:
        """Get CVD-based directional bias"""
        bias = 0
        
        for timeframe in ['1h', '4h']:
            if timeframe in self.collected_data and 'cvd' in self.collected_data[timeframe]:
                cvd_data = pd.DataFrame(self.collected_data[timeframe]['cvd'])
                
                if not cvd_data.empty and 'cvd_close' in cvd_data.columns:
                    aggregate_cvd = cvd_data[cvd_data['exchange'] == 'AGGREGATE'] if 'exchange' in cvd_data.columns else cvd_data
                    
                    if not aggregate_cvd.empty:
                        cvd_values = aggregate_cvd['cvd_close'].values
                        if len(cvd_values) > 5:
                            recent_cvd = cvd_values[-5:].mean()
                            if recent_cvd > 0:
                                bias += 1
                            elif recent_cvd < 0:
                                bias -= 1
        return bias

    def _generate_recommendation(self, market_state: Dict, directional_bias: Dict,
                                targets: Dict, risk_metrics: Dict) -> str:
        """Generate mathematical recommendation"""
        # Base recommendation on market state and bias
        if market_state['state'] == 'TRENDING':
            if directional_bias['direction'] == 'BULLISH':
                return "Follow strong upward momentum. Models suggest trend continuation. High conviction."
            elif directional_bias['direction'] == 'BEARISH':
                return "Follow strong downward momentum. Models suggest trend continuation. High conviction."
            else:
                return "Trending market showing indecision. Await clearer directional bias."
        
        elif market_state['state'] == 'RANGING':
            return "Market in consolidation. Mean-reversion strategies mathematically optimal. Fade extremes."
        
        elif market_state['state'] == 'VOLATILE':
            if risk_metrics['var_95'] < -0.05:
                return "High volatility with significant downside risk. Reduce exposure, prioritize capital preservation."
            return "High volatility regime. Trade breakouts or wait for subsidence. Risk management critical."
        
        return "Market conditions mixed. No high-conviction mathematical strategy apparent."
    
    def _calculate_timeframe_alignment(self) -> float:
        """Calculate alignment score across timeframes"""
        alignment_scores = []
        
        # Check directional alignment
        directions = []
        for timeframe in self.timeframes:
            if timeframe in self.kline_data:
                df = self.kline_data[timeframe]
                if len(df) > 20:
                    short_ma = df['close'].rolling(10).mean().iloc[-1]
                    long_ma = df['close'].rolling(20).mean().iloc[-1]
                    directions.append(1 if short_ma > long_ma else -1)
        
        if directions:
            alignment = abs(sum(directions)) / len(directions) * 100
            alignment_scores.append(alignment)
        
        # Check momentum alignment
        momentums = []
        for timeframe in self.timeframes:
            if timeframe in self.analysis_results and 'ha_ms_signal' in self.analysis_results[timeframe]:
                ha_results = self.analysis_results[timeframe]['ha_ms_signal']
                if not ha_results.empty and 'sum_signal' in ha_results:
                    signal = ha_results['sum_signal'].iloc[-1]
                    momentums.append(1 if signal > 0 else -1)
        
        if momentums:
            momentum_alignment = abs(sum(momentums)) / len(momentums) * 100
            alignment_scores.append(momentum_alignment)
        
        # Check regime alignment
        regimes = []
        for timeframe in self.timeframes:
            if timeframe in self.analysis_results and 'hmm' in self.analysis_results[timeframe]:
                hmm = self.analysis_results[timeframe]['hmm']
                if 'states' in hmm and len(hmm['states']) > 0:
                    current_state = hmm['states'][-1]
                    if len(hmm.get('state_labels', [])) == 3:
                        regime_map = {0: -1, 1: 0, 2: 1}
                        regimes.append(regime_map.get(current_state, 0))
        
        if regimes:
            regime_alignment = abs(sum(regimes)) / len(regimes) * 100
            alignment_scores.append(regime_alignment)
        
        return float(np.mean(alignment_scores)) if alignment_scores else 50.0

    def save_results(self):
        """Save all analysis results to JSON file"""
        print("\n[SAVING ANALYSIS STATE]")
        
        # Create serializable representation
        output_data = {
            'metadata': {
                'version': self.version,
                'analyst': self.analyst,
                'timestamp_utc': self.current_time.isoformat(),
                'timeframes': self.timeframes,
            },
            'stats': self.stats,
            'kline_data': {},
            'collected_data': self.collected_data,
            'analysis_results': {},
            'performance_metrics': self.performance_metrics
        }
        
        # Convert kline data
        for tf, df in self.kline_data.items():
            output_data['kline_data'][tf] = df.to_dict('list')
        
        # Custom serialization for analysis results
        for timeframe, results in self.analysis_results.items():
            output_data['analysis_results'][timeframe] = {}
            for key, value in results.items():
                if isinstance(value, pd.DataFrame):
                    output_data['analysis_results'][timeframe][key] = value.to_dict('list')
                elif isinstance(value, pd.Series):
                    output_data['analysis_results'][timeframe][key] = value.tolist()
                elif isinstance(value, np.ndarray):
                    output_data['analysis_results'][timeframe][key] = value.tolist()
                elif isinstance(value, (np.int64, np.float64)):
                     output_data['analysis_results'][timeframe][key] = value.item()
                elif isinstance(value, dict):
                    # Recursively clean dicts
                    clean_dict = {}
                    for k, v in value.items():
                        if isinstance(v, np.ndarray): 
                            clean_dict[k] = v.tolist()
                        elif isinstance(v, pd.DataFrame): 
                            clean_dict[k] = v.to_dict('list')
                        elif isinstance(v, pd.Series):
                            clean_dict[k] = v.tolist()
                        elif isinstance(v, (np.int64, np.float64)): 
                            clean_dict[k] = v.item()
                        elif isinstance(v, nx.Graph): 
                            clean_dict[k] = "Graph data not serialized"
                        else: 
                            clean_dict[k] = v
                    output_data['analysis_results'][timeframe][key] = clean_dict
                else:
                    output_data['analysis_results'][timeframe][key] = value
        
        # Handle special cases in analysis_results
        for key in ['tensor_fusion', 'information_flow', 'tda_analysis', 'order_flow_analysis', 
                    'positioning_analysis', 'liquidation_analysis']:
            if key in self.analysis_results:
                output_data['analysis_results'][key] = self._serialize_analysis_result(self.analysis_results[key])
        
        try:
            with open(self.results_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, allow_nan=True, default=str)
            print(f"  ✓ Analysis state saved to: {self.results_file}")
        except Exception as e:
            self.log_error("SaveResults", {}, str(e), "Failed to save JSON results", exc_info=True)
    
    def _serialize_analysis_result(self, result: Any) -> Any:
        """Helper to serialize complex analysis results"""
        if isinstance(result, dict):
            serialized = {}
            for k, v in result.items():
                if isinstance(v, np.ndarray):
                    serialized[k] = v.tolist()
                elif isinstance(v, pd.DataFrame):
                    serialized[k] = v.to_dict('list')
                elif isinstance(v, pd.Series):
                    serialized[k] = v.tolist()
                elif isinstance(v, (np.int64, np.float64)):
                    serialized[k] = v.item()
                elif isinstance(v, nx.Graph):
                    serialized[k] = f"Graph with {v.number_of_nodes()} nodes and {v.number_of_edges()} edges"
                elif isinstance(v, dict):
                    serialized[k] = self._serialize_analysis_result(v)
                else:
                    serialized[k] = v
            return serialized
        elif isinstance(result, list):
            return [self._serialize_analysis_result(item) for item in result]
        else:
            return result
    
    def execute_analysis(self):
        """Execute complete optimized analysis workflow"""
        print(f"\n{'='*80}")
        print(f"INTEGRATED MATHEMATICAL ANALYSIS SYSTEM {self.version.upper()}")
        print(f"Analyst: {self.analyst}")
        print(f"Started: {self.current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"Timeframes: {', '.join(self.timeframes)}")
        print(f"{'='*80}\n")
        
        self.start_time = time.time()
        
        try:
            # Phase 1: Concurrent Kline Data Collection
            self.fetch_klines_concurrent()
            
            # Phase 2: Market Data Collection
            self.collect_all_market_data_by_endpoint()
            
            # Phase 3: Optimized Mathematical Analysis
            print("\n[PHASE 3] PERFORMING OPTIMIZED MATHEMATICAL ANALYSIS")
            self.perform_comprehensive_analysis_optimized()
            
            # Phase 4: Generate Reports and Save State
            print("\n[PHASE 4] GENERATING REPORTS & SAVING STATE")
            self.save_results()
            self.generate_comprehensive_summary_enhanced()
            
            duration = time.time() - self.start_time
            
            # Print performance metrics
            print(f"\n{'='*80}")
            print(f"ANALYSIS COMPLETE")
            print(f"{'='*80}")
            print(f"Total Time: {duration:.2f} seconds")
            print(f"Results File: {self.results_file}")
            print(f"Summary File: {self.summary_file}")
            print(f"Error Log: {self.error_log_file}")
            print(f"Data Points Collected: {self.stats['data_points_collected']:,}")
            print(f"Cache Hits: {self.stats['cache_hits']}, Cache Misses: {self.stats['cache_misses']}")
            
            # Cache statistics
            cache_stats = self.cache.get_cache_stats()
            print(f"\nCache Statistics:")
            print(f"  - Total Entries: {cache_stats['total_entries']}")
            print(f"  - Total Size: {cache_stats['total_size_mb']:.2f} MB")
            
            print(f"{'='*80}\n")
            
        except Exception as e:
            self.log_error("FATAL", {}, str(e), "System Crash", exc_info=True)
            raise
        finally:
            # Clean up executors
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=True)
            # ProcessPoolExecutor cleanup removed


if __name__ == "__main__":
    try:
        # These values can be passed as arguments in production
        user_login = "ASJFOJ1"
        current_utc_time_str = "2025-07-17 09:15:46"
        
        analyzer = IntegratedMathematicalAnalysisSystem()
        
        # Override defaults with provided values
        analyzer.analyst = user_login
        analyzer.current_time = datetime.strptime(current_utc_time_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
        analyzer.current_timestamp = int(analyzer.current_time.timestamp() * 1000)
        
        print(f"Executing analysis for user '{analyzer.analyst}' at time '{analyzer.current_time.isoformat()}'")
        
        analyzer.execute_analysis()
    except Exception as e:
        print(f"[CRITICAL] Unexpected error during initialization or execution: {e}")
        traceback.print_exc()