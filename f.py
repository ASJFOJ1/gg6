--- smod11.py	2025-01-01 00:00:00.000000000 +0000
+++ smod11_updated.py	2025-01-01 00:00:00.000000000 +0000
@@ -808,81 +808,115 @@
 
 
 class MarkovSwitchingGARCH:
-    """
-    Markov-Switching GARCH model combining HMM and GARCH
-    """
+    """DEPRECATED - Use RegimeSwitchingStochasticVolatility instead"""
+    pass
 
-    def __init__(self, n_states: int = 3, garch_p: int = 1, garch_q: int = 1):
+
+class RegimeSwitchingStochasticVolatility:
+    """
+    Regime-Switching Stochastic Volatility model
+    Replaces MS-GARCH with more robust volatility modeling
+    """
+
+    def __init__(self, n_states: int = 3):
         self.n_states = n_states
-        self.garch_p = garch_p
-        self.garch_q = garch_q
         self.hmm_model = None
-        self.garch_models = {}
         self.state_params = {}
         self.scaler = StandardScaler()
         self.logger = logging.getLogger(__name__)
 
     def fit(self, returns: np.ndarray):
         """
-        Fit MS-GARCH model with improved convergence robustness.
+        Fit Regime-Switching Stochastic Volatility model
         """
-        # Step 1: Fit HMM to identify regimes
+        # Use log-squared returns as volatility proxy
+        log_squared_returns = np.log(returns**2 + 1e-10)
+        
+        # Fit HMM to log-squared returns
         self.hmm_model = hmm.GaussianHMM(
             n_components=self.n_states,
             covariance_type="diag",
-            n_iter=200,  # Increased iterations
+            n_iter=200,
             random_state=42,
-            tol=1e-3,  # Adjusted tolerance
-            init_params="cm",  # Initialize with kmeans and means
-            params="cmt"  # Re-estimate covars, means, transmat
+            tol=1e-3,
+            init_params="cm",
+            params="cmt"
         )
 
-        # Scale returns for HMM stability
-        X = self.scaler.fit_transform(returns.reshape(-1, 1))
+        # Scale log-squared returns for HMM stability
+        X = self.scaler.fit_transform(log_squared_returns.reshape(-1, 1))
 
         try:
             self.hmm_model.fit(X)
             states = self.hmm_model.predict(X)
             self.logger.info(f"HMM converged successfully with {self.n_states} states")
         except Exception as e:
-            # Fallback if HMM fails to converge
             self.logger.warning(f"HMM fitting failed: {e}. Using fallback k-means clustering.")
-            # Use k-means clustering as fallback
             from sklearn.cluster import KMeans
             kmeans = KMeans(n_clusters=self.n_states, random_state=42, n_init=10)
             states = kmeans.fit_predict(X)
 
-        # Step 2: Fit separate GARCH model for each state
+        # Calculate state-specific volatility parameters
         for state in range(self.n_states):
             state_returns = returns[states == state]
-
-            if len(state_returns) > 50:  # Need enough data
-                # Fit GARCH model
-                model = arch_model(
-                    state_returns * 100,  # Scale for numerical stability
-                    vol='GARCH',
-                    p=self.garch_p,
-                    q=self.garch_q,
-                    rescale=False
-                )
-
-                try:
-                    res = model.fit(disp='off', show_warning=False)
-                    self.garch_models[state] = res
-
-                    # Store parameters
-                    self.state_params[state] = {
-                        'omega': res.params['omega'],
-                        'alpha': res.params.get('alpha[1]', 0),
-                        'beta': res.params.get('beta[1]', 0),
-                        'mean': np.mean(state_returns),
-                        'unconditional_vol': np.std(state_returns)
-                    }
-                    self.logger.info(f"GARCH fitted for state {state}")
-                except Exception as e:
-                    # Fallback to simple volatility
-                    self.logger.warning(f"GARCH fitting failed for state {state}: {e}")
-                    self.state_params[state] = {
-                        'omega': np.var(state_returns),
-                        'alpha': 0.1,
-                        'beta': 0.8,
+            state_log_sq_returns = log_squared_returns[states == state]
+
+            if len(state_returns) > 10:
+                # Calculate stochastic volatility parameters
+                self.state_params[state] = {
+                    'mean_log_vol': np.mean(state_log_sq_returns),
+                    'vol_of_vol': np.std(state_log_sq_returns),
+                    'persistence': self._calculate_persistence(state_log_sq_returns),
+                    'mean_return': np.mean(state_returns),
+                    'unconditional_vol': np.std(state_returns)
+                }
+                self.logger.info(f"SV parameters fitted for state {state}")
+            else:
+                # Fallback parameters
+                self.state_params[state] = {
+                    'mean_log_vol': np.log(np.var(returns)),
+                    'vol_of_vol': 0.2,
+                    'persistence': 0.95,
+                    'mean_return': np.mean(returns),
+                    'unconditional_vol': np.std(returns)
+                }
+
+        return self
+
+    def _calculate_persistence(self, log_sq_returns: np.ndarray) -> float:
+        """Calculate persistence parameter using autocorrelation"""
+        if len(log_sq_returns) < 10:
+            return 0.95
+        
+        # Calculate first-order autocorrelation
+        from statsmodels.tsa.stattools import acf
+        try:
+            autocorr = acf(log_sq_returns, nlags=1, fft=True)[1]
+            # Ensure persistence is in reasonable range
+            return np.clip(autocorr, 0.8, 0.99)
+        except:
+            return 0.95
+
+    def predict_volatility(self, returns: np.ndarray, horizon: int = 1, n_simulations: int = 1000) -> Dict[str, Any]:
+        """
+        Predict volatility using Monte Carlo simulation
+        """
+        # Get current state probabilities
+        log_squared_returns = np.log(returns**2 + 1e-10)
+        X = self.scaler.transform(log_squared_returns.reshape(-1, 1))
+        
+        try:
+            state_probs = self.hmm_model.predict_proba(X)
+            current_state_probs = state_probs[-1]
+            trans_mat = self.hmm_model.transmat_
+        except:
+            # Fallback
+            current_state_probs = np.ones(self.n_states) / self.n_states
+            trans_mat = np.ones((self.n_states, self.n_states)) / self.n_states
+
+        # Monte Carlo simulation for volatility paths
+        vol_paths = np.zeros((n_simulations, horizon))
+        
+        for sim in range(n_simulations):
+            # Sample initial state
+            current_state = np.random.choice(self.n_states, p=current_state_probs)
+            
+            for h in range(horizon):
+                params = self.state_params[current_state]
+                
+                # Simulate log-volatility using AR(1) process
+                if h == 0:
+                    log_vol = params['mean_log_vol']
+                else:
+                    # AR(1) with stochastic innovation
+                    innovation = np.random.normal(0, params['vol_of_vol'])
+                    log_vol = (params['persistence'] * log_vol + 
+                              (1 - params['persistence']) * params['mean_log_vol'] + 
+                              innovation)
+                
+                vol_paths[sim, h] = np.exp(log_vol / 2)
+                
+                # Transition to next state
+                if h < horizon - 1:
+                    current_state = np.random.choice(self.n_states, p=trans_mat[current_state])
+
+        # Calculate percentiles
+        vol_percentiles = np.percentile(vol_paths, [5, 25, 50, 75, 95], axis=0)
+        
+        # Calculate risk metrics
+        final_vols = vol_paths[:, -1]
+        VaR_95 = np.percentile(final_vols, 95)
+        CVaR_95 = np.mean(final_vols[final_vols >= VaR_95])
+
+        # Determine current regime
+        current_regime = np.argmax(current_state_probs)
+        regime_labels = ['Low Vol', 'Medium Vol', 'High Vol']
+        
+        # Sort states by mean volatility
+        state_vols = [self.state_params[s]['unconditional_vol'] for s in range(self.n_states)]
+        sorted_idx = np.argsort(state_vols)
+        current_regime_label = regime_labels[np.where(sorted_idx == current_regime)[0][0]]
+
+        return {
+            'volatility_forecast': vol_percentiles[2],  # Median forecast
+            'volatility_percentiles': {
+                '5%': vol_percentiles[0],
+                '25%': vol_percentiles[1],
+                '50%': vol_percentiles[2],
+                '75%': vol_percentiles[3],
+                '95%': vol_percentiles[4]
+            },
+            'state_volatilities': np.array([vol_percentiles[2] for _ in range(self.n_states)]),  # For compatibility
+            'current_regime': current_regime_label,
+            'regime_probabilities': dict(enumerate(current_state_probs)),
+            'state_parameters': self.state_params,
+            'transition_matrix': trans_mat,
+            'VaR_95': VaR_95,
+            'CVaR_95': CVaR_95,
+            'mean_forecast': np.mean(vol_paths, axis=0)
+        }
+
+
+class AdaptiveParticleFilter:
+    """
+    Adaptive Particle Filter for non-linear state estimation
+    Better than UKF for handling multi-modal distributions
+    """
+
+    def __init__(self, dim_x: int, dim_z: int, n_particles: int = 2000):
+        self.dim_x = dim_x
+        self.dim_z = dim_z
+        self.n_particles = n_particles
+        self.logger = logging.getLogger(__name__)
+        
+        # Initialize particles
+        self.particles = np.zeros((n_particles, dim_x))
+        self.weights = np.ones(n_particles) / n_particles
+        
+        # Process and measurement noise
+        self.Q = np.eye(dim_x) * 0.01
+        self.R = np.eye(dim_z) * 0.1
+        
+        # Resampling threshold
+        self.resample_threshold = n_particles / 2
+        
+        # Track regime changes
+        self.regime_history = []
+        self.current_regime = 0
+
+    def initialize(self, initial_state: np.ndarray, initial_cov: np.ndarray):
+        """Initialize particle distribution"""
+        for i in range(self.n_particles):
+            self.particles[i] = np.random.multivariate_normal(initial_state, initial_cov)
+        self.weights = np.ones(self.n_particles) / self.n_particles
+
+    def state_transition(self, particle: np.ndarray, dt: float = 1.0) -> np.ndarray:
+        """Non-linear state transition with regime-dependent dynamics"""
+        x_new = particle.copy()
+        
+        # Price evolution with stochastic volatility
+        if self.dim_x >= 3:  # [price, velocity, volatility]
+            # Update volatility using mean-reverting process
+            vol_mean = 0.02
+            vol_speed = 0.1
+            vol_vol = 0.1
+            x_new[2] = x_new[2] + vol_speed * (vol_mean - x_new[2]) * dt + vol_vol * np.sqrt(dt) * np.random.normal()
+            x_new[2] = max(0.001, x_new[2])  # Ensure positive volatility
+            
+            # Update velocity with volatility-dependent noise
+            x_new[1] = x_new[1] * 0.98 + np.sqrt(x_new[2]) * np.random.normal() * np.sqrt(dt)
+            
+            # Update price
+            x_new[0] = x_new[0] + x_new[1] * dt
+        else:
+            # Simple model for 2D state
+            x_new[0] = particle[0] + particle[1] * dt
+            x_new[1] = particle[1] * 0.98 + np.random.normal() * 0.01
+        
+        return x_new
+
+    def measurement_function(self, particle: np.ndarray) -> np.ndarray:
+        """Measurement function"""
+        z = np.zeros(self.dim_z)
+        z[0] = particle[0]  # Observe price
+        return z
+
+    def predict(self, dt: float = 1.0):
+        """Prediction step"""
+        for i in range(self.n_particles):
+            # Add process noise
+            noise = np.random.multivariate_normal(np.zeros(self.dim_x), self.Q * dt)
+            self.particles[i] = self.state_transition(self.particles[i], dt) + noise
+
+    def update(self, z: np.ndarray):
+        """Update step with observation"""
+        # Calculate likelihood for each particle
+        for i in range(self.n_particles):
+            z_pred = self.measurement_function(self.particles[i])
+            innovation = z - z_pred
+            
+            # Likelihood using Gaussian measurement model
+            likelihood = np.exp(-0.5 * innovation.T @ np.linalg.inv(self.R) @ innovation)
+            self.weights[i] *= likelihood
+        
+        # Normalize weights
+        self.weights += 1e-300  # Avoid numerical issues
+        self.weights /= np.sum(self.weights)
+        
+        # Check for regime change
+        self._detect_regime_change()
+        
+        # Resample if effective sample size is too low
+        n_eff = 1.0 / np.sum(self.weights**2)
+        if n_eff < self.resample_threshold:
+            self._systematic_resample()
+
+    def _systematic_resample(self):
+        """Systematic resampling to avoid particle degeneracy"""
+        positions = (np.arange(self.n_particles) + np.random.uniform()) / self.n_particles
+        cumulative_sum = np.cumsum(self.weights)
+        
+        new_particles = np.zeros_like(self.particles)
+        i, j = 0, 0
+        
+        while i < self.n_particles:
+            if positions[i] < cumulative_sum[j]:
+                new_particles[i] = self.particles[j].copy()
+                i += 1
+            else:
+                j += 1
+        
+        self.particles = new_particles
+        self.weights = np.ones(self.n_particles) / self.n_particles
+
+    def _detect_regime_change(self):
+        """Detect regime changes based on particle distribution"""
+        if self.dim_x >= 3:  # If we have volatility dimension
+            # Cluster particles by volatility
+            volatilities = self.particles[:, 2]
+            vol_mean = np.average(volatilities, weights=self.weights)
+            
+            # Simple regime detection based on volatility level
+            if vol_mean < 0.01:
+                new_regime = 0  # Low volatility
+            elif vol_mean < 0.03:
+                new_regime = 1  # Medium volatility
+            else:
+                new_regime = 2  # High volatility
+            
+            if new_regime != self.current_regime:
+                self.regime_history.append((len(self.regime_history), new_regime))
+                self.current_regime = new_regime
+
+    def get_state_estimate(self) -> Tuple[np.ndarray, np.ndarray]:
+        """Get weighted mean and covariance of particles"""
+        mean = np.average(self.particles, weights=self.weights, axis=0)
+        
+        # Calculate weighted covariance
+        diff = self.particles - mean
+        cov = np.zeros((self.dim_x, self.dim_x))
+        for i in range(self.n_particles):
+            cov += self.weights[i] * np.outer(diff[i], diff[i])
+        
+        return mean, cov
+
+    def filter_series(self, observations: np.ndarray) -> Dict[str, np.ndarray]:
+        """Filter a complete time series"""
+        n = len(observations)
+        filtered_states = np.zeros((n, self.dim_x))
+        innovations = np.zeros(n)
+        volatility_filtered = np.zeros(n) if self.dim_x >= 3 else None
+        
+        # Initialize with first observation
+        initial_state = np.array([observations[0], 0.0])
+        if self.dim_x >= 3:
+            initial_state = np.append(initial_state, 0.02)  # Initial volatility
+        
+        self.initialize(initial_state, np.eye(self.dim_x) * 0.1)
+        
+        for i in range(n):
+            self.predict()
+            
+            # Get prediction before update
+            pred_mean, _ = self.get_state_estimate()
+            z_pred = self.measurement_function(pred_mean)
+            innovations[i] = observations[i] - z_pred[0]
+            
+            # Update with observation
+            self.update(np.array([observations[i]]))
+            
+            # Get filtered estimate
+            filtered_mean, _ = self.get_state_estimate()
+            filtered_states[i] = filtered_mean
+            
+            if self.dim_x >= 3:
+                volatility_filtered[i] = filtered_mean[2]
+        
+        # Detect regime changes
+        regime_changes = []
+        if len(self.regime_history) > 1:
+            for i in range(1, len(self.regime_history)):
+                regime_changes.append({
+                    'time': self.regime_history[i][0],
+                    'from_regime': self.regime_history[i-1][1],
+                    'to_regime': self.regime_history[i][1]
+                })
+        
+        result = {
+            'filtered_states': filtered_states,
+            'innovations': innovations,
+            'final_covariance': self.get_state_estimate()[1]
+        }
+        
+        if volatility_filtered is not None:
+            result['volatility_filtered'] = volatility_filtered
+        
+        if regime_changes:
+            result['regime_changes'] = regime_changes
+        
+        return result
+
+
+class OrderFlowImbalancePredictor:
+    """
+    Order Flow Imbalance based prediction
+    Replaces MicrostructureKlineBridge with OFI-focused approach
+    """
+
+    def __init__(self):
+        self.lgb_model = None
+        self.feature_names = []
+        self.logger = logging.getLogger(__name__)
+
+    def calculate_ofi(self, orderbook_data: pd.DataFrame) -> pd.Series:
+        """Calculate Order Flow Imbalance (OFI)"""
+        if orderbook_data.empty:
+            return pd.Series()
+        
+        # Ensure we have required columns
+        required_cols = ['bid_size', 'ask_size', 'bid_price', 'ask_price']
+        if not all(col in orderbook_data.columns for col in required_cols):
+            # Try alternative column names
+            if 'buy_usd' in orderbook_data.columns and 'sell_usd' in orderbook_data.columns:
+                orderbook_data['bid_size'] = orderbook_data['buy_usd']
+                orderbook_data['ask_size'] = orderbook_data['sell_usd']
+            else:
+                return pd.Series()
+        
+        # Calculate OFI components
+        bid_size = orderbook_data['bid_size']
+        ask_size = orderbook_data['ask_size']
+        
+        # OFI = Change in bid size - Change in ask size (when price doesn't change)
+        # Plus bid size when best ask increases, minus ask size when best bid decreases
+        ofi = pd.Series(index=orderbook_data.index, dtype=float)
+        
+        for i in range(1, len(orderbook_data)):
+            # Check for price changes
+            bid_price_change = orderbook_data['bid_price'].iloc[i] - orderbook_data['bid_price'].iloc[i-1]
+            ask_price_change = orderbook_data['ask_price'].iloc[i] - orderbook_data['ask_price'].iloc[i-1]
+            
+            # Calculate OFI
+            if bid_price_change == 0 and ask_price_change == 0:
+                # No price change - use size changes
+                ofi.iloc[i] = (bid_size.iloc[i] - bid_size.iloc[i-1]) - (ask_size.iloc[i] - ask_size.iloc[i-1])
+            else:
+                ofi_value = 0
+                if ask_price_change > 0:  # Best ask increased
+                    ofi_value += bid_size.iloc[i]
+                if bid_price_change < 0:  # Best bid decreased
+                    ofi_value -= ask_size.iloc[i]
+                ofi.iloc[i] = ofi_value
+        
+        ofi.iloc[0] = 0  # First value
+        return ofi
+
+    def extract_microstructure_features(self, orderflow_data: pd.DataFrame, 
+                                      orderbook_data: pd.DataFrame,
+                                      kline_data: pd.DataFrame) -> pd.DataFrame:
+        """Extract comprehensive microstructure features"""
+        features = pd.DataFrame(index=kline_data.index)
+        
+        # Price features
+        features['returns'] = kline_data['close'].pct_change()
+        features['log_returns'] = np.log(kline_data['close'] / kline_data['close'].shift(1))
+        features['high_low_ratio'] = kline_data['high'] / kline_data['low']
+        features['close_to_high'] = kline_data['close'] / kline_data['high']
+        features['volume'] = kline_data['volume']
+        
+        # Order flow features
+        if not orderflow_data.empty and 'delta' in orderflow_data.columns:
+            # Aggregate to kline frequency
+            of_resampled = orderflow_data.resample(kline_data.index.freq).agg({
+                'delta': 'sum',
+                'volume': 'sum',
+                'price': 'last'
+            })
+            
+            features['cum_delta'] = of_resampled['delta'].fillna(0)
+            features['of_volume'] = of_resampled['volume'].fillna(0)
+            features['delta_volume_ratio'] = features['cum_delta'] / (features['of_volume'] + 1e-10)
+        
+        # OFI features
+        if not orderbook_data.empty:
+            ofi = self.calculate_ofi(orderbook_data)
+            if not ofi.empty:
+                # Aggregate OFI to kline frequency
+                ofi_resampled = ofi.resample(kline_data.index.freq).sum()
+                features['ofi'] = ofi_resampled.reindex(features.index).fillna(0)
+                features['ofi_ma5'] = features['ofi'].rolling(5).mean()
+                features['ofi_std5'] = features['ofi'].rolling(5).std()
+            
+            # Book imbalance
+            if 'bid_size' in orderbook_data.columns and 'ask_size' in orderbook_data.columns:
+                book_imbalance = (orderbook_data['bid_size'] - orderbook_data['ask_size']) / \
+                               (orderbook_data['bid_size'] + orderbook_data['ask_size'] + 1e-10)
+                imb_resampled = book_imbalance.resample(kline_data.index.freq).mean()
+                features['book_imbalance'] = imb_resampled.reindex(features.index).fillna(0)
+        
+        # Technical indicators
+        features['rsi'] = self._calculate_rsi(kline_data['close'])
+        features['volatility'] = features['returns'].rolling(20).std()
+        
+        # Momentum features
+        for period in [5, 10, 20]:
+            features[f'return_{period}'] = kline_data['close'].pct_change(period)
+            features[f'volume_ma{period}'] = features['volume'].rolling(period).mean()
+        
+        # Clean up
+        features = features.fillna(0)
+        
+        return features
+
+    def train_lgb_model(self, features: pd.DataFrame, target: pd.Series):
+        """Train LightGBM model"""
+        try:
+            import lightgbm as lgb
+        except ImportError:
+            self.logger.error("LightGBM not installed. Using fallback XGBoost.")
+            import xgboost as xgb
+            self.lgb_model = xgb.XGBRegressor(
+                n_estimators=100,
+                learning_rate=0.05,
+                max_depth=5
+            )
+            self.lgb_model.fit(features, target)
+            self.feature_names = features.columns.tolist()
+            return
+        
+        # Prepare data
+        self.feature_names = features.columns.tolist()
+        
+        # LightGBM parameters
+        params = {
+            'objective': 'regression',
+            'metric': 'mae',
+            'boosting_type': 'gbdt',
+            'num_leaves': 31,
+            'learning_rate': 0.05,
+            'feature_fraction': 0.8,
+            'bagging_fraction': 0.8,
+            'bagging_freq': 5,
+            'verbose': -1,
+            'random_state': 42
+        }
+        
+        # Create dataset
+        train_data = lgb.Dataset(features, label=target)
+        
+        # Train model
+        self.lgb_model = lgb.train(
+            params,
+            train_data,
+            num_boost_round=100,
+            valid_sets=[train_data],
+            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
+        )
+
+    def predict_price_impact(self, features: pd.DataFrame) -> Dict[str, float]:
+        """Predict price impact based on order flow imbalance"""
+        if self.lgb_model is None:
+            return {'error': 'Model not trained'}
+        
+        # Make prediction
+        if hasattr(self.lgb_model, 'predict'):
+            if hasattr(self.lgb_model, 'booster_'):  # LightGBM
+                prediction = self.lgb_model.predict(
+                    features[self.feature_names],
+                    num_iteration=self.lgb_model.best_iteration
+                )
+            else:  # XGBoost fallback
+                prediction = self.lgb_model.predict(features[self.feature_names])
+        else:
+            return {'error': 'Invalid model type'}
+        
+        # Calculate confidence based on feature importance
+        confidence = self._calculate_prediction_confidence(features)
+        
+        return {
+            'predicted_price': float(prediction[0]) if isinstance(prediction, np.ndarray) else float(prediction),
+            'confidence': confidence,
+            'features_used': len(self.feature_names)
+        }
+
+    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
+        """Calculate RSI"""
+        delta = prices.diff()
+        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
+        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
+        rs = gain / (loss + 1e-10)
+        rsi = 100 - (100 / (1 + rs))
+        return rsi.fillna(50)
+
+    def _calculate_prediction_confidence(self, features: pd.DataFrame) -> float:
+        """Calculate prediction confidence"""
+        # Base confidence on data quality
+        confidence = 50.0
+        
+        # Check for OFI signal strength
+        if 'ofi' in features.columns:
+            ofi_strength = abs(features['ofi'].iloc[-1])
+            if ofi_strength > features['ofi_std5'].iloc[-1] * 2:
+                confidence += 20
+            elif ofi_strength > features['ofi_std5'].iloc[-1]:
+                confidence += 10
+        
+        # Check for consistent order flow
+        if 'cum_delta' in features.columns:
+            recent_delta = features['cum_delta'].iloc[-5:].mean()
+            if abs(recent_delta) > features['cum_delta'].std():
+                confidence += 15
+        
+        # Check book imbalance
+        if 'book_imbalance' in features.columns:
+            if abs(features['book_imbalance'].iloc[-1]) > 0.3:
+                confidence += 10
+        
+        return min(confidence, 95.0)
+
+    # Compatibility methods to match MicrostructureKlineBridge interface
+    def calculate_microprice(self, book_data: pd.DataFrame) -> float:
+        """Calculate microprice for compatibility"""
+        if book_data.empty:
+            return 0.0
+        
+        latest = book_data.iloc[-1]
+        if 'bid_size' in latest and 'ask_size' in latest:
+            bid_size = float(latest['bid_size'])
+            ask_size = float(latest['ask_size'])
+            bid_price = float(latest.get('bid_price', latest.get('bid', 0)))
+            ask_price = float(latest.get('ask_price', latest.get('ask', 0)))
+            
+            if bid_size + ask_size > 0 and bid_price > 0 and ask_price > 0:
+                return (bid_price * ask_size + ask_price * bid_size) / (bid_size + ask_size)
+        
+        return float(latest.get('mid', 0))
+
+    def project_vwap(self, orderflow_data: pd.DataFrame, current_bar_data: Dict[str, Any]) -> float:
+        """VWAP projection for compatibility"""
+        if orderflow_data.empty:
+            return current_bar_data.get('close', 0)
+        
+        if 'price' in orderflow_data.columns and 'volume' in orderflow_data.columns:
+            total_volume = orderflow_data['volume'].sum()
+            if total_volume > 0:
+                vwap = (orderflow_data['price'] * orderflow_data['volume']).sum() / total_volume
+                return float(vwap)
+        
+        return current_bar_data.get('close', 0)
+
+    # Placeholder for XGBoost compatibility
+    xgb_model = None
+
+
+class MicrostructureKlineBridge:
+    """DEPRECATED - Use OrderFlowImbalancePredictor instead"""
+    pass
+
+
@@ -2352,12 +2386,13 @@
         """
         if n_samples < 100:  # FIX: Increase minimum sample requirement
             return 0.0
-        
+
         # Combine variables to create joint distributions
         xyz = np.vstack([y_future, y_past, x_past]).T
         yz = np.vstack([y_future, y_past]).T
         yx = np.vstack([y_past, x_past]).T
         y = y_past.reshape(-1, 1)
-        
+
         # Calculate probabilities by counting unique occurrences
         p_xyz = self._get_probs(xyz)
         p_yz = self._get_probs(yz)
@@ -2903,7 +2938,7 @@
         self.logger = logging.getLogger(__name__)
 
     def align_arrays(self, *arrays):
-        """Ensure all arrays have same length by truncating to the shortest."""
+        """Ensure all arrays have same length by truncating to the shortest."""
         if not arrays:
             return []
         min_len = min((len(arr) for arr in arrays if arr is not None), default=0)
@@ -3078,9 +3113,9 @@
         # Initialize components
         self.tensor_fusion = MultiModalTensorFusion(self.lookback_size)
         self.info_flow_analyzer = InformationFlowAnalyzer()
-        self.microstructure_bridge = MicrostructureKlineBridge()
+        self.ofip = OrderFlowImbalancePredictor()
         self.advanced_tools = AdvancedMathematicalTools()
 
         # Initialize enhanced components
         self.hawkes_process = HawkesProcess()
-        self.ukf_filter = UnscentedKalmanFilter(dim_x=2, dim_z=1)  # Price and velocity
-        self.ms_garch = MarkovSwitchingGARCH()
+        self.ukf_filter = AdaptiveParticleFilter(dim_x=3, dim_z=1)  # Price, velocity, volatility
+        self.ms_garch = RegimeSwitchingStochasticVolatility()
         self.cnn_predictor = CNNMarketPredictor()
@@ -4435,7 +4470,7 @@
             return decomposition_results
 
         # Advanced Mathematical Tools
         print(f"  1. Running Multifractal Spectrum Analysis...")
-        self.analysis_results[timeframe]['multifractal'] = self.advanced_tools.multifractal_spectrum_analysis(returns)
+        self.analysis_results[timeframe]['multifractal'] = self.advanced_tools.multifractal_spectrum_analysis(returns)
 
         print(f"  2. Running Sample Entropy...")
@@ -4657,68 +4692,82 @@
 
     def unscented_kalman_filter(self, observations: np.ndarray) -> Dict[str, Any]:
-        """Apply Unscented Kalman Filter to price data"""
-        # Initialize UKF with 2D state (price and velocity)
-        ukf = UnscentedKalmanFilter(dim_x=2, dim_z=1)
-
-        # Set initial state
-        ukf.x = np.array([observations[0], 0])  # Initial price and zero velocity
-        ukf.P = np.eye(2) * 10  # Reduced initial uncertainty from 100 to 10
-
-        # Process and measurement noise - adjusted for better stability
-        ukf.Q = np.array([[0.001, 0], [0, 0.0001]])  # Reduced process noise
-        ukf.R = np.array([[0.1]])  # Measurement noise
-
-        # Filter the series
+        """Apply Particle Filter to price data"""
+        # Initialize Particle Filter with 3D state (price, velocity, volatility)
+        pf = AdaptiveParticleFilter(dim_x=3, dim_z=1, n_particles=2000)
+        
+        # Set initial state with volatility
+        initial_state = np.array([observations[0], 0, 0.02])  # Initial price, zero velocity, 2% volatility
+        initial_cov = np.diag([10, 0.1, 0.01])
+        
+        # Initialize particles
+        pf.initialize(initial_state, initial_cov)
+        
+        # Process and measurement noise
+        pf.Q = np.diag([0.001, 0.0001, 0.0001])
+        pf.R = np.array([[0.1]])
+        
+        # Filter the series
         try:
-            results = ukf.filter_series(observations)
+            results = pf.filter_series(observations)
         except Exception as e:
-            # Fallback to simple estimates if UKF fails
-            self.logger.warning(f"UKF failed, using fallback: {e}")
+            # Fallback to simple estimates if PF fails
+            self.logger.warning(f"Particle Filter failed, using fallback: {e}")
             return {
                 'filtered': observations,
                 'velocity': np.zeros_like(observations),
                 'innovations': np.zeros_like(observations),
                 'future_price': observations[-1],
                 'future_velocity': 0,
-                'final_covariance': np.eye(2),
+                'final_covariance': np.eye(3),
                 'price_uncertainty': np.std(observations),
                 'velocity_uncertainty': 0,
+                'volatility_filtered': np.full_like(observations, np.std(observations)),
+                'regime_changes': [],
                 'error': str(e)
             }
-
+        
         # Extract predictions
         filtered_prices = results['filtered_states'][:, 0]
         filtered_velocity = results['filtered_states'][:, 1]
-
+        filtered_volatility = results.get('volatility_filtered', results['filtered_states'][:, 2] if results['filtered_states'].shape[1] > 2 else np.full_like(filtered_prices, 0.02))
+        
         # Future prediction (1 step ahead)
-        ukf.predict()
-        future_price = ukf.x[0]
-        future_velocity = ukf.x[1]
-
+        pf.predict()
+        future_state, future_cov = pf.get_state_estimate()
+        future_price = future_state[0]
+        future_velocity = future_state[1]
+        
         return {
             'filtered': filtered_prices,
             'velocity': filtered_velocity,
             'innovations': results['innovations'],
             'future_price': future_price,
             'future_velocity': future_velocity,
             'final_covariance': results['final_covariance'],
             'price_uncertainty': np.sqrt(results['final_covariance'][0, 0]),
-            'velocity_uncertainty': np.sqrt(results['final_covariance'][1, 1])
+            'velocity_uncertainty': np.sqrt(results['final_covariance'][1, 1]),
+            'volatility_filtered': filtered_volatility,
+            'regime_changes': results.get('regime_changes', [])
         }
 
     def markov_switching_garch(self, returns: np.ndarray) -> Dict[str, Any]:
-        """Apply Markov-Switching GARCH model"""
+        """Apply Regime-Switching Stochastic Volatility model"""
         try:
-            # Fit the MS-GARCH model
-            ms_garch = MarkovSwitchingGARCH(n_states=3)
-            ms_garch.fit(returns)
-
-            # Predict volatility
-            vol_forecast = ms_garch.predict_volatility(returns, horizon=10)
-
-            # Get current regime probabilities
-            current_probs = vol_forecast['regime_probabilities']
-
-            # Determine current regime
+            # Fit the RS-SV model
+            rs_sv = RegimeSwitchingStochasticVolatility(n_states=3)
+            rs_sv.fit(returns)
+            
+            # Predict volatility with Monte Carlo
+            vol_forecast = rs_sv.predict_volatility(returns, horizon=10, n_simulations=1000)
+            
+            # Extract results
+            current_probs = vol_forecast['regime_probabilities']
             regime_labels = ['Low Vol', 'Medium Vol', 'High Vol']
             current_regime = regime_labels[vol_forecast['current_regime']]
 
             return {
                 'volatility_forecast': vol_forecast['volatility_forecast'],
+                'volatility_percentiles': vol_forecast['volatility_percentiles'],
                 'state_volatilities': vol_forecast['state_volatilities'],
                 'current_regime': current_regime,
                 'regime_probabilities': vol_forecast['regime_probabilities'],
-                'state_parameters': ms_garch.state_params,
-                'transition_matrix': vol_forecast.get('transition_matrix')
+                'state_parameters': rs_sv.state_params,
+                'transition_matrix': vol_forecast.get('transition_matrix'),
+                'VaR_95': vol_forecast.get('VaR_95', np.nan),
+                'CVaR_95': vol_forecast.get('CVaR_95', np.nan)
             }
         except Exception as e:
-            self.logger.exception("MS-GARCH error")
+            self.logger.exception("RS-SV error")
             # Fallback to standard volatility
             vol = np.std(returns)
             return {
                 'volatility_forecast': np.full(10, vol),
+                'volatility_percentiles': {'50%': np.full(10, vol)},
                 'state_volatilities': np.array([[vol] * 10]),
                 'current_regime': 'Unknown',
                 'regime_probabilities': {'Low Vol': 0.33, 'Medium Vol': 0.33, 'High Vol': 0.34},
+                'VaR_95': vol * 1.645,
+                'CVaR_95': vol * 2.063,
                 'error': str(e)
             }
@@ -4768,39 +4817,54 @@
 
     def permutation_entropy(self, signal: np.ndarray, order: int = 3) -> Dict[str, float]:
-        """Calculate permutation entropy"""
+        """Calculate weighted permutation entropy"""
         n = len(signal)
         if n < order + 1:
             return {'pe': 1.0, 'complexity': 'HIGH', 'predictability': 0.0}
-
-        # Create ordinal patterns
-        patterns = []
+        
+        # Use defaultdict for pattern accumulation
+        from collections import defaultdict
+        pattern_weights = defaultdict(float)
+        total_weight = 0.0
+        
         for i in range(n - order + 1):
-            # Get order indices
-            indices = np.argsort(signal[i:i+order])
+            # Get segment
+            segment = signal[i:i+order]
+            
+            # Calculate amplitude weight
+            segment_std = np.std(segment)
+            signal_std = np.std(signal)
+            weight = segment_std / (signal_std + 1e-10)
+            
+            # Get ordinal pattern
+            indices = np.argsort(segment)
             pattern = tuple(indices)
-            patterns.append(pattern)
-
-        # Count pattern occurrences
-        pattern_counts = Counter(patterns)
-
-        # Calculate probabilities
-        probs = np.array(list(pattern_counts.values())) / len(patterns)
-
-        # Shannon entropy
-        pe = -np.sum(probs * np.log(probs))
-
-        # Normalize
+            
+            # Accumulate weighted pattern
+            pattern_weights[pattern] += weight
+            total_weight += weight
+        
+        # Calculate weighted probabilities
+        if total_weight > 0:
+            probs = np.array([w / total_weight for w in pattern_weights.values()])
+        else:
+            # Fallback to uniform distribution
+            probs = np.ones(len(pattern_weights)) / len(pattern_weights)
+        
+        # Calculate weighted entropy
+        pe = -np.sum(probs * np.log(probs + 1e-10))
+        
+        # Normalize
         max_entropy = np.log(math.factorial(order))
         pe_normalized = self.safe_divide(pe, max_entropy, 1.0)
-
-        # Interpret
+        
+        # Interpret complexity
         if pe_normalized < 0.3:
             complexity = "LOW"
         elif pe_normalized < 0.7:
             complexity = "MEDIUM"
         else:
             complexity = "HIGH"
-
+        
         predictability = 1 - pe_normalized
 
         return {
             'pe': pe_normalized,
             'complexity': complexity,
             'predictability': predictability,
-            'n_unique_patterns': len(pattern_counts),
-            'max_patterns': math.factorial(order)
+            'n_unique_patterns': len(pattern_weights),
+            'max_patterns': math.factorial(order),
+            'weighted': True
         }
@@ -4941,77 +5005,105 @@
             n_iter=100,
             random_state=42,
-            init_params="kmeans",  # Better initialization
-            tol=0.01  # Increase tolerance
+            init_params="kmeans",
+            tol=0.01
         )
         
+        # Add Variational Bayes parameters
+        model.transmat_prior = 1.0
+        model.startprob_prior = 1.0
+        
         model.fit(X_scaled)
 
         # Get states
         states = model.predict(X_scaled)
+        
+        # Get posterior probabilities for uncertainty
+        posterior_probs = model.predict_proba(X_scaled)
+        
+        # Calculate state uncertainty using entropy
+        state_uncertainty = -np.sum(posterior_probs * np.log(posterior_probs + 1e-10), axis=1)
+        mean_uncertainty = np.mean(state_uncertainty)
 
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
-            'log_likelihood': log_prob
+            'log_likelihood': log_prob,
+            'state_uncertainty': state_uncertainty,
+            'mean_uncertainty': mean_uncertainty,
+            'posterior_probs': posterior_probs
         }
 
     except Exception as e:
         self.logger.exception("HMM error")
         return {
             'current_state': 'UNKNOWN',
             'state_probabilities': [0.33, 0.33, 0.34],
             'state_means': {'BEAR': -0.01, 'NEUTRAL': 0, 'BULL': 0.01},
-            'error': str(e)
+            'error': str(e),
+            'state_uncertainty': np.array([1.0]),
+            'mean_uncertainty': 1.0,
+            'posterior_probs': np.array([[0.33, 0.33, 0.34]])
         }
 
@@ -5562,8 +5654,8 @@
                         
                         # Prepare features
-                        features = self.microstructure_bridge._prepare_xgb_features(
+                        features = self.ofip.extract_microstructure_features(
+                            of_slice, ob_slice, pd.DataFrame(current_bar, index=[current_time])
-                            current_bar, of_slice, ob_slice
                         )
                         
@@ -5643,7 +5735,7 @@
                     'timeframe': '15m'
                 }
                 
                 # Get orderflow data
                 orderflow_data = pd.DataFrame()
                 orderflow_analysis = {}
                 if '15m' in self.collected_data and 'order_flow' in self.collected_data['15m']:
@@ -5683,7 +5775,7 @@
                 if '15m' in self.collected_data and 'orderbook' in self.collected_data['15m']:
                     book_df = pd.DataFrame(self.collected_data['15m']['orderbook'])
                     if not book_df.empty:
                         # Calculate microprice
-                        microprice = self.microstructure_bridge.calculate_microprice(book_df)
+                        microprice = self.ofip.calculate_microprice(book_df)
                         
                         # Get latest book imbalance
                         latest_book = book_df.iloc[-1]
@@ -6731,11 +6823,11 @@
             
             # Load XGBoost model
             xgb_model_path = os.path.join(model_dir, "xgb_microstructure_model.pkl")
             if os.path.exists(xgb_model_path):
                 with open(xgb_model_path, 'rb') as f:
                     model_data = pickle.load(f)
                     
                     # Check if it's versioned format
                     if isinstance(model_data, dict) and 'model' in model_data:
-                        self.microstructure_bridge.xgb_model = model_data['model']
+                        self.ofip.lgb_model = model_data['model']
                         self.logger.info(f"XGBoost model loaded (version: {model_data.get('version', 'unknown')})")
                     else:
                         # Legacy format
-                        self.microstructure_bridge.xgb_model = model_data
+                        self.ofip.lgb_model = model_data
                         self.logger.info("XGBoost model loaded (legacy format)")
                 
-                models_loaded.append("XGBoost Microstructure Model")
+                models_loaded.append("LightGBM/XGBoost Microstructure Model")
             
             # Load MS-GARCH model
             msgarch_model_path = os.path.join(model_dir, "ms_garch_model.pkl")
             if os.path.exists(msgarch_model_path):
@@ -6810,11 +6902,11 @@
                 saved_models.append("CNN Market Predictor")
             
             # Save XGBoost model
-            if self.microstructure_bridge.xgb_model is not None:
+            if self.ofip.lgb_model is not None:
                 with open(os.path.join(model_dir, "xgb_microstructure_model.pkl"), 'wb') as f:
                     model_data = {
                         'version': '1.0',
-                        'model': self.microstructure_bridge.xgb_model,
+                        'model': self.ofip.lgb_model,
                         'created_at': datetime.now().isoformat()
                     }
                     pickle.dump(model_data, f)
-                saved_models.append("XGBoost Microstructure Model")
+                saved_models.append("LightGBM/XGBoost Microstructure Model")
             
             # Save MS-GARCH model