import os
import json
import hashlib
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import utide
import shap

from skyfield.api import load, wgs84

from .utils import cosd, sind, custom_round, calc_stats, save_inputs_to_pickle, load_inputs_from_pickle, fit_trend_initial_coeffs
from .models import build_model, get_custom_objects
from . import models


DEFAULT_INPUT_CONFIG = {
    # orders intentionally start at 1 to match legacy column naming/behavior
    "Radiational": {"degrees": [1, 2], "orders": {1: [1], 2: [1, 2]}},
    "Gravitational": {"degrees": [2, 3], "orders": {2: [1, 2], 3: [1, 2, 3]}},
}


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Input DataFrame must have a DatetimeIndex.")
    return df


def _median_dt_seconds(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 0.0
    diffs = np.diff(index.view("int64")) / 1e9
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    return float(np.median(diffs)) if len(diffs) else 0.0


class RTide:
    """
    RTide: empirical response method for tidal analysis/prediction.

    Supports:
      - Scalar elevation output: DataFrame first column named 'observations'
      - Vector currents output: DataFrame first two columns named 'u', 'v'

    New in this refactor:
      - Fixed-location forcing option (default: legacy station-specific):
          location_mode="station" or "fixed"
      - Optional precomputed forcing cache (only applies in fixed mode):
          use_precomputed_inputs=True
        Precompute is stored on a 6-minute grid in compressed float16 npz.
        Higher-resolution requests are interpolated in time.
      - Architecture selection via models.py (supports 'response' and 'siren')
      - Feature-wise X scaling is opt-in (legacy global scaling remains default)
      - Legacy linear warm-start training is deprecated/ignored
    """

    def __init__(self, ts, lat, lon):
        """
        Parameters
        ----------
        ts : pandas.DataFrame
            Time series with DatetimeIndex. Either:
              - first column 'observations' (scalar elevation), or
              - first two columns 'u','v' (vector currents).
            Additional columns are treated as exogenous inputs.
        lat, lon : float
            Station latitude/longitude (metadata and for legacy station-specific forcing).

        Notes
        -----
        This __init__ signature is intentionally kept identical to the legacy RTide API.
        New functionality (fixed-location forcing, precomputed cache, etc.) is configured
        via Prepare_Inputs().
        """
        self.ts = _ensure_datetime_index(pd.DataFrame(ts).copy())
        self.lat = float(lat)
        self.lon = float(lon)

        # Physical constants (legacy)
        self.M_E = 5.9722e24
        self.M_M = 7.3e22
        self.M_S = 1.989e30
        self.E_r = 6371.01e3
        self.solar_constant = 1.946 / 100  # preserved from legacy

        # Default forcing/location configuration (overridable in Prepare_Inputs)
        self.location_mode = "station"      # 'station' (legacy) or 'fixed'
        self.fixed_lat = 45.0
        self.fixed_lon = 0.0
        self.use_precomputed_inputs = False
        self.precomputed_cache_dir = None
        self.precomputed_dtype = "float16"
        self.precomputed_base_freq = "6min"
        self.precomputed_version = "v1"
        self.allow_precompute_write = True

        # Input-function configuration
        self.input_config = DEFAULT_INPUT_CONFIG
        self.ephemeris = "de421.bsp"

        # Schema inference (outputs + exogenous columns)
        self._infer_io_schema()
        self._validate_inputs()

        # State
        self.sample_rate = None
        self.nonunilags: List[float] = [0.0]
        self.multivariate_lags: List[float] = [0.0]
        self.const_names: List[str] = ["real"]
        self.max_lag: float = 0.0
        self.padded_ts: Optional[pd.DataFrame] = None
        self.global_tide_base: Optional[pd.DataFrame] = None
        self.prepped_dfs: Optional[pd.DataFrame] = None
        self.prediction_dfs: Optional[pd.DataFrame] = None

        # ML artifacts
        self.model: Optional[tf.keras.Model] = None
        self.scaler_X: Optional[StandardScaler] = None
        self.scaler_Y: Optional[StandardScaler] = None
        self.featurewise_X_scaling: bool = False
        self.history: Optional[dict] = None

        # Skyfield cached objects
        self._skyfield_cache: dict = {}

    # ----------------------------
    # Schema
    # ----------------------------
    def _infer_io_schema(self):
      """Infer output(s) and exogenous inputs from the input DataFrame.

      Supported schemas (backwards compatible):
      - Elevation: first column is named 'observations'
      - Currents: first two columns are named 'u' and 'v' (in that order)
      """
      if not isinstance(self.ts, pd.DataFrame):
          raise ValueError("ts must be a pandas DataFrame")
      if self.ts.shape[1] < 1:
          raise ValueError("ts must have at least one column")

      cols = list(self.ts.columns)
      if cols[0] == 'observations':
          self.output_mode = 'elevation'
          self.output_columns = ['observations']
      elif len(cols) >= 2 and cols[0] == 'u' and cols[1] == 'v':
          self.output_mode = 'currents'
          self.output_columns = ['u', 'v']
      else:
          raise ValueError(
              "Unsupported input schema. Expected either: "
              "(1) first column named 'observations', or "
              "(2) first two columns named 'u' and 'v' (in that order). "
              f"Got columns={cols}"
          )

      self.n_outputs = len(self.output_columns)
      self.exog_columns = cols[self.n_outputs:]

      # Backwards compatible meaning: multi == has extra (non-output) columns.
      self.multi = len(self.exog_columns) > 0
    def _validate_inputs(self):
      if not isinstance(self.ts.index, pd.DatetimeIndex):
          raise ValueError("The DataFrame does not have a DatetimeIndex.")
      if not isinstance(self.lat, (int, float)):
          raise ValueError("Latitude must be a number.")
      if not isinstance(self.lon, (int, float)):
          raise ValueError("Longitude must be a number.")
      # Schema validation is handled in _infer_io_schema().
    # ----------------------------
    # Location helpers
    # ----------------------------
    def _inputs_latlon(self) -> Tuple[float, float]:
        if self.location_mode == "fixed":
            return self.fixed_lat, self.fixed_lon
        return self.lat, self.lon

    # ----------------------------
    # Precompute cache helpers
    # ----------------------------
    def _cache_dir(self) -> str:
        if self.precomputed_cache_dir:
            d = self.precomputed_cache_dir
        else:
            d = os.path.join(os.path.expanduser("~"), ".cache", "rtide")
        os.makedirs(d, exist_ok=True)
        return d

    def _precompute_config(self) -> dict:
        lat, lon = self._inputs_latlon()
        cfg = {
            "version": self.precomputed_version,
            "base_freq": self.precomputed_base_freq,
            "dtype": self.precomputed_dtype,
            "location_mode": self.location_mode,
            "fixed_lat": float(lat),
            "fixed_lon": float(lon),
            "input_config": self.input_config,
            "ephemeris": self.ephemeris,
        }
        return cfg

    def _precompute_key(self) -> str:
        cfg = json.dumps(self._precompute_config(), sort_keys=True).encode("utf-8")
        return hashlib.sha1(cfg).hexdigest()[:10]

    def _year_cache_path(self, year: int) -> str:
        key = self._precompute_key()
        return os.path.join(self._cache_dir(), f"rtide_forcing_{key}_{int(year)}.npz")

    def _load_year_cache(self, year: int):
        path = self._year_cache_path(year)
        if not os.path.exists(path):
            return None
        z = np.load(path, allow_pickle=False)
        cfg = json.loads(str(z["cfg"]))
        if cfg != self._precompute_config():
            return None
        t_ns = z["t_ns"].astype("int64", copy=False)
        X = z["X"]  # stored float16
        cols = [str(c) for c in z["cols"].tolist()]
        return t_ns, X, cols

    def _save_year_cache(self, year: int, t_ns: np.ndarray, X: np.ndarray, cols: list[str]):
        if not self.allow_precompute_write:
            return
        path = self._year_cache_path(year)
        cfg_str = json.dumps(self._precompute_config(), sort_keys=True)
        cols_arr = np.asarray(cols, dtype="U")
        # Ensure float16 for size, then compress
        X = np.asarray(X, dtype=np.float16)
        np.savez_compressed(path, t_ns=t_ns.astype("int64"), X=X, cols=cols_arr, cfg=cfg_str)

    def _six_min_grid(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
        # normalize to base grid
        start = pd.Timestamp(start).floor(self.precomputed_base_freq)
        end = pd.Timestamp(end).ceil(self.precomputed_base_freq)
        return pd.date_range(start=start, end=end, freq=self.precomputed_base_freq)

    def _get_or_build_precomputed_base(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """
        Get or build base (unlagged) forcing features on the 6-minute grid
        over [start,end], using year-chunked cache.
        """
        if self.location_mode != "fixed":
            raise ValueError("_get_or_build_precomputed_base is only valid for location_mode='fixed'.")

        idx6 = self._six_min_grid(start, end)
        years = np.unique(idx6.year)
        chunks = []

        for y in years:
            idx_y = idx6[idx6.year == y]
            t_req = idx_y.view("int64")

            cached = self._load_year_cache(int(y))
            if cached is None:
                df_feat = self._compute_global_tide_base(index=idx_y)
                X = df_feat.to_numpy(dtype=np.float16, copy=False)
                self._save_year_cache(int(y), t_req, X, list(df_feat.columns))
                chunks.append(df_feat)
                continue

            t_ns, X_cache, cols = cached
            pos = {int(t): i for i, t in enumerate(t_ns)}
            miss = np.array([int(t) not in pos for t in t_req], dtype=bool)

            if miss.any():
                idx_missing = idx_y[miss]
                df_missing = self._compute_global_tide_base(index=idx_missing)
                if list(df_missing.columns) != list(cols):
                    raise ValueError("Precomputed cache columns mismatch. Bump precomputed_version.")
                X_missing = df_missing.to_numpy(dtype=np.float16, copy=False)

                t_new = np.concatenate([t_ns, idx_missing.view("int64")])
                X_new = np.concatenate([X_cache, X_missing], axis=0)
                order = np.argsort(t_new)
                t_new = t_new[order]
                X_new = X_new[order]
                self._save_year_cache(int(y), t_new, X_new, cols)

                t_ns, X_cache = t_new, X_new
                pos = {int(t): i for i, t in enumerate(t_ns)}

            rows = [pos[int(t)] for t in t_req]
            df_y = pd.DataFrame(np.asarray(X_cache[rows, :], dtype=np.float32), index=idx_y, columns=cols)
            chunks.append(df_y)

        base = pd.concat(chunks).sort_index()
        return base.loc[idx6]

    def _interpolate_to_index(self, base_df: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.DataFrame:
        # Union index to allow time interpolation
        target_index = pd.DatetimeIndex(target_index)
        union = base_df.index.union(target_index)
        tmp = base_df.reindex(union)
        tmp = tmp.interpolate(method="time")
        tmp = tmp.ffill().bfill()
        return tmp.reindex(target_index)

    # ----------------------------
    # Forcing computations
    # ----------------------------
    def Spherical_Harmonic(self, degree: int, order: int, lat: float, lon: float) -> complex:
        """
        Complex spherical harmonic Y_degree^order at station location.

        Uses geographic colatitude (theta = 90 - lat) and longitude (phi = lon).
        """
        from scipy.special import lpmv
        degree = int(degree)
        order = int(order)
        if order < 0 or degree < 0 or order > degree:
            raise ValueError("Require 0 <= order <= degree.")

        theta = np.deg2rad(90.0 - float(lat))  # colatitude
        phi = np.deg2rad(float(lon))

        # Associated Legendre P_l^m(cos(theta)) (SciPy includes Condon-Shortley phase)
        Plm = lpmv(order, degree, np.cos(theta))
        # Normalization
        from math import factorial, sqrt, pi
        norm = sqrt(((2 * degree + 1) / (4 * pi)) * (factorial(degree - order) / factorial(degree + order)))
        return norm * Plm * np.exp(1j * order * phi)

    def _get_skyfield(self):
        if "tscale" not in self._skyfield_cache:
            self._skyfield_cache["tscale"] = load.timescale()
        if "planets" not in self._skyfield_cache:
            self._skyfield_cache["planets"] = load(self.ephemeris)
        return self._skyfield_cache["tscale"], self._skyfield_cache["planets"]

    def _compute_astro(self, index: pd.DatetimeIndex, lat: float, lon: float) -> dict:
        """
        Compute astro geometry relative to (lat,lon) for all times in index.
        Returns dict with distances and zenith angles.
        """
        tscale, planets = self._get_skyfield()
        times = tscale.from_datetimes(index.to_pydatetime())

        earth = planets["earth"]
        moon = planets["moon"]
        sun = planets["sun"]

        # Earth-centric distances
        earth2moon = (moon.at(times) - earth.at(times)).distance().m
        earth2sun = (sun.at(times) - earth.at(times)).distance().m

        # Topocentric observer at chosen lat/lon
        observer = earth + wgs84.latlon(lat, lon)

        app_moon = observer.at(times).observe(moon).apparent()
        alt_moon, az_moon, dist_moon = app_moon.altaz()
        zenith_moon = 90.0 - alt_moon.degrees

        app_sun = observer.at(times).observe(sun).apparent()
        alt_sun, az_sun, dist_sun = app_sun.altaz()
        zenith_sun = 90.0 - alt_sun.degrees

        mu_moon = cosd(zenith_moon)
        mu_sun = cosd(zenith_sun)

        # Legendre P_n(mu) for degrees up to max in config
        from scipy.special import eval_legendre
        max_deg = 0
        for it in ("Radiational", "Gravitational"):
            max_deg = max(max_deg, max(self.input_config.get(it, {}).get("degrees", [0])))

        solar_leg = {n: eval_legendre(n, mu_sun) for n in range(max_deg + 1)}
        lunar_leg = {n: eval_legendre(n, mu_moon) for n in range(max_deg + 1)}

        return {
            "station2moon": dist_moon.m,
            "station2sun": dist_sun.m,
            "earth2moon": earth2moon,
            "earth2sun": earth2sun,
            "zenith_moon": np.asarray(zenith_moon),
            "zenith_sun": np.asarray(zenith_sun),
            "mu_moon": np.asarray(mu_moon),
            "mu_sun": np.asarray(mu_sun),
            "mean_r_moon": float(np.mean(earth2moon)),
            "mean_r_sun": float(np.mean(earth2sun)),
            "Solar_Legendre": solar_leg,
            "Lunar_Legendre": lunar_leg,
        }

    def Radiational(self, degree: int, order: int, astro: dict, lat: float, lon: float) -> np.ndarray:
        """
        Radiational input function, normalized by spherical harmonic.
        """
        degree = int(degree)
        order = int(order)
        if degree > 2:
            raise ValueError("Currently only equipped to handle degree 0..2 radiational.")
        parallax = 1.0 / 23455.0
        k_n = [1 / 4 + (1 / 6) * parallax, (1 / 2) + (3 / 8) * parallax, (5 / 16) + (1 / 3) * parallax]
        k = k_n[degree]

        SPH = self.Spherical_Harmonic(degree, order, lat=lat, lon=lon)
        zenith_sun = astro["zenith_sun"]
        station2sun = astro["station2sun"]
        mu_sun = astro["mu_sun"]
        mean_r_sun = astro["mean_r_sun"]

        rad = self.solar_constant * (mean_r_sun / station2sun) * k * (mu_sun ** degree)
        mask = (zenith_sun >= 90) & (zenith_sun <= 180)
        rad = np.asarray(rad, dtype=np.float64)
        rad[mask] = 0.0
        return rad / SPH

    def Gravitational(self, degree: int, order: int, astro: dict, lat: float, lon: float) -> np.ndarray:
        """
        Gravitational input function (moon + sun), normalized by spherical harmonic.
        """
        degree = int(degree)
        order = int(order)
        if degree > 3:
            raise ValueError("Currently only equipped to handle degree 0..3 gravitational.")
        SPH = self.Spherical_Harmonic(degree, order, lat=lat, lon=lon)

        earth2moon = astro["earth2moon"]
        earth2sun = astro["earth2sun"]
        mean_r_moon = astro["mean_r_moon"]
        mean_r_sun = astro["mean_r_sun"]

        K_n_Moon = self.E_r * (self.M_M / self.M_E) * (self.E_r / earth2moon) ** (degree + 1)
        K_n_Sun = self.E_r * (self.M_S / self.M_E) * (self.E_r / earth2sun) ** (degree + 1)

        leg_m = astro["Lunar_Legendre"][degree]
        leg_s = astro["Solar_Legendre"][degree]

        grav_Moon = K_n_Moon * (mean_r_moon / earth2moon) ** (degree + 1) * leg_m
        grav_Sun = K_n_Sun * (mean_r_sun / earth2sun) ** (degree + 1) * leg_s

        return (grav_Moon + grav_Sun) / SPH

    def _compute_global_tide_base(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Compute unlagged (base) forcing features for given index at the selected inputs lat/lon.
        Returns real-valued DataFrame with columns:
            '{InputType}_{degree}_{order}'
        """
        index = pd.DatetimeIndex(index)
        lat, lon = self._inputs_latlon()
        astro = self._compute_astro(index=index, lat=lat, lon=lon)

        cols = {}
        for input_type, cfg in self.input_config.items():
            degrees = cfg.get("degrees", [])
            orders_map = cfg.get("orders", {})
            if input_type not in ("Radiational", "Gravitational"):
                continue
            for deg in degrees:
                for ord_ in orders_map.get(int(deg), []):
                    if input_type == "Radiational":
                        vals = self.Radiational(deg, ord_, astro=astro, lat=lat, lon=lon)
                    else:
                        vals = self.Gravitational(deg, ord_, astro=astro, lat=lat, lon=lon)
                    cols[f"{input_type}_{int(deg)}_{int(ord_)}"] = np.real(vals).astype(np.float32)

        return pd.DataFrame(cols, index=index)

    # ----------------------------
    # Lag computation (legacy-compatible)
    # ----------------------------
    def Prep(self, uniform, symmetrical, multivariate_lags):
        """
        Compute appropriate lags (hours) and multivariate lag policy.

        Parameters
        ----------
        uniform : False or [s, tau]
            If provided, perform uniform lag embedding where s is number of steps and
            tau is lag spacing in hours.
        symmetrical : bool
            Whether to include positive lags (future) as well as negative lags.
        multivariate_lags : False/'standard'/'all'/'negative'/list
            Controls how exogenous inputs are lagged. Default False uses only realtime (0).
        """
        def neg(lst):
            return [x for x in lst if x <= 0]

        if uniform:
            # Uniform lag embedding (legacy behavior)
            assert isinstance(uniform, list), "uniform must be a list"
            assert len(uniform) == 2, "uniform must be a list of length 2"
            assert isinstance(uniform[0], int), "uniform[0] must be an integer"

            nonuni = []
            for s in range(-uniform[0], uniform[0] + 1):
                nonuni.append(self.sample_rate * s * uniform[1])

            if not symmetrical:
                nonuni = neg(nonuni)
            self.const_names = [str(ba) for ba in nonuni]

        else:
            # Standard non-uniform RTide lags (legacy, tidal-periodicity-based)
            constits = ['K1', 'M2', 'M3', 'M4', '2MK5', '2SK5', 'M6', '3MK7', 'M8']
            frequencies = [0.04178075, 0.0805114, 0.1207671, 0.1610228, 0.20280355, 0.20844741, 0.2415342, 0.28331495, 0.3220456]

            nonuni = []
            for _, j in enumerate(frequencies):
                cand = -self.sample_rate * 2 * np.pi / j
                if (len(nonuni) != 0) and (nonuni[-1] == cand):
                    pass
                else:
                    nonuni.append(cand)
            nonuni.append(0)

            for i in np.flip(frequencies):
                cand = self.sample_rate * 2 * np.pi / i
                if nonuni[-1] == cand:
                    pass
                else:
                    nonuni.append(cand)

            new_vals = constits.copy()
            new_vals.append('real')
            if not symmetrical:
                nonuni = neg(nonuni)
                self.const_names = new_vals
            else:
                new_vals2 = list(np.flip(constits))
                self.const_names = new_vals + new_vals2

        # Convert from 'samples' scale to hours (legacy convention) and round
        if len(nonuni) == 0:
            self.max_lag = 0.0
            self.nonunilags = [0.0]
        else:
            self.max_lag = abs(min(nonuni) / self.sample_rate)
            roundval = 1 if self.sample_rate <= 1 else self.sample_rate
            lagstouse = [lag / self.sample_rate for lag in nonuni]
            self.nonunilags = custom_round(lagstouse, roundval)

        # Multivariate lagging policy (for exogenous inputs)
        if self.multi:
            if not multivariate_lags:
                self.multivariate_lags = [0]
            elif multivariate_lags in ('standard', 'all'):
                # Avoid duplicating realtime exog values
                multi_lags_list = self.nonunilags.copy()
                if 0 in multi_lags_list:
                    multi_lags_list.pop(multi_lags_list.index(0))
                self.multivariate_lags = multi_lags_list
            elif multivariate_lags == 'negative':
                self.multivariate_lags = [x for x in self.nonunilags if x < 0]
            elif isinstance(multivariate_lags, list):
                self.multivariate_lags = [float(x) for x in multivariate_lags]
                # Only include realtime (0.0) if explicitly allowed AND not a negative-only mode
                #if self.multivariate_realtime and multivariate_lags == 'negative':
                #    pass  # negative-only: do NOT add 0.0
                #elif self.multivariate_realtime:
                if 0.0 not in self.multivariate_lags and self.multivariate_realtime:
                    self.multivariate_lags.insert(0, 0.0)
            else:
                self.multivariate_lags = [0]
        else:
            self.multivariate_lags = [0]

    def Compute_Time_Lags(self):
        """
        Pad the time index so that ts shifted by all lags is available.
        """
        times = self.ts.index.to_numpy()
        times_with_lags = []
        for lagval in self.nonunilags:
            shifted_times = times + pd.Timedelta(hours=lagval)
            times_with_lags.append(pd.DatetimeIndex(shifted_times))

        combined = times_with_lags[0]
        for idx in times_with_lags[1:]:
            combined = combined.union(idx)

        padded_df = pd.DataFrame(index=combined)
        padded_df = padded_df.merge(self.ts, how="left", left_index=True, right_index=True)
        self.padded_ts = padded_df

    # ----------------------------
    # Prepare Inputs
    # ----------------------------
    def Prepare_Inputs(self, **kwargs):
        """
        Compute and format input features for response analysis, handling lags and precomputation.

        This method synchronizes class attributes with provided keyword arguments, 
        prepares time-lagged forcing features (tides/astronomy), and manages 
        local caching or precomputed data interpolation.

        Parameters (via **kwargs):
            location_mode (str): "station" (default) or "fixed". "fixed" allows for 
                coordinate-based precomputation.
            use_precomputed_inputs (bool): If True, attempts to load features from 
                a precomputed cache (only applies to "fixed" mode).
            precomputed_cache_dir (str): Path to store/load binary feature files.
            multivariate_realtime (bool): If True, includes real-time exogenous 
                inputs without lagging.
            prediction (bool): If True, stores result in `self.prediction_dfs`; 
                otherwise in `self.prepped_dfs`.
            save (bool): If True, caches the resulting DataFrame to a CSV 
                (station mode only).
            fixed_lat/fixed_lon (float): Coordinates for "fixed" location mode.
            sample_rate (float): Sampling frequency in hours. If None, inferred 
                from the index.
            uniform_lags/multivariate_lags (bool): Toggles for specific lagging 
                strategies defined in Prep().
            self_prediction (bool) or list: If list, lags given in the list of the output variable are 
                included as features.
            radiational (bool): If False, does not include radiational tide components (From MC 1966).
        """
        defaults = {
            "uniform_lags": False,
            "multivariate_lags": False,
            "multivariate_realtime": True,
            "self_prediction": False,
            "radiational": True,
            "symmetrical": False,
            "path": None,
            "sample_rate": None,
            "prediction": False,
            "save": True,
            "fixed_location": False,
            "verbose": True,
            "force_recompute": False,
            # new toggles (can override init)
            "location_mode": None,
            "use_precomputed_inputs": None,
            "precomputed_cache_dir": None,
            "precomputed_dtype": None,
            "fixed_lat": None,
            "fixed_lon": None,
            "precomputed_base_freq": None,
            "precomputed_version": None,
            "allow_precompute_write": None,
            "input_config": None,
            "ephemeris": None,
        }
        inputs = {**defaults, **kwargs}

        if inputs["location_mode"] is not None:
            self.location_mode = str(inputs["location_mode"]).lower().strip()
        if inputs["use_precomputed_inputs"] is not None:
            self.use_precomputed_inputs = bool(inputs["use_precomputed_inputs"])
        if inputs["precomputed_cache_dir"] is not None:
            self.precomputed_cache_dir = inputs["precomputed_cache_dir"]
        if inputs["precomputed_dtype"] is not None:
            self.precomputed_dtype = str(inputs["precomputed_dtype"])
        if inputs["fixed_lat"] is not None:
            self.fixed_lat = float(inputs["fixed_lat"])
        if inputs["fixed_lon"] is not None:
            self.fixed_lon = float(inputs["fixed_lon"])
        if inputs["precomputed_base_freq"] is not None:
            self.precomputed_base_freq = str(inputs["precomputed_base_freq"])
        if inputs["precomputed_version"] is not None:
            self.precomputed_version = str(inputs["precomputed_version"])
        if inputs["allow_precompute_write"] is not None:
            self.allow_precompute_write = bool(inputs["allow_precompute_write"])
        if inputs["input_config"] is not None:
            self.input_config = dict(inputs["input_config"])
        if inputs["ephemeris"] is not None:
            self.ephemeris = str(inputs["ephemeris"])
        if inputs.get("fixed_location", False):
            self.location_mode = "fixed"
            self.fixed_lat = 45.0
            self.fixed_lon = 0.0
            self.use_precomputed_inputs = True
            self.precomputed_cache_dir = os.path.expanduser("~/.cache/rtide")
        

        try:
            # Local variables for backward compatibility with existing function logic
            uniform_lags = self.uniform_lags
            multivariate_lags = self.multivariate_lags
            self_prediction = self.self_prediction
            radiational = self.radiational
            symmetrical = self.symmetrical
            path = self.path
        except:
            self.uniform_lags = inputs["uniform_lags"]
            self.multivariate_lags = inputs["multivariate_lags"]
            self.self_prediction = inputs["self_prediction"]
            self.radiational = inputs["radiational"]
            self.symmetrical = inputs["symmetrical"]
            self.path = inputs["path"]
            self.multivariate_realtime = inputs["multivariate_realtime"]

            uniform_lags = self.uniform_lags
            multivariate_lags = self.multivariate_lags
            self_prediction = self.self_prediction
            radiational = self.radiational
            symmetrical = self.symmetrical
            path = self.path
            
        verbose = bool(inputs.get("verbose", True))
        force_recompute = bool(inputs.get("force_recompute", False))

        prediction = inputs["prediction"]
        save = inputs["save"]

        # Determine sample_rate
        if inputs["sample_rate"] is not None:
            self.sample_rate = float(inputs["sample_rate"])
        else:
            dt_hours = _median_dt_seconds(self.ts.index) / 3600.0
            if dt_hours <= 0:
                raise ValueError("Could not infer sample rate from index.")
            self.sample_rate = 1.0 / dt_hours

        # Save path for legacy csv save/load
        save_directory = "./rtide_saves/"
        os.makedirs(save_directory, exist_ok=True)

        # Ensure self.path becomes a full path that does NOT double-prefix save_directory
        if path is not None:
            # if user passed an already-prefixed path, accept it as-is
            if isinstance(path, str) and path.startswith(save_directory):
                self.path = path
            else:
                # join avoids accidental double slashes and is clearer
                self.path = os.path.join(save_directory, path)
        else:
            self.path = os.path.join(save_directory, "RTide")

        # Compute lags + padded index
        self.Prep(uniform_lags, symmetrical, multivariate_lags)
        self.Compute_Time_Lags()

        # Base forcing features (unlagged) computed on padded_ts.index
        pad_index = self.padded_ts.index

        # Legacy csv load/save applies only to station-mode and not prediction mode
        loaded = False
        if (self.location_mode == "station") and save and (not prediction) and (not force_recompute):
            try:
                csv_path = f"{self.path}_global_tide.csv"
                pkl_path = f"{self.path}_inputs.pkl"

                prepped_dfs = pd.read_csv(csv_path, index_col=0)
                prepped_dfs.index = pd.to_datetime(prepped_dfs.index)
                inputs_used = load_inputs_from_pickle(self.path)

                if inputs_used != inputs:
                    raise ValueError("Inputs changed; recomputing.")

                self.prepped_dfs = prepped_dfs.reindex(self.ts.index)
                loaded = True

                if verbose:
                    print(
                        "[RTide] Using cached input features from previous run.\n"
                        f"         CSV : {csv_path}\n"
                        f"         PKL : {pkl_path}\n"
                        "         NOTE: Cached inputs reused. "
                        "Set force_recompute=True to override."
                    )

            except Exception:
                loaded = False


        if not loaded:
            # Build base features
            if (self.location_mode == "fixed") and self.use_precomputed_inputs:
                base6 = self._get_or_build_precomputed_base(pad_index.min(), pad_index.max())
                base_df = self._interpolate_to_index(base6, pad_index)
            else:
                base_df = self._compute_global_tide_base(index=pad_index)

            # Optionally drop radiational
            if not radiational:
                base_df = base_df[[c for c in base_df.columns if not c.startswith("Radiational_")]]

            self.global_tide_base = base_df

            # Lagging forcing features onto original ts index
            ts_to_concat = [self.ts[self.output_columns].copy()]

            # Add exogenous inputs (optionally lagged)
            # Add exogenous inputs (optionally lagged)
            if self.exog_columns:
                for col in self.exog_columns:
                    if col not in self.ts.columns:
                        continue

                    # Default / legacy: only realtime exogenous inputs unless user explicitly requests lagging
                    if not multivariate_lags:
                        ts_to_concat.append(self.ts[[col]])
                        continue

                    # If lagging requested, use the lag list computed by Prep()
                    for lagh in self.multivariate_lags:
                        shifted = self.ts[col].reindex(self.ts.index + pd.Timedelta(hours=lagh))
                        ts_to_concat.append(pd.DataFrame({f"{col}_{lagh}": shifted.to_numpy()}, index=self.ts.index))

            # Self-prediction (lag observations/u/v as inputs)
            if self_prediction:
    		# Expect self_prediction to be a list of lags
                if not isinstance(self_prediction, (list, tuple, np.ndarray)):
                    raise ValueError(
                        "self_prediction must be False or a list of lag hours "
                    f"(got {self_prediction!r})"
                    )

                for out_col in self.output_columns:
                        for lagh in self_prediction:
                            if lagh == 0:
                                    continue
                            shifted = self.ts[out_col].reindex(self.ts.index + pd.Timedelta(hours=lagh))
                            ts_to_concat.append(
                                    pd.DataFrame({f"{out_col}_{lagh}": shifted.to_numpy()}, index=self.ts.index))
            # Forcing lags
            for col in base_df.columns:
                series = base_df[col]
                for lagh in self.nonunilags:
                    shifted = series.reindex(self.ts.index + pd.Timedelta(hours=lagh))
                    ts_to_concat.append(pd.DataFrame({f"{col}_{lagh}": shifted.to_numpy()}, index=self.ts.index))

            prepped = pd.concat(ts_to_concat, axis=1)

            if prediction:
                self.prediction_dfs = prepped
                if save and (self.location_mode == "station"):
                    prepped.to_csv(f"{self.path}_global_tide_prediction.csv")
            else:
                self.prepped_dfs = prepped
                if save and (self.location_mode == "station"):
                    self.prepped_dfs.to_csv(f"{self.path}_global_tide.csv")
                    save_inputs_to_pickle(inputs, self.path)
                if save:
                    save_inputs_to_pickle(inputs, self.path)  # ✅ Always saved!

    # ----------------------------
    # Scaling helpers (legacy-safe)
    # ----------------------------
    def _fit_scale_X(self, X: np.ndarray, featurewise: bool) -> np.ndarray:
        self.scaler_X = StandardScaler()
        self.featurewise_X_scaling = bool(featurewise)
        if self.featurewise_X_scaling:
            return self.scaler_X.fit_transform(X)
        return self.scaler_X.fit_transform(X.reshape(-1, 1)).reshape(X.shape)

    def _transform_X(self, X: np.ndarray, featurewise: bool) -> np.ndarray:
        if self.scaler_X is None:
            raise RuntimeError("X scaler not fitted. Train first.")
        trained_mode = getattr(self, "featurewise_X_scaling", False)
        if bool(featurewise) != bool(trained_mode):
            raise ValueError(
                f"featurewise_X_scaling mismatch: trained={trained_mode}, predict={featurewise}"
            )
        if trained_mode:
            return self.scaler_X.transform(X)
        return self.scaler_X.transform(X.reshape(-1, 1)).reshape(X.shape)

    def _compute_normalized_time(self, index: pd.DatetimeIndex, 
                                  train_start: pd.Timestamp = None,
                                  train_end: pd.Timestamp = None) -> np.ndarray:
        """Compute normalized time values for trend estimation.
        
        Parameters
        ----------
        index : pd.DatetimeIndex
            Time index to normalize.
        train_start : pd.Timestamp, optional
            Start of training period. If None, uses min(index).
        train_end : pd.Timestamp, optional
            End of training period. If None, uses max(index).
            
        Returns
        -------
        normalized_time : np.ndarray
            Time values normalized to [0, 1] over the training period.
            Values outside training period are extrapolated proportionally.
        """
        if train_start is None:
            train_start = index.min()
        if train_end is None:
            train_end = index.max()
        
        # Convert to numeric (seconds since epoch)
        time_numeric = index.astype(np.int64) / 1e9
        train_start_numeric = train_start.value / 1e9
        train_end_numeric = train_end.value / 1e9
        
        # Normalize to [0, 1] over training period
        time_range = train_end_numeric - train_start_numeric
        if time_range <= 0:
            return np.zeros(len(index))
        
        normalized = (time_numeric - train_start_numeric) / time_range
        return normalized.to_numpy()

    # ----------------------------
    # Train / Predict
    # ----------------------------
    def Train(self, **kwargs):
        """
        Function to train RTide model using a standard neural network.

        Inputs:
        --------
        kwargs: dictionary
            A dictionary of parameters with the following keys:
                - lr: float
                    Learning rate for the training process.

                - loss: str
                    Loss function to be used ('MAE' or any other loss function).

                - train_epochs: int
                    Number of training epochs for the initial training phase.

                - train_epochs2: int
                    Number of training epochs for the second training phase.

                - verbose: bool
                    Option to print training progress and details.

                - regularization_strength: float
                    Strength of the regularization term.

                - hidden_nodes: str or int
                    Configuration for hidden nodes ('standard' or a specific number).

                - depth: int
                    Depth of the neural network (number of hidden layers).

                - early_stoppage: bool
                    Option to enable early stopping during training.

                - save_weights: bool
                    Option to save the weights of the trained model.
                    
                - trend: str or None
                    Trend estimation type: None (default, no trend), 'linear', or 'quadratic'.
                    When specified, the model will simultaneously estimate both tidal dynamics
                    and a polynomial trend during training.

        Outputs:
        --------
        self.model: Model
            Trained neural network model.

        self.train_predictions:  dictionary
            A dictionary of training predictions with the following keys:
                - rtide_train: np.array
                    Array containing the predictions for the RTide model.

                - train_observations: np.array
                    Array containing the actual observations.

                - utide_tide_train: np.array
                    Array containing standard UTide OLS predictions.

                - rtide_tide_train: np.array
                    Array containing RTide harmonic equivalent predictions.

                - irls_utide_train: np.array
                    Array containing UTide IRLS predictions.
        
        self.train_prediction_df: DataFrame
            DataFrame containing the hindcasted training predictions alongside 
            input observations.

        """

        import warnings

        # Default values
        defaults = {
            'lr': 0.0001,
            'loss': 'MAE',
            'architecture': 'response',
            'siren_w0': 30.0,
            # Legacy arg kept for compatibility. It is no longer used.
            'linear_epochs': 0,
            'standard_epochs': 500,
            'verbose': True,
            'regularization_strength': 0.0,
            'hidden_nodes': 'standard',
            'depth': 3,
            'early_stoppage': False,
            'save_weights': True,
            'trend': None,  # NEW: trend estimation
        }

        # Update defaults with kwargs
        inputs = {**defaults, **kwargs}

        lr = inputs['lr']
        loss = inputs['loss']
        architecture = inputs.get('architecture', 'standard')
        self.architecture = architecture
        siren_w0 = inputs.get('siren_w0', 30.0)
        train_epochs2 = inputs['standard_epochs']
        verbose = inputs['verbose']
        regularization_strength = inputs['regularization_strength']
        hidden_nodes = inputs['hidden_nodes']
        depth = inputs['depth']
        early_stoppage = inputs['early_stoppage']
        save_weights = inputs['save_weights']
        featurewise_X_scaling = inputs.get('featurewise_scaling', False)
        trend = inputs.get('trend', None)  # NEW
        
        # Store trend type for use in Predict
        self.trend = trend

        if inputs.get('linear_epochs', 0) not in (0, None):
            warnings.warn(
                "linear_epochs is deprecated and is ignored (legacy linear/UTide warm-start training was removed).",
                DeprecationWarning,
                stacklevel=2,
            )

        if verbose:
            print('#### Model Overview ####')
            print('Mode:', self.output_mode)
            print('Outputs:', self.output_columns)
            print('Learning Rate:', lr)
            print('Loss:', loss)
            print('Standard Epochs:', train_epochs2)
            print('Regularization:', regularization_strength)
            print('Number of Layers:', depth)
            print('Has Exogenous Inputs (multi):', self.multi)
            print('Save Weights:', save_weights)
            if trend is not None:
                print('Trend Estimation:', trend)

        df = self.prepped_dfs.dropna()
        dataset = df.values
        num_cols = dataset.shape[1]
        n_outputs = self.n_outputs

        # X/Y split
        train_X = dataset[:, n_outputs:num_cols]
        train_Y = dataset[:, 0:n_outputs]

        # Scale
        scaled_train_X = self._fit_scale_X(train_X, featurewise=featurewise_X_scaling)
        scaled_train_X = self._transform_X(train_X, featurewise=featurewise_X_scaling)

        self.scaler_Y = StandardScaler()
        scaled_train_Y = self.scaler_Y.fit_transform(train_Y)

        # Compute normalized time for trend estimation
        if trend is not None:
            train_time = self._compute_normalized_time(df.index)
            self.train_time_start = df.index.min()
            self.train_time_end = df.index.max()
        else:
            train_time = None

        input_dims = num_cols - n_outputs
        if hidden_nodes == 'standard':
            hidden_nodes = input_dims

        if not early_stoppage:
            early_stoppage = train_epochs2
        custom_early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=early_stoppage,
        min_delta=0.0001,
        mode='min'
        )

        if early_stoppage < 10:
            reduce_lr_patience = early_stoppage
        elif early_stoppage == train_epochs2:
            reduce_lr_patience = 30
        else:
            reduce_lr_patience = early_stoppage - 10

        reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=reduce_lr_patience,
        min_lr=0.00000001,
        )

        # Regularization
        l2_strength = regularization_strength

        if trend is not None:
            train_time = self._compute_normalized_time(df.index)
            self.train_time_start = df.index.min()
            self.train_time_end = df.index.max()

            # NEW: initialize trend from a quick least-squares fit (works for 1D and 2D outputs)
            trend_initial_coeffs = fit_trend_initial_coeffs(
                t_norm=train_time,
                y_scaled=scaled_train_Y,
                trend=trend,
            )
        else:
            train_time = None
            trend_initial_coeffs = None

  
        # Model
        model = models.build_model(
        architecture=architecture,
        input_dims=input_dims,
        n_outputs=n_outputs,
        hidden_nodes=hidden_nodes,
        depth=depth,
        l1_strength=0.0,
        l2_strength=l2_strength,
        siren_w0=siren_w0,
        trend=trend, 
        trend_initial_coeffs = trend_initial_coeffs
        )

        if loss == 'SSP':
            model.compile(loss=compute_ssp, optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
        else:
            model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=lr))

        # Shuffle BEFORE validation split.
        if trend is None:
            # Legacy single-input model
            scaled_train_X_shuffled, scaled_train_Y_shuffled = shuffle(
                scaled_train_X, scaled_train_Y, random_state=0
            )

            history2 = model.fit(
                scaled_train_X_shuffled,
                scaled_train_Y_shuffled,
                epochs=train_epochs2,
                batch_size=32,
                verbose=2,
                shuffle=True,
                validation_split=0.15,
                callbacks=[custom_early_stopping, reduce_lr],
            )
        else:
            # Dual-input model with trend
            scaled_train_X_shuffled, train_time_shuffled, scaled_train_Y_shuffled = shuffle(
                scaled_train_X, train_time.reshape(-1, 1), scaled_train_Y, random_state=0
            )

            history2 = model.fit(
                [scaled_train_X_shuffled, train_time_shuffled],
                scaled_train_Y_shuffled,
                epochs=train_epochs2,
                batch_size=32,
                verbose=2,
                shuffle=True,
                validation_split=0.15,
                callbacks=[custom_early_stopping, reduce_lr],
            )

        if train_epochs2 > 0:
            plt.plot(history2.history['loss'], label='Loss')
            plt.plot(history2.history['val_loss'], label='Val Loss')
            plt.legend()
            plt.title('Standard Training')
            plt.show()

        # Train predictions
        if trend is None:
            train_predictions = self.scaler_Y.inverse_transform(model.predict(scaled_train_X))
        else:
            train_predictions = self.scaler_Y.inverse_transform(
                model.predict([scaled_train_X, train_time.reshape(-1, 1)])
            )
        
        train_labels = dataset[:, 0:n_outputs]

        if n_outputs == 1:
            preds_train = train_predictions.reshape(-1)
            labels_train = train_labels.reshape(-1)
        else:
            preds_train = train_predictions
            labels_train = train_labels

        # Pure-tide predictions by zeroing exogenous inputs (if present).
        if self.multi:
            n_exog = len(self.exog_columns)
            # if multivariate_lags == [0] then no lagged exog columns exist; keep logic but still compute rtide_noforcing
            n_lagged_exog = 0 if self.multivariate_lags == [0] else n_exog * len(self.multivariate_lags)

            # Build X_tide by copying train_X and zeroing exogenous columns
            X_tide = train_X.copy()

            # Zero realtime exogenous columns if present:
            if n_exog > 0:
                X_tide[:, 0:n_exog] = 0

            # If there are lagged exogenous columns, zero those too by targeting the tail slice
            if n_lagged_exog > 0:
                X_tide[:, -n_lagged_exog:] = 0

            # Always scale / predict the tide-only input matrix (handles both realtime-only and lagged cases)
            scaled_X_tide = self._transform_X(X_tide, featurewise=featurewise_X_scaling)
            if trend is None:
                rtide_noforcing = self.scaler_Y.inverse_transform(model.predict(scaled_X_tide))
            else:
                rtide_noforcing = self.scaler_Y.inverse_transform(
                    model.predict([scaled_X_tide, train_time.reshape(-1, 1)])
                )
            rtide_noforcing = rtide_noforcing.reshape(-1) if n_outputs == 1 else rtide_noforcing
        else:
            rtide_noforcing = preds_train.copy()
            
        # UTide baselines / harmonic equivalents.
        try:
            if trend is not None:
                utide_trend = True
            else:
                utide_trend = False
            if self.output_mode == 'elevation':
                short_utide = utide.solve(
                df.index,
                df['observations'],
                lat=self.lat,
                method='ols',
                conf_int='none',
                verbose=False,
                trend=utide_trend,
                nodal=True,
                )
                utide_train_pred = utide.reconstruct(df.index, short_utide, verbose=False)
                utide_train = utide_train_pred['h']

                rtide_utide = utide.solve(
                df.index,
                rtide_noforcing,
                lat=self.lat,
                method='ols',
                conf_int='none',
                verbose=False,
                trend=utide_trend,
                nodal=True,
                )

                irls_utide = utide.solve(
                df.index,
                df['observations'],
                lat=self.lat,
                method='robust',
                conf_int='none',
                verbose=False,
                trend=utide_trend,
                nodal=True,
                )

                rtide_utide_reconstruction = utide.reconstruct(df.index, rtide_utide, verbose=False)
                rtide_tide = rtide_utide_reconstruction['h']
                irls_utide_reconstruction = utide.reconstruct(df.index, irls_utide, verbose=False)
                irls_tide = irls_utide_reconstruction['h']

            else:
                # Currents: UTide supports u/v directly.
                short_utide = utide.solve(
                df.index,
                df['u'],
                df['v'],
                lat=self.lat,
                method='ols',
                conf_int='none',
                verbose=False,
                trend=utide_trend,
                nodal=True,
                )
                utide_train_pred = utide.reconstruct(df.index, short_utide, verbose=False)
                utide_train = np.column_stack([utide_train_pred['u'], utide_train_pred['v']])

                rt_u = rtide_noforcing[:, 0] if isinstance(rtide_noforcing, np.ndarray) and rtide_noforcing.ndim == 2 else rtide_noforcing
                rt_v = rtide_noforcing[:, 1] if isinstance(rtide_noforcing, np.ndarray) and rtide_noforcing.ndim == 2 else None

                rtide_utide = utide.solve(
                df.index,
                rt_u,
                rt_v,
                lat=self.lat,
                method='ols',
                conf_int='none',
                verbose=False,
                trend=utide_trend,
                nodal=True,
                )

                irls_utide = utide.solve(
                df.index,
                df['u'],
                df['v'],
                lat=self.lat,
                method='robust',
                conf_int='none',
                verbose=False,
                trend=utide_trend,
                nodal=True,
                )

                rtide_utide_reconstruction = utide.reconstruct(df.index, rtide_utide, verbose=False)
                rtide_tide = np.column_stack([rtide_utide_reconstruction['u'], rtide_utide_reconstruction['v']])
                irls_utide_reconstruction = utide.reconstruct(df.index, irls_utide, verbose=False)
                irls_tide = np.column_stack([irls_utide_reconstruction['u'], irls_utide_reconstruction['v']])

            self.rtide_ha = rtide_utide
            self.utide_ha = short_utide
            self.utide_irls = irls_utide

        except Exception as e:
            print(f"UTide error, this is not a problem with RTide: {e}")
            utide_train = []
            rtide_tide = []
            irls_tide = []

        self.model = model
        
        
        if save_weights:
            model_dir = os.path.dirname(f'{self.path}_model_weights.keras')
            if model_dir:
                os.makedirs(model_dir, exist_ok=True)
            model.save(f'{self.path}_model_weights.keras')
            joblib.dump(self.scaler_X, f'{self.path}_scaler_X.save')
            joblib.dump(self.scaler_Y, f'{self.path}_scaler_Y.save')
            
            # Save metadata including trend configuration
            meta = {
                'featurewise_X_scaling': featurewise_X_scaling,
                'n_outputs': n_outputs,
                'output_columns': self.output_columns,
                'trend': trend,
            }
            if trend is not None:
                meta['train_time_start'] = str(self.train_time_start)
                meta['train_time_end'] = str(self.train_time_end)
            
            with open(f'{self.path}_meta.json', 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2)

        # Predictions dict (keep existing keys; values become Nx2 arrays for currents)
        self.train_predictions = {
        'rtide_train': preds_train,
        'train_observations': labels_train,
        'utide_tide_train': utide_train,
        'rtide_tide_train': rtide_tide,
        'irls_utide_train': irls_tide,
        'RTide_nomulti': rtide_noforcing,
        }
        self.model_predictions = self.train_predictions

        # Convenience prediction dataframe
        if self.output_mode == 'elevation':
            self.train_prediction_df = pd.DataFrame(
                {'observations': df['observations'].to_numpy(), 'rtide': preds_train},
                index=df.index,
            )
        else:
            self.train_prediction_df = pd.DataFrame(
                {
                'u': df['u'].to_numpy(),
                'v': df['v'].to_numpy(),
                'rtide_u': preds_train[:, 0],
                'rtide_v': preds_train[:, 1],
                },
                index=df.index,
            )

    def Load_Model(self, path: Optional[str] = None):
        """
        Load saved keras model + scalers.
        """
        if path is not None:
            self.path = path
        self.model = tf.keras.models.load_model(
            f"{self.path}_model_weights.keras",
            custom_objects=get_custom_objects(),
        )
        try:
            self.scaler_X = joblib.load(f"{self.path}_scaler_X.save")
            self.scaler_Y = joblib.load(f"{self.path}_scaler_Y.save")
            meta_path = f"{self.path}_meta.json"
            if os.path.exists(meta_path):
                meta = json.loads(open(meta_path, "r", encoding="utf-8").read())
                self.featurewise_X_scaling = bool(meta.get("featurewise_X_scaling", False))
                self.n_outputs = int(meta.get("n_outputs", self.n_outputs))
                self.output_columns = list(meta.get("output_columns", self.output_columns))
                # Load trend configuration
                self.trend = meta.get("trend", None)
                if self.trend is not None:
                    self.train_time_start = pd.Timestamp(meta.get("train_time_start"))
                    self.train_time_end = pd.Timestamp(meta.get("train_time_end"))
        except Exception:
            pass
        return self
    def Predict(self, df, featurewise_X_scaling = None):
        """
        Function to generate predictions using the learned model at associated times.

        Inputs:
        --------
        df: dataframe
            Dataframe with datetime index at the desired timestamps and observations column.
            Observations should contain NaNs, or the actualy observations.
            If concurrent observations are included, RTide will automatically compute summary statistics.
            **Note if additional multivariate input functions were included, they must be passed as an additional column.

        Returns:
        --------
        self.test_predictions:  dictionary
            A dictionary of test predictions with the following keys:
                - rtide_test: np.array
                    Array containing the predictions for the RTide model.

                - testobservations: np.array
                    Array containing the actual observations (NaNs if not provided).

                - utide_tide_test np.array
                    Array containing standard UTide OLS predictions.

                - rtide_tide_test: np.array
                    Array containing RTide harmonic equivalent predictions.

                - irls_utide_test: np.array
                    Array containing UTide IRLS predictions.
        """
        # Update schema for the prediction dataframe and verify it matches the model.
        previous_mode = getattr(self, 'output_mode', None)
        self.ts = df
        self._infer_io_schema()
        self._validate_inputs()
        if previous_mode is not None and self.output_mode != previous_mode:
            raise ValueError(
                f"Prediction dataframe schema ({self.output_mode}) does not match model schema ({previous_mode})."
            )

        try:
            model = self.model
        except Exception:
            try:
                custom_objects = models.get_custom_objects()
                custom_objects['compute_ssp'] = compute_ssp
                model = self.model = tf.keras.models.load_model(
                f'{self.path}_model_weights.keras',
                custom_objects=custom_objects,
                compile=False,
                )
                self.scaler_X = joblib.load(f'{self.path}_scaler_X.save')
                self.scaler_Y = joblib.load(f'{self.path}_scaler_Y.save')
            except Exception:
                raise print("No model has been trained or has been saved.")


        # Reuse the exact Prepare_Inputs configuration used during training
        try:
            train_inputs = load_inputs_from_pickle(self.path)
        except Exception:
            raise RuntimeError(
                "Could not load training Prepare_Inputs configuration. "
                "Prediction requires the same feature configuration as training."
            )

        # Force prediction mode, do not overwrite saved files
        train_inputs["prediction"] = True
        train_inputs["save"] = False

        self.Prepare_Inputs(**train_inputs)


        dfp = self.prediction_dfs.dropna()
        
        dataset = dfp.values
        num_cols = dataset.shape[1]
        n_outputs = self.n_outputs
        if featurewise_X_scaling is None:
            featurewise_X_scaling = getattr(self, "featurewise_X_scaling", False)

        test_X = dataset[:, n_outputs:num_cols]
        scaled_test_X = self._transform_X(test_X, featurewise=featurewise_X_scaling)

        # Check if model uses trend estimation
        trend = getattr(self, 'trend', None)
        
        if trend is None:
            # Legacy single-input model
            rtide_test = self.scaler_Y.inverse_transform(model.predict(scaled_test_X))
        else:
            # Dual-input model with trend
            # Compute normalized time for prediction period relative to training period
            test_time = self._compute_normalized_time(
                dfp.index,
                train_start=self.train_time_start,
                train_end=self.train_time_end
            )
            rtide_test = self.scaler_Y.inverse_transform(
                model.predict([scaled_test_X, test_time.reshape(-1, 1)])
            )
	
        # UTide reconstructions (if available)
        try:
            ols_utide_reconstruction = utide.reconstruct(dfp.index, self.utide_ha, verbose=False)
            rtide_utide_reconstruction = utide.reconstruct(dfp.index, self.rtide_ha, verbose=False)
            irls_utide_reconstruction = utide.reconstruct(dfp.index, self.utide_irls, verbose=False)

            if self.output_mode == 'elevation':
                ols_utide = ols_utide_reconstruction['h']
                rtide_tide = rtide_utide_reconstruction['h']
                irls_tide = irls_utide_reconstruction['h']
            else:
                ols_utide = np.column_stack([ols_utide_reconstruction['u'], ols_utide_reconstruction['v']])
                rtide_tide = np.column_stack([rtide_utide_reconstruction['u'], rtide_utide_reconstruction['v']])
                irls_tide = np.column_stack([irls_utide_reconstruction['u'], irls_utide_reconstruction['v']])
        except Exception:
            ols_utide = []
            rtide_tide = []
            irls_tide = []

        # Observations may be NaN for pure forecasting.
        obs_block = dataset[:, 0:n_outputs]
        has_nans = np.any(np.isnan(obs_block))
        test_observations = [] if has_nans else obs_block

        self.predict_indexes = dfp.index
        if n_outputs == 1:
            self.test_predictions = {
                'rtide_test': rtide_test.reshape(-1),
                'test_observations': np.array(test_observations).reshape(-1),
                'utide_tide_test': ols_utide,
                'rtide_tide_test': rtide_tide,
                'irls_utide_test': irls_tide,
            }
            self.test_prediction_df = pd.DataFrame(
                {'observations': dfp['observations'].to_numpy(), 'rtide': rtide_test.reshape(-1)},
                index=dfp.index,
            )
        else:
            self.test_predictions = {
                'rtide_test': rtide_test,
                'test_observations': test_observations,
                'utide_tide_test': ols_utide,
                'rtide_tide_test': rtide_tide,
                'irls_utide_test': irls_tide,
                'rtide_test_u': rtide_test[:, 0],
                'rtide_test_v': rtide_test[:, 1],
            }
            self.test_prediction_df = pd.DataFrame(
                {
                'u': dfp['u'].to_numpy(),
                'v': dfp['v'].to_numpy(),
                'rtide_u': rtide_test[:, 0],
                'rtide_v': rtide_test[:, 1],
                },
                index=dfp.index,
            )

    # ----------------------------
    # Visualization helpers (minimal)
    # ----------------------------
    def Visualize_Residuals(self, savefig = False, returnfig = False, tides = False):
        """
            Function to visualize the residuals of the model.

            Inputs:
            -------
            savefig: False or str
                Option to save the figure. if not False, user must specify a file name.
            returnfig: bool
                Option to return the figure without plotting so the user can edit it.

            Returns:
            ---------
            Optionally returns figure object if return == True.

        """
        fig = plt.figure(figsize=(12, 6), dpi = 200)  # Adjust the width and height as needed
        try:
            train_predictions = self.train_predictions
            ### PLOTTING TRAINING PREDICTIONS
            plt.subplot(1, 2, 1)
            train_predictions = self.train_predictions
            indexs = self.prepped_dfs.dropna().index

            rtide_residuals = np.array(train_predictions['rtide_train']) - np.array(train_predictions['train_observations'])
            if self.n_outputs == 1:
                plt.plot(indexs, rtide_residuals, label=fr'RTide: $\mu$ = {round(np.average(rtide_residuals),4)}, $\sigma$ = {round(np.std(rtide_residuals),4)}', color='red')
            else:
                plt.plot(indexs, rtide_residuals[:, 0], label=fr'RTide u: $\mu$ = {round(np.average(rtide_residuals[:,0]),4)}, $\sigma$ = {round(np.std(rtide_residuals[:,0]),4)}', color='red')
                plt.plot(indexs, rtide_residuals[:, 1], linestyle='--', label=fr'RTide v: $\mu$ = {round(np.average(rtide_residuals[:,1]),4)}, $\sigma$ = {round(np.std(rtide_residuals[:,1]),4)}', color='red')
            if tides:
                if len(train_predictions['utide_tide_train']) > 0:
                    rtide_tide_residuals = np.array(train_predictions['rtide_tide_train']) - np.array(train_predictions['train_observations'])
                    utide_residuals = np.array(train_predictions['utide_tide_train']) - np.array(train_predictions['train_observations'])
                if self.n_outputs == 1:
                    plt.plot(indexs, rtide_tide_residuals, label=fr'RTide Tide: $\mu$ = {round(np.average(rtide_tide_residuals),4)}, $\sigma$ = {round(np.std(rtide_tide_residuals),4)}', color='green')
                    plt.plot(indexs, utide_residuals, label=fr'UTide: $\mu$ = {round(np.average(utide_residuals),4)}, $\sigma$ = {round(np.std(utide_residuals),4)}', color='blue')
                else:
                    plt.plot(indexs, rtide_tide_residuals[:, 0], label='RTide Tide u', color='green')
                    plt.plot(indexs, rtide_tide_residuals[:, 1], linestyle='--', label='RTide Tide v', color='green')
                    plt.plot(indexs, utide_residuals[:, 0], label='UTide u', color='blue')
                    plt.plot(indexs, utide_residuals[:, 1], linestyle='--', label='UTide v', color='blue')

            plt.xlim(indexs[0], indexs[-1])
            plt.legend()
            plt.gcf().autofmt_xdate()
            plt.ylabel("Residual (units)")
            plt.title('Train Prediction Residuals')
        except:
            print("No training data available to plot. Retrain model if desired.")

        try:
            test_predictions = self.test_predictions
        except:
            test_predictions = None


        ### PLOTTING TEST PREDICTIONS
        if test_predictions is not None and np.size(test_predictions.get('test_observations', [])) > 0:
            plt.subplot(1, 2, 2)

            indexs = self.predict_indexes

            rtide_residuals = np.array(test_predictions['rtide_test']) - np.array(test_predictions['test_observations'])
            if self.n_outputs == 1:
                plt.plot(indexs, rtide_residuals, label=fr'RTide: $\mu$ = {round(np.average(rtide_residuals),4)}, $\sigma$ = {round(np.std(rtide_residuals),4)}', color='red')
            else:
                plt.plot(indexs, rtide_residuals[:, 0], label=fr'RTide u: $\mu$ = {round(np.average(rtide_residuals[:,0]),4)}, $\sigma$ = {round(np.std(rtide_residuals[:,0]),4)}', color='red')
                plt.plot(indexs, rtide_residuals[:, 1], linestyle='--', label=fr'RTide v: $\mu$ = {round(np.average(rtide_residuals[:,1]),4)}, $\sigma$ = {round(np.std(rtide_residuals[:,1]),4)}', color='red')
            if tides:
                if len(test_predictions['utide_tide_test']) > 0:
                    rtide_tide_residuals = np.array(test_predictions['rtide_tide_test']) - np.array(test_predictions['test_observations'])
                    utide_residuals = np.array(test_predictions['utide_tide_test']) - np.array(test_predictions['test_observations'])
                    if self.n_outputs == 1:
                        plt.plot(indexs, rtide_tide_residuals, label=fr'RTide Tide: $\mu$ = {round(np.average(rtide_tide_residuals),4)}, $\sigma$ = {round(np.std(rtide_tide_residuals),4)}', color='green')
                        plt.plot(indexs, utide_residuals, label=fr'UTide: $\mu$ = {round(np.average(utide_residuals),4)}, $\sigma$ = {round(np.std(utide_residuals),4)}', color='blue')
                    else:
                        plt.plot(indexs, rtide_tide_residuals[:, 0], label='RTide Tide u', color='green')
                        plt.plot(indexs, rtide_tide_residuals[:, 1], linestyle='--', label='RTide Tide v', color='green')
                        plt.plot(indexs, utide_residuals[:, 0], label='UTide u', color='blue')
                        plt.plot(indexs, utide_residuals[:, 1], linestyle='--', label='UTide v', color='blue')


            plt.xlim(indexs[0], indexs[-1])
            plt.legend()
            plt.ylabel("Residual (units)")
            plt.gcf().autofmt_xdate()

            plt.title('Test Prediction Residuals')

        if savefig:
            plt.savefig(f'{savefig}.png', dpi =300)

        if returnfig:
            return fig
        else:
            plt.show()
    
    def Visualize_Predictions(self, savefig = False, returnfig = False, verbose = True):
        """
        Helper function to visualize the train/test predictions of the model.

        Function will automatically plot whatever data is available. e.g. if you have loaded
        a model weights and only have test predictions, on the test predictions will display.

        Inputs:
        --------
        savefig: False or str
            Option to save the figure. if not False, user must specify a file name.
        returnfig: bool
            Option to return the figure without plotting so the user can edit it.
        verbose: bool
            Option to print out summary statistics of the model. (R^2, RMSE, MAE, MAPE)

        Returns:
        --------  
        Optionally returns figure object if return == True.
        """
        fig = plt.figure(figsize=(18, 9), dpi=200)

        # Train
        try:
            train_predictions = self.train_predictions
        except Exception:
            train_predictions = None

        if train_predictions is not None:
            indexs = self.prepped_dfs.dropna().index
            plt.subplot(2, 1, 1)

            obs = np.array(train_predictions['train_observations'])
            pred = np.array(train_predictions['rtide_train'])

            if self.n_outputs == 1:
                plt.plot(indexs, obs, color='k', label='Actual')
                plt.plot(indexs, pred, color='red', label='RTide')
                ylabel = 'Sea Level (units)'
                if verbose:
                    print('Train Results')
                    _ = calc_stats(pred, obs)
            else:
                plt.plot(indexs, obs[:, 0], color='k', label='Actual u')
                plt.plot(indexs, pred[:, 0], color='red', label='RTide u')
                plt.plot(indexs, obs[:, 1], color='k', linestyle='--', label='Actual v')
                plt.plot(indexs, pred[:, 1], color='red', linestyle='--', label='RTide v')
                ylabel = 'Current (units)'
                if verbose:
                    print('Train Results (u)')
                    _ = calc_stats(pred[:, 0], obs[:, 0])
                    print('Train Results (v)')
                    _ = calc_stats(pred[:, 1], obs[:, 1])

            plt.gcf().autofmt_xdate()
            plt.title('Train results')
            plt.ylabel(ylabel)
            plt.xlim(indexs[0], indexs[-1])
            plt.legend()

        # Test
        try:
            test_predictions = self.test_predictions
        except Exception:
            test_predictions = None

        if test_predictions is not None and np.size(test_predictions.get('test_observations', [])) > 0:
            indexs = self.prediction_dfs.dropna().index
            plt.subplot(2, 1, 2)

            obs = np.array(test_predictions['test_observations'])
            pred = np.array(test_predictions['rtide_test'])

            if self.n_outputs == 1:
                plt.plot(indexs, obs, color='k', label='Actual')
                plt.plot(indexs, pred, color='red', label='RTide')
                ylabel = 'Sea Level (units)'
                if verbose:
                    print('Test Results')
                    _ = calc_stats(pred, obs)
            else:
                plt.plot(indexs, obs[:, 0], color='k', label='Actual u')
                plt.plot(indexs, pred[:, 0], color='red', label='RTide u')
                plt.plot(indexs, obs[:, 1], color='k', linestyle='--', label='Actual v')
                plt.plot(indexs, pred[:, 1], color='red', linestyle='--', label='RTide v')
                ylabel = 'Current (units)'
                if verbose:
                    print('Test Results (u)')
                    _ = calc_stats(pred[:, 0], obs[:, 0])
                    print('Test Results (v)')
                    _ = calc_stats(pred[:, 1], obs[:, 1])

            plt.title('Test results')
            plt.ylabel(ylabel)
            plt.xlabel('Time')
            plt.xlim(indexs[0], indexs[-1])
            plt.gcf().autofmt_xdate()
            plt.legend()

        if savefig:
            plt.savefig(f'{savefig}.png', dpi=300)

        if returnfig:
            return fig
        plt.show()
        
    def Shap_Analysis(self, plot: bool = True, output_index: int = 0):
        """Compute (and optionally plot) SHAP values.
        
        For models with trend estimation, SHAP analysis is performed on the forcing
        features only, with time held constant at the median normalized time value.
        This allows interpretation of how forcing features affect predictions independent
        of the secular trend component.

        Parameters
        ----------
        plot : bool, optional
            Whether to plot SHAP dependence plots. Default is True.
        output_index : int, optional
            For multi-output models (currents u/v), which output to analyze. Default is 0.

        Notes
        -----
        * For multi-output models (currents u/v), SHAP can return a list of arrays
          (one per output) depending on SHAP version. This method handles both
          formats and defaults to plotting `output_index`.
        * For trend-aware models, only the forcing features are analyzed. Time is
          held constant at the median value to isolate the effect of forcing features
          from the secular trend.
        """
        try:
            _ = self.model
        except Exception:
            try:
                custom_objects = models.get_custom_objects()
                custom_objects['compute_ssp'] = compute_ssp
                self.model = tf.keras.models.load_model(
                f'{self.path}_model_weights.keras',
                custom_objects=custom_objects,
                compile=False,
                )
                self.scaler_X = joblib.load(f'{self.path}_scaler_X.save')
                self.scaler_Y = joblib.load(f'{self.path}_scaler_Y.save')
            except Exception:
                raise print("No model has been trained or has been saved.")

        # Use whichever prepared dataframe exists.
        if hasattr(self, 'prediction_dfs'):
            df = self.prediction_dfs.dropna()
        else:
            df = self.prepped_dfs.dropna()

        dataset = df.values
        num_features = dataset.shape[1]
        n_outputs = self.n_outputs

        test_X = dataset[:, n_outputs:num_features]
        scaled_test_X = self.scaler_X.transform(test_X.reshape(-1, 1)).reshape(test_X.shape)

        # Check if model uses trend estimation
        trend = getattr(self, 'trend', None)
        
        if trend is None:
            # Legacy single-input model
            def f(x):
                return self.scaler_Y.inverse_transform(self.model.predict(x))
            
            background = shap.sample(scaled_test_X, min(15, scaled_test_X.shape[0]))
            self.shap_explainer = shap.KernelExplainer(f, background)
            self.shap_values = self.shap_explainer.shap_values(scaled_test_X[:], nsamples=300)
            
        else:
            # Trend-aware dual-input model
            # Compute normalized time for the dataset
            time_values = self._compute_normalized_time(
                df.index,
                train_start=getattr(self, 'train_time_start', df.index.min()),
                train_end=getattr(self, 'train_time_end', df.index.max())
            )
            
            # Use median time as a representative fixed time point for SHAP analysis
            # This makes interpretation clearer: "how do forcing features affect predictions
            # at a typical time point, independent of the trend?"
            median_time = np.median(time_values)
            
            # Create wrapper function that provides both inputs
            # Time is held constant at the median, SHAP analyzes forcing features only
            def f(x):
                # x has shape (n_samples, n_features) - these are the forcing features
                n_samples = x.shape[0]
                
                # Create time array with median value repeated for all samples
                time_input = np.full((n_samples, 1), median_time)
                
                pred = self.model.predict([x, time_input], verbose=0)
                return self.scaler_Y.inverse_transform(pred)
            
            background = shap.sample(scaled_test_X, min(15, scaled_test_X.shape[0]))
            
            # Inform user about the analysis approach
            print(f"Computing SHAP values for forcing features (trend={trend})...")
            print(f"Note: Time is held constant at median value (t_norm={median_time:.3f}) for this analysis.")
            print("This shows how forcing features affect predictions independent of the secular trend.")
            
            self.shap_explainer = shap.KernelExplainer(f, background)
            self.shap_values = self.shap_explainer.shap_values(scaled_test_X[:], nsamples=300)

        column_names = df.columns[n_outputs:]

        if not plot:
            return

        # Select output slice.
        shap_vals = self.shap_values
        if isinstance(shap_vals, list):
            shap_for_output = shap_vals[output_index]
        elif getattr(shap_vals, 'ndim', 0) == 3:
            shap_for_output = shap_vals[:, :, output_index]
        else:
            shap_for_output = shap_vals

        n_exog = len(self.exog_columns)
        for ind in range(n_exog):
            shap.dependence_plot(
                ind,
                shap_for_output,
                scaled_test_X[:],
                show=True,
                feature_names=column_names,
                interaction_index=None,
            )