"""
Data handling and batching for the forecasting models.

The :class:`TimeSeriesDataModule` is responsible for reading raw CSV files,
performing pivot operations to obtain a multivariate time series, adding
calendar features via :func:`src.core.feature_engineer.add_date_features`
and applying optional lag/moving average transformations provided by a
:class:`src.core.feature_engineer.FeatureEngineer` instance.  It also
constructs sliding windows suitable for sequence‑to‑sequence models.

During training the module splits the data chronologically into
training, validation and test segments.  The default split is 70 % for
training, 15 % for validation and 15 % for testing.  These ratios can
be adjusted by subclassing this module or modifying the code.  At
inference time the :meth:`preprocess_inference_data` method can be
used to convert unseen test files into the same feature space as the
training data.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
try:
    import torch  # type: ignore
    from torch.utils.data import DataLoader, TensorDataset  # type: ignore
except ImportError:
    # torch is optional; only required for training.  When absent,
    # DataLoader and TensorDataset remain undefined and any attempt to
    # instantiate them will raise an error.
    torch = None  # type: ignore
    DataLoader = None  # type: ignore
    TensorDataset = None  # type: ignore
try:
    import pytorch_lightning as pl  # type: ignore
except ImportError:
    # Lightning is optional; data module can still be used outside of training.
    class DummyLightningDataModule:
        pass
    pl = type("pl", (), {"LightningDataModule": DummyLightningDataModule})  # type: ignore

from .feature_engineer import add_date_features, FeatureEngineer


class TimeSeriesDataModule(pl.LightningDataModule):
    """Load, transform and batch multivariate time series data.

    Parameters
    ----------
    file_path : str
        Path to the CSV containing the training data.  The CSV must
        include at least the columns ``영업일자``, ``영업장명_메뉴명`` and
        ``매출수량``.
    sequence_length : int
        Length of the encoder input sequence fed to the model.
    forecast_horizon : int
        Number of future timesteps to predict.
    label_len : int
        Length of the decoder context provided during training.  The
        decoder input is composed of ``label_len`` timesteps from the
        end of the encoder sequence followed by ``forecast_horizon``
        zeros when using the original FEDformer.  In our simplified
        wrapper we still honour this shape for compatibility.
    batch_size : int
        Number of sequences per batch.
    feature_engineer : FeatureEngineer or None, default None
        Optional feature engineer used to generate lag and moving
        average features from the pivoted table.
    num_workers : int, default 0
        Number of worker processes used by the DataLoader.  The
        default of 0 disables multiprocessing and falls back to
        synchronous loading.
    """

    def __init__(
        self,
        file_path: str,
        sequence_length: int,
        forecast_horizon: int,
        label_len: int,
        batch_size: int,
        feature_engineer: Optional[FeatureEngineer] = None,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.file_path = file_path
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.label_len = label_len
        self.batch_size = batch_size
        self.feature_engineer = feature_engineer
        self.num_workers = num_workers

        # Internal storage for the pivoted and feature‑augmented DataFrame
        self.data: Optional[pd.DataFrame] = None
        # Names of the original series columns (items)
        self.target_columns: Optional[pd.Index] = None
        # Names of the calendar and engineered feature columns
        self.time_feature_columns: Optional[pd.Index] = None
        # Tensor datasets for each split
        self.train_dataset: Optional[Tuple[torch.Tensor, ...]] = None
        self.val_dataset: Optional[Tuple[torch.Tensor, ...]] = None
        self.test_dataset: Optional[Tuple[torch.Tensor, ...]] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def input_dim(self) -> int:
        """Return the number of original series (items).

        Triggers preparation of the data on first access.
        """
        if self.data is None:
            self.prepare_data()
        assert self.data is not None
        return len(self.target_columns)

    @property
    def output_dim(self) -> int:
        """Alias for :attr:`input_dim`.

        The model predicts one value per series at each forecast step.
        """
        return self.input_dim

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    def prepare_data(self) -> None:
        """Read the CSV, pivot it and attach calendar features.

        This method populates :attr:`data`, :attr:`target_columns` and
        :attr:`time_feature_columns`.  It is idempotent—calling it
        multiple times has no effect once ``self.data`` is populated.
        """
        if self.data is not None:
            return

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Training file not found: {self.file_path}")

        df = pd.read_csv(self.file_path)
        # Ensure correct types
        df["영업일자"] = pd.to_datetime(df["영업일자"])

        # Pivot to a wide table: index by date, columns by item, values are sales
        pivot_df = df.pivot_table(
            index="영업일자", columns="영업장명_메뉴명", values="매출수량"
        ).fillna(0)

        # Store original series names
        self.target_columns = pivot_df.columns

        parts = [pivot_df]

        # Apply custom feature engineering (lags and moving averages)
        if self.feature_engineer is not None:
            engineered = self.feature_engineer.transform(pivot_df)
            if engineered is not None:
                parts.append(engineered)

        # Add calendar features derived solely from the date index
        date_df = pd.DataFrame({"영업일자": pivot_df.index})
        time_features = add_date_features(date_df, date_col="영업일자")
        time_features.set_index("영업일자", inplace=True)
        self.time_feature_columns = time_features.columns
        parts.append(time_features)

        # Concatenate along columns and ensure float32 dtype
        self.data = pd.concat(parts, axis=1).astype(np.float32)
        self.data.fillna(0, inplace=True)

    def preprocess_inference_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare an unseen test file for inference.

        Given a DataFrame with columns ``영업일자``, ``영업장명_메뉴명`` and
        ``매출수량``, pivot it to the same set of series as seen
        during training, apply the feature engineer (if present) and
        append calendar features.  Any missing series are filled with
        zeros to maintain the same dimensionality.
        """
        if self.data is None:
            self.prepare_data()
        assert self.target_columns is not None and self.time_feature_columns is not None

        df = df.copy()
        df["영업일자"] = pd.to_datetime(df["영업일자"])
        pivot_df = df.pivot_table(
            index="영업일자", columns="영업장명_메뉴명", values="매출수량"
        ).fillna(0)
        # Reindex to the training columns, filling missing entries with zeros
        pivot_df = pivot_df.reindex(columns=self.target_columns, fill_value=0)

        parts = [pivot_df]
        if self.feature_engineer is not None:
            engineered = self.feature_engineer.transform(pivot_df)
            if engineered is not None:
                parts.append(engineered)
        date_df = pd.DataFrame({"영업일자": pivot_df.index})
        time_features = add_date_features(date_df, date_col="영업일자")
        time_features.set_index("영업일자", inplace=True)
        parts.append(time_features[self.time_feature_columns])  # preserve order
        final_df = pd.concat(parts, axis=1).astype(np.float32)
        final_df.fillna(0, inplace=True)
        return final_df

    # ------------------------------------------------------------------
    # Sequence construction
    # ------------------------------------------------------------------
    def _create_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate sliding windows from a fully prepared DataFrame.

        The input ``data`` must contain the same columns as produced
        during :meth:`prepare_data`, namely the original series and any
        engineered features.  The returned arrays have shapes:

        - ``x_enc``: ``(num_windows, sequence_length, num_series)``
        - ``y_seq``: ``(num_windows, label_len + forecast_horizon, num_series)``
        - ``x_mark``: ``(num_windows, sequence_length, num_time_features)``
        - ``y_mark``: ``(num_windows, label_len + forecast_horizon, num_time_features)``

        These arrays are later stacked into tensors in the DataLoader.
        """
        total_len = self.sequence_length + self.forecast_horizon
        # Extract series and features separately
        series_df = data[self.target_columns]
        mark_df = data[self.time_feature_columns]

        x_values: List[np.ndarray] = []
        y_values: List[np.ndarray] = []
        x_mark_values: List[np.ndarray] = []
        y_mark_values: List[np.ndarray] = []

        n_rows = len(data)
        for start in range(n_rows - total_len + 1):
            end_enc = start + self.sequence_length
            end_total = start + total_len
            x = series_df.iloc[start:end_enc].values
            x_mark = mark_df.iloc[start:end_enc].values
            # Decoder input: last label_len timesteps from the encoder sequence plus
            # the future horizon.  For the series target we use the real
            # observed values; at inference the future portion will be
            # replaced by zeros by the model wrapper if needed.
            dec_start = end_enc - self.label_len
            y = series_df.iloc[dec_start:end_total].values
            y_mark = mark_df.iloc[dec_start:end_total].values

            x_values.append(x)
            y_values.append(y)
            x_mark_values.append(x_mark)
            y_mark_values.append(y_mark)

        return (
            np.array(x_values, dtype=np.float32),
            np.array(y_values, dtype=np.float32),
            np.array(x_mark_values, dtype=np.float32),
            np.array(y_mark_values, dtype=np.float32),
        )

    # ------------------------------------------------------------------
    # Setup splits
    # ------------------------------------------------------------------
    def setup(self, stage: Optional[str] = None) -> None:
        """Split the data into train/val/test and construct datasets.

        The splits are determined chronologically: the earliest 70 % of
        windows are used for training, the next 15 % for validation and
        the final 15 % for testing.  This method must be called before
        requesting any dataloaders.

        Parameters
        ----------
        stage : str or None, default None
            Identifies whether the caller is training, validating or
            testing.  This parameter is currently unused but kept for
            compatibility with the PyTorch Lightning API.
        """
        if self.data is None:
            self.prepare_data()

        sequences = self._create_sequences(self.data)
        # Determine number of windows
        n_windows = sequences[0].shape[0]
        train_end = int(n_windows * 0.7)
        val_end = int(n_windows * 0.85)

        if train_end == 0 or val_end <= train_end:
            raise RuntimeError(
                "Not enough data to create train/val/test splits. "
                "Consider using a smaller sequence length or horizon."
            )

        self.train_dataset = tuple(
            seq[:train_end] for seq in sequences
        )
        self.val_dataset = tuple(
            seq[train_end:val_end] for seq in sequences
        )
        self.test_dataset = tuple(
            seq[val_end:] for seq in sequences
        )

    # ------------------------------------------------------------------
    # DataLoader factories
    # ------------------------------------------------------------------
    def _make_dataloader(self, dataset: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], shuffle: bool = False) -> DataLoader:
        """Wrap arrays into a TensorDataset and DataLoader.

        If PyTorch is not available this method will raise a
        RuntimeError.  DataLoaders are only required when training via
        PyTorch Lightning; for inference the raw numpy arrays can be
        used directly.
        """
        if torch is None or DataLoader is None or TensorDataset is None:
            raise RuntimeError("PyTorch is not available; cannot construct DataLoader.")
        tensors = [torch.from_numpy(arr) for arr in dataset]
        data_set = TensorDataset(*tensors)
        return DataLoader(
            data_set,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=False,
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None, "setup() must be called before accessing the train dataloader"
        return self._make_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None, "setup() must be called before accessing the val dataloader"
        return self._make_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None, "setup() must be called before accessing the test dataloader"
        return self._make_dataloader(self.test_dataset, shuffle=False)


__all__ = ["TimeSeriesDataModule"]