import sys
import pandas as pd
from pathlib import Path
from typing import Union
sys.path.append(str(Path(__file__).resolve().parent.parent))
class DatasetLoader:
    def __init__(self, file_path: Union[str, Path]):
        self._file_path = Path(file_path)
        self._df: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        if not self._file_path.exists():
            raise FileNotFoundError(f"File not found: {self._file_path}")
        
        if self._file_path.suffix == ".json":
            self._df = pd.read_json(self._file_path)
        elif self._file_path.suffix == ".csv":
            self._df = pd.read_csv(self._file_path)
        else:
            raise ValueError(f"Unsupported file format: {self._file_path.suffix}")
        
        return self._df
    
    def describe(self, n: int = 5):
        if self._df is None:
            raise ValueError("Dataset not loaded. Call .load() first.")
        
        print("📊 Dataset shape:", self._df.shape)
        print("🗂 Columns:", list(self._df.columns))
        print(f"\n🔎 First {n} rows:")
        print(self._df.head(n))