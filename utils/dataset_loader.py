import sys
import pandas as pd
from pathlib import Path
from typing import Union, Iterator, List
import ijson

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
    
    def load_batch(self, batch_size: int = 200) -> Iterator[pd.DataFrame]:
        with open(self._file_path, "r", encoding="utf-8") as f:
            parser = ijson.items(f, "item")
            batch = []
            for item in parser:
                batch.append(item)
                if len(batch) >= batch_size:
                    yield pd.DataFrame(batch)
                    batch = []
            if batch:
                yield pd.DataFrame(batch)
                
    def describe(self, n: int = 5):
        if self._df is None:
            raise ValueError("Dataset not loaded. Call .load() first.")
        
        print("📌 Dataset shape:", self._df.shape)
        print("📌 Columns:", list(self._df.columns))
        print(f"\n📌 First {n} rows:")
        print(self._df.head(n))

    def analyze_by(self, col: set) -> None:
        if self._df is None:
            raise ValueError("Dataset not loaded. Call .load() first")
        df = self._df

        print("\n📌 Missing values and empty strings:")
        print(df.isnull().sum())
        print(f"\n📌 Empty '{col}': ", (df[col] == "").sum())
        
        print("\n📌 Class distribution (counts):")
        print(df[col].value_counts())

        print("\n📌 Class distribution (normalized):")
        print(df[col].value_counts(normalize=True))

        print("\n📌 Text length statistics (characters):")
        print(df[col].str.len().describe())

        print("\n📌 Text length statistics (words):")
        print(df[col].str.split().str.len().describe())

     