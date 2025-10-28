import pandas as pd

def read_parquet_file(file_path):
    """
    Wczytuje plik Parquet i zwraca jako DataFrame Pandas.
    
    Args:
        file_path (str): Ścieżka do pliku Parquet.
        
    Returns:
        pd.DataFrame: Wczytane dane jako DataFrame.
    """
    df = pd.read_parquet(file_path)
    return df

if __name__ == "__main__":
    file_path = "Data/sample_submission.parquet"
    data = read_parquet_file(file_path)
    print(data.head())