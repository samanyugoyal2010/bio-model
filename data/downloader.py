"""
Dataset downloader for ISIC, HAM10000, and PH2 datasets.
"""

import os
import requests
import zipfile
import tarfile
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
import pandas as pd


class DatasetDownloader:
    """
    Downloader for skin cancer datasets.
    """
    
    def __init__(self, download_dir: str = "./data/raw"):
        """
        Initialize downloader.
        
        Args:
            download_dir: Directory to save downloaded datasets
        """
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
    
    def download_file(self, url: str, filename: str, chunk_size: int = 8192):
        """Download a file with progress bar."""
        filepath = self.download_dir / filename
        
        if filepath.exists():
            print(f"File already exists: {filepath}")
            return filepath
        
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"Downloaded: {filepath}")
        return filepath
    
    def download_isic(self, version: str = "2020") -> Path:
        """
        Download ISIC dataset.
        Note: ISIC requires API access. This is a placeholder structure.
        """
        print("ISIC Archive requires API access.")
        print("Please visit https://www.isic-archive.com/ to register and download.")
        print("After downloading, extract to:", self.download_dir / "ISIC")
        return self.download_dir / "ISIC"
    
    def download_ham10000(self) -> Path:
        """
        Download HAM10000 dataset.
        Note: Actual download URLs may need to be updated.
        """
        print("Downloading HAM10000 dataset...")
        
        # HAM10000 dataset URLs (these may need to be updated)
        base_url = "https://dataverse.harvard.edu/api/access/datafile/"
        
        # Metadata
        metadata_url = f"{base_url}:persistentId?persistentId=doi:10.7910/DVN/DBW86T/CLBHCY"
        metadata_file = self.download_file(metadata_url, "HAM10000_metadata.csv")
        
        # Images (this is a placeholder - actual download may require different method)
        print("HAM10000 images download:")
        print("Please download from: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T")
        print("Extract to:", self.download_dir / "HAM10000")
        
        return self.download_dir / "HAM10000"
    
    def download_ph2(self) -> Path:
        """
        Download PH2 dataset.
        Note: PH2 may require registration. This is a placeholder.
        """
        print("PH2 dataset requires registration.")
        print("Please visit: https://www.fc.up.pt/addi/ph2%20database.html")
        print("After downloading, extract to:", self.download_dir / "PH2")
        return self.download_dir / "PH2"
    
    def download_datasets(self, datasets: List[str]) -> dict:
        """
        Download multiple datasets.
        
        Args:
            datasets: List of dataset names to download ('ISIC', 'HAM10000', 'PH2')
        
        Returns:
            Dictionary mapping dataset names to their paths
        """
        paths = {}
        
        for dataset_name in datasets:
            dataset_name = dataset_name.upper()
            
            if dataset_name == "ISIC":
                paths["ISIC"] = self.download_isic()
            elif dataset_name == "HAM10000":
                paths["HAM10000"] = self.download_ham10000()
            elif dataset_name == "PH2":
                paths["PH2"] = self.download_ph2()
            else:
                print(f"Unknown dataset: {dataset_name}")
        
        return paths
    
    def extract_archive(self, archive_path: Path, extract_to: Optional[Path] = None):
        """Extract zip or tar archive."""
        if extract_to is None:
            extract_to = archive_path.parent
        
        extract_to.mkdir(parents=True, exist_ok=True)
        
        print(f"Extracting {archive_path} to {extract_to}...")
        
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.suffix in ['.tar', '.gz', '.bz2']:
            mode = 'r'
            if archive_path.suffix == '.gz':
                mode = 'r:gz'
            elif archive_path.suffix == '.bz2':
                mode = 'r:bz2'
            
            with tarfile.open(archive_path, mode) as tar_ref:
                tar_ref.extractall(extract_to)
        
        print(f"Extracted to: {extract_to}")

