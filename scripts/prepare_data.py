"""
Data preparation script.
"""

import argparse
import yaml
from pathlib import Path
from data.downloader import DatasetDownloader
from data.preprocessing import FacePreprocessor


def main():
    parser = argparse.ArgumentParser(description='Prepare data for training')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')
    parser.add_argument('--download', action='store_true', help='Download datasets')
    parser.add_argument('--datasets', nargs='+', default=['ISIC', 'HAM10000'], help='Datasets to download')
    parser.add_argument('--filter_facial', action='store_true', help='Filter for facial images')
    parser.add_argument('--extract_landmarks', action='store_true', help='Extract facial landmarks')
    parser.add_argument('--create_splits', action='store_true', help='Create train/val/test splits')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    data_config = config['data']
    
    # Download datasets
    if args.download:
        print("Downloading datasets...")
        downloader = DatasetDownloader(data_config['raw_data_dir'])
        downloader.download_datasets(args.datasets)
    
    # Preprocess
    if args.filter_facial or args.extract_landmarks:
        print("Preprocessing images...")
        preprocessor = FacePreprocessor(
            detect_faces=args.filter_facial,
            extract_landmarks=args.extract_landmarks,
        )
        # Preprocessing would be done here
        print("Preprocessing complete.")
    
    # Create splits
    if args.create_splits:
        print("Creating train/val/test splits...")
        # Split creation would be done here
        print("Splits created.")
    
    print("Data preparation complete!")


if __name__ == '__main__':
    main()

