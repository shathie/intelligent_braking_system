"""
Automated dataset download script.
Note: This may not work if datasets require login/approval.
"""

import os
import requests
from tqdm import tqdm
import zipfile
import gdown


def download_file(url: str, output_path: str) -> None:
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=output_path) as pbar:
            for data in response.iter_content(chunk_size=8192):
                f.write(data)
                pbar.update(len(data))


def download_thu_dataset():
    """Download Tsinghua dataset."""
    # Note: This URL might need to be updated based on actual download link
    url = "https://thu-rsxd.com/dxhdiefb/download"  # Placeholder - check actual URL
    output_dir = "data/external/thu_road_surface"
    os.makedirs(output_dir, exist_ok=True)
    
    # This would need to be adapted based on actual download mechanism
    print("Tsinghua dataset requires manual download from: https://thu-rsxd.com/dxhdiefb/")
    print(f"Please download and extract to: {os.path.abspath(output_dir)}")


def download_mendeley_dataset():
    """Download Mendeley dataset."""
    # Note: Mendeley might require API key or direct download link
    url = "https://data.mendeley.com/public-api/datasets/w86hvkrzc5/files"  # Placeholder
    output_dir = "data/external/mendeley_vehicle"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Mendeley dataset requires manual download from: https://data.mendeley.com/datasets/w86hvkrzc5/3")
    print(f"Please download and extract to: {os.path.abspath(output_dir)}")


def download_sample_data():
    """Download sample data for testing."""
    # Create sample data if real datasets aren't available
    output_dir = "data/external/sample_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create sample images
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    
    import numpy as np
    from PIL import Image
    
    # Generate 10 sample images per class
    classes = ['dry', 'wet', 'icy', 'rough', 'slippery']
    for class_name in classes:
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        for i in range(10):
            # Generate random image (simulating road texture)
            img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            Image.fromarray(img).save(os.path.join(class_dir, f"{i:03d}.jpg"))
    
    # Create sample CAN data
    import pandas as pd
    
    num_samples = 500
    timestamps = pd.date_range(start='2023-01-01', periods=num_samples, freq='10ms')
    
    can_data = pd.DataFrame({
        'timestamp': timestamps,
        'a_x': np.random.randn(num_samples) * 2.0,
        'a_y': np.random.randn(num_samples) * 1.0,
        'omega_FL': np.random.rand(num_samples) * 100,
        'omega_FR': np.random.rand(num_samples) * 100,
        'omega_RL': np.random.rand(num_samples) * 100,
        'omega_RR': np.random.rand(num_samples) * 100,
        'v_x': np.random.rand(num_samples) * 30,
        'steering_angle': np.random.randn(num_samples) * 10,
        'brake_pressure': np.random.rand(num_samples) * 100,
        'yaw_rate': np.random.randn(num_samples) * 2,
        'mu': np.random.rand(num_samples) * 0.8 + 0.1
    })
    
    can_data.to_csv(os.path.join(output_dir, "can_data.csv"), index=False)
    
    # Copy images to Mendeley-style structure
    import shutil
    for class_name in classes:
        for i in range(10):
            src = os.path.join(output_dir, class_name, f"{i:03d}.jpg")
            dst_dir = os.path.join(output_dir, "images")
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy(src, os.path.join(dst_dir, f"{i:03d}_{class_name}.jpg"))
    
    print(f"Sample data created at: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    print("="*70)
    print("DATASET DOWNLOAD SCRIPT")
    print("="*70)
    
    # Try to download real datasets
    download_thu_dataset()
    download_mendeley_dataset()
    
    # Intentionally do not generate synthetic/sample data in the default workflow.
    # Call download_sample_data() manually only for isolated debugging.
    
    print("\n" + "="*70)
    print("Please check the data/external/ directory and verify datasets are present.")
    print("="*70)