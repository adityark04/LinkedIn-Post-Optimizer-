"""
Merge scraped data with existing training data and retrain models.
"""

import json
import os
import random


def merge_datasets():
    """Merge scraped data with existing curated data."""
    print("="*60)
    print("Merging Datasets")
    print("="*60)
    
    datasets = []
    
    # Load existing curated data
    curated_file = 'data/full_dataset.json'
    if os.path.exists(curated_file):
        with open(curated_file, 'r', encoding='utf-8') as f:
            curated = json.load(f)
        print(f"✓ Loaded {len(curated)} curated posts")
        datasets.extend(curated)
    
    # Load scraped data
    scraped_file = 'data/scraped_training_data.json'
    if os.path.exists(scraped_file):
        with open(scraped_file, 'r', encoding='utf-8') as f:
            scraped = json.load(f)
        print(f"✓ Loaded {len(scraped)} scraped posts")
        datasets.extend(scraped)
    else:
        print("⚠️  No scraped data found. Run scrape_linkedin.py first.")
    
    if not datasets:
        print("❌ No data to merge")
        return
    
    # Shuffle combined dataset
    random.shuffle(datasets)
    
    # Split into train/test (80/20)
    split_idx = int(0.8 * len(datasets))
    train_data = datasets[:split_idx]
    test_data = datasets[split_idx:]
    
    # Save merged datasets
    os.makedirs('data', exist_ok=True)
    
    with open('data/train_data.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open('data/test_data.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    with open('data/full_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(datasets, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Dataset merge complete!")
    print(f"   Total: {len(datasets)} posts")
    print(f"   Train: {len(train_data)} posts")
    print(f"   Test:  {len(test_data)} posts")
    print(f"\n📁 Files saved:")
    print(f"   - data/train_data.json")
    print(f"   - data/test_data.json")
    print(f"   - data/full_dataset.json")
    print("="*60)


if __name__ == '__main__':
    merge_datasets()
    
    print("\n🚀 Ready to retrain models!")
    print("   Run: python train_models.py")
