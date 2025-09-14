import random
from tqdm import tqdm
import subprocess
from lerobot.datasets.lerobot_dataset import LeRobotDataset

class DatasetChecker:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        print(f"Downloading and loading dataset: {dataset_name}")
        
        # Download dataset
        local_dir = f"./{dataset_name.replace('/', '_')}"
        subprocess.run([
            "hf", "download", dataset_name, 
            "--repo-type", "dataset", 
            "--local-dir", local_dir
        ], check=True)
        
        # Load dataset without downloading videos to avoid FFmpeg issues
        self.dataset = LeRobotDataset(
            dataset_name, 
            root=local_dir, 
            download_videos=False,  # Skip video download
            video_backend="pyav"    # Use alternative video backend
        )
        
        # Categorize keys
        self.all_keys = list(self.dataset.features.keys())
        self.image_keys = [k for k in self.all_keys if "image" in k.lower()]
        self.action_keys = [k for k in self.all_keys if "action" in k.lower()]
        self.state_keys = [k for k in self.all_keys if "state" in k.lower()]
        self.timestamp_keys = [k for k in self.all_keys if "timestamp" in k.lower()]
        self.index_keys = [k for k in self.all_keys if "index" in k.lower()]
        
        print(f"Dataset loaded successfully! Size: {len(self.dataset):,} samples")
        
    def check_images(self, max_samples=1000):
        """Check image data validity"""
        print(f"=== IMAGE CHECK ===")
        return self._check_keys(self.image_keys, "images", max_samples)
    
    def check_actions(self, max_samples=1000):
        """Check action data validity"""
        print(f"=== ACTION CHECK ===")
        return self._check_keys(self.action_keys, "actions", max_samples)
    
    def check_states(self, max_samples=1000):
        """Check state data validity"""
        print(f"=== STATE CHECK ===")
        return self._check_keys(self.state_keys, "states", max_samples)
    
    def check_timestamps(self, max_samples=1000):
        """Check timestamp data validity"""
        print(f"=== TIMESTAMP CHECK ===")
        return self._check_keys(self.timestamp_keys, "timestamps", max_samples)
    
    def check_indices(self, max_samples=1000):
        """Check index data validity"""
        print(f"=== INDEX CHECK ===")
        return self._check_keys(self.index_keys, "indices", max_samples)
    
    def check_all(self, max_samples=1000):
        """Check all data types"""
        print(f"Dataset size: {len(self.dataset):,} samples")
        print(f"Available keys: {self.all_keys}\n")
        
        results = {}
        results['images'] = self.check_images(max_samples)
        results['actions'] = self.check_actions(max_samples)
        results['states'] = self.check_states(max_samples)
        results['timestamps'] = self.check_timestamps(max_samples)
        results['indices'] = self.check_indices(max_samples)
        
        print(f"\n=== SUMMARY ===")
        for data_type, (valid, missing) in results.items():
            if valid + missing > 0:
                success_rate = (valid / (valid + missing)) * 100
                print(f"{data_type.capitalize()}: {valid:,} valid, {missing:,} missing ({success_rate:.1f}%)")
        
        return results
    
    def _check_keys(self, keys, data_type, max_samples):
        """Internal method to check specific keys"""
        if not keys:
            print(f"No {data_type} keys found")
            return 0, 0
            
        total_samples = len(self.dataset)
        check_count = min(max_samples, total_samples)
        
        print(f"{data_type.capitalize()} keys: {keys}")
        print(f"Checking {check_count:,} samples...")
        
        valid_count = 0
        missing_count = 0
        
        for i in range(check_count):
            if i % 5000 == 0 and i > 0:
                print(f"Progress: {i:,}/{check_count:,}")
            
            try:
                sample = self.dataset[i]
                if all(key in sample and sample[key] is not None for key in keys):
                    valid_count += 1
                else:
                    missing_count += 1
            except:
                missing_count += 1
        
        success_rate = (valid_count / check_count) * 100 if check_count > 0 else 0
        print(f"Results: {valid_count:,} valid, {missing_count:,} missing ({success_rate:.1f}%)\n")
        
        return valid_count, missing_count

def main():
    """Main function to run dataset validation"""
    dataset_name = "yadan0418/record-test"
    
    # Create checker (this will download and load the dataset)
    checker = DatasetChecker(dataset_name)
    
    # Run comprehensive validation
    results = checker.check_all(max_samples=1000)
    
    return checker, results

# Run main function
if __name__ == "__main__":
    checker, results = main()