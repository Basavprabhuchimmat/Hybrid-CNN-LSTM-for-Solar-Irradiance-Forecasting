"""
Test script for Solar Forecasting API endpoints
Run this after starting the FastAPI server to verify all endpoints work correctly.

Usage:
    python test_api.py
"""

import requests
import json
from pathlib import Path

# API base URL
BASE_URL = "http://127.0.0.1:5000"

def test_health_check():
    """Test the health check endpoint"""
    print("\n=== Testing Health Check ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_forecast():
    """Test the forecasting endpoint"""
    print("\n=== Testing Forecast Endpoint ===")
    data = {
        "irradiance_sequence": [
            450.2, 478.5, 490.1, 502.3, 515.7, 530.2, 
            545.8, 560.1, 575.3, 590.5, 605.2, 618.9,
            632.1, 645.8, 658.3, 670.5, 682.1, 693.8, 
            704.2, 714.6
        ]
    }
    response = requests.post(f"{BASE_URL}/forecast", json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_training_data():
    """Test the training data endpoint"""
    print("\n=== Testing Training Data Endpoint ===")
    response = requests.get(f"{BASE_URL}/api/training-data")
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Available training logs: {list(data.keys())}")
    return response.status_code == 200

def test_predict_single(image_path=None):
    """Test the single image prediction endpoint"""
    print("\n=== Testing Single Image Prediction ===")
    
    if image_path is None or not Path(image_path).exists():
        print("⚠️  No test image provided or image not found")
        print("To test this endpoint, provide an image path:")
        print("  test_predict_single('path/to/image.png')")
        return None
    
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{BASE_URL}/predict", files=files)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_hybrid_predict(image_dir=None):
    """Test the hybrid prediction endpoint"""
    print("\n=== Testing Hybrid Prediction ===")
    
    if image_dir is None or not Path(image_dir).exists():
        print("⚠️  No test image directory provided or directory not found")
        print("To test this endpoint, provide a directory with 10+ images:")
        print("  test_hybrid_predict('path/to/images/')")
        return None
    
    image_files = sorted(Path(image_dir).glob("*.png"))[:20]
    
    if len(image_files) < 10:
        print(f"⚠️  Need at least 10 images, found {len(image_files)}")
        return None
    
    files = [('files', open(img, 'rb')) for img in image_files]
    
    try:
        response = requests.post(f"{BASE_URL}/hybrid_predict", files=files)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    finally:
        for _, f in files:
            f.close()

def run_all_tests():
    """Run all available tests"""
    print("="*50)
    print("Solar Forecasting API Test Suite")
    print("="*50)
    
    results = {
        "Health Check": test_health_check(),
        "Forecast": test_forecast(),
        "Training Data": test_training_data(),
    }
    
    print("\n" + "="*50)
    print("Test Results Summary")
    print("="*50)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print("\n⚠️  Note: Single image and hybrid prediction tests require test data.")
    print("Run them separately with actual image files/directories.")

if __name__ == "__main__":
    # Run basic tests
    run_all_tests()
    
    # Uncomment and update paths to test image endpoints
    # test_predict_single("path/to/test_image.png")
    # test_hybrid_predict("path/to/test_images/")
