#!/usr/bin/env python3
"""
Quick Ollama Model Recommendation Script
Helps users choose the best model for their system and document size.
"""

import requests
import time
import json
from typing import Dict, List, Tuple

def test_ollama_connection() -> bool:
    """Test if Ollama is running."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_available_models() -> List[str]:
    """Get list of available Ollama models."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            return [model['name'] for model in models_data.get('models', [])]
    except:
        pass
    return []

def test_model_speed(model_name: str, test_text: str = "Hello, how are you?") -> Tuple[bool, float]:
    """Test model response speed with a simple prompt."""
    try:
        start_time = time.time()
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": f"Respond with just 'Hello' to this message: {test_text}",
                "stream": False,
                "options": {"temperature": 0.1}
            },
            timeout=30
        )
        
        end_time = time.time()
        
        if response.status_code == 200:
            return True, end_time - start_time
        else:
            return False, 0
            
    except Exception as e:
        return False, 0

def recommend_model_for_document(doc_size: int) -> Dict:
    """Recommend best model based on document size and available models."""
    
    # Model recommendations based on size - simplified to two models
    size_recommendations = {
        "small": {  # < 2000 chars
            "preferred": ["phi3:mini", "llama3.2:3b"],
            "description": "Fast model for small documents"
        },
        "medium": {  # 2000-10000 chars
            "preferred": ["llama3.2:3b", "phi3:mini"],
            "description": "Balanced model for medium documents"
        },
        "large": {  # > 10000 chars
            "preferred": ["llama3.2:3b", "phi3:mini"],
            "description": "Capable model for large documents"
        }
    }
    
    # Determine size category
    if doc_size < 2000:
        category = "small"
    elif doc_size < 10000:
        category = "medium"
    else:
        category = "large"
    
    available_models = get_available_models()
    recommended = size_recommendations[category]
    
    # Find the best available model
    best_model = None
    for model in recommended["preferred"]:
        if model in available_models:
            best_model = model
            break
    
    # If no preferred model available, use first available
    if not best_model and available_models:
        best_model = available_models[0]
    
    return {
        "category": category,
        "document_size": doc_size,
        "recommended_model": best_model,
        "description": recommended["description"],
        "all_preferred": recommended["preferred"],
        "available_models": available_models
    }

def main():
    """Main function to test and recommend models."""
    
    print("🦙 Ollama Model Recommendation Tool")
    print("=" * 40)
    
    # Test connection
    if not test_ollama_connection():
        print("❌ Ollama is not running or not accessible")
        print("\n🔧 To fix this:")
        print("1. Start Ollama: ollama serve")
        print("2. Install a model: ollama pull llama3.2:3b")
        return
    
    print("✅ Ollama is running!")
    
    # Get available models
    available_models = get_available_models()
    
    if not available_models:
        print("❌ No models installed")
        print("\n📥 Install a recommended model:")
        print("• ollama pull phi3:mini        # Fastest (2.2GB)")
        print("• ollama pull llama3.2:3b      # Balanced (2.0GB)")
        return
    
    print(f"\n📋 Available Models ({len(available_models)}):")
    for i, model in enumerate(available_models, 1):
        print(f"  {i}. {model}")
    
    # Test model speeds
    print(f"\n⚡ Testing Model Speeds...")
    model_speeds = []
    
    for model in available_models[:3]:  # Test up to 3 models
        print(f"  Testing {model}...", end=" ")
        success, speed = test_model_speed(model)
        if success:
            model_speeds.append((model, speed))
            print(f"✅ {speed:.2f}s")
        else:
            print("❌ Failed")
    
    # Sort by speed
    model_speeds.sort(key=lambda x: x[1])
    
    if model_speeds:
        fastest_model = model_speeds[0][0]
        print(f"\n🏆 Fastest Model: {fastest_model} ({model_speeds[0][1]:.2f}s)")
    
    # Document size recommendations
    print(f"\n📄 Document Size Recommendations:")
    
    test_sizes = [1000, 5000, 15000]  # Small, medium, large
    
    for size in test_sizes:
        rec = recommend_model_for_document(size)
        print(f"\n  📊 {size:,} chars ({rec['category'].upper()}):")
        print(f"    Best model: {rec['recommended_model'] or 'None available'}")
        print(f"    Description: {rec['description']}")
    
    print(f"\n💡 Quick Tips:")
    print(f"• For resumes (2-5K chars): Use phi3:mini or llama3.2:3b")
    print(f"• For long documents (10K+ chars): Use llama3.2:3b")
    print(f"• If getting timeouts: Try phi3:mini (fastest model)")
    print(f"• Install models: ollama pull <model-name>")

if __name__ == "__main__":
    main()