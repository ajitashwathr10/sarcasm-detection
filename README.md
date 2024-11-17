# Sarcasm Detection using Deep Learning  

This project combines traditional machine learning and advanced deep learning techniques to accurately detect sarcasm in text. By leveraging both TF-IDF-based Random Forest models and the power of BERT, it ensures robust performance across various datasets.

## Features  

### 1. Advanced Text Preprocessing  
- Custom `TextPreprocessor` Class: Handles text cleaning, lemmatization, and tokenization.  
- Stopword Removal: Removes common words that don’t contribute to meaning.  
- Special Character Handling: Filters out unnecessary symbols and punctuations.  

### 2. Dual Model Architecture  
- Traditional ML Model: Uses TF-IDF features and a Random Forest classifier for sarcasm detection.  
- Deep Learning Model: Implements a fine-tuned BERT model for contextual understanding.  
- Ensemble Prediction: Combines predictions from both models for improved accuracy.  

### 3. BERT Integration  
- Pre-Trained Model: Uses BERT for advanced feature extraction.  
- Custom Dataset Class: Prepares input for BERT with attention masks and tokenization.  
- Context Understanding: Captures nuanced meanings for accurate sarcasm detection.  

### 4. Better Feature Engineering  
- TF-IDF Representation: Replaces basic CountVectorizer for better feature weighting.  
- N-Gram Support: Includes uni-grams, bi-grams, and tri-grams for richer features.  
- BERT Word Embeddings: Utilizes BERT's contextual embeddings for deep learning.  

### 5. Improved Training Process  
- Train/Validation Split: Ensures reliable performance evaluation.  
- Cross-Validation: Supports robust model tuning.  
- Learning Rate Optimization: Implements dynamic learning rates for faster convergence.  
- Dropout Regularization: Prevents overfitting in deep learning models.  

### 6. Additional Functionality  
- Device Support: Compatible with both CPU and GPU for training and inference.  
- Comprehensive Evaluation Metrics: Includes accuracy, precision, recall, F1-score, and more.  
- Flexible Predictions: Choose between traditional ML, BERT, or ensemble methods.  

## Requirements  
- Python 3.12+
- Jupyter Notebook
- Libraries:  
  - TensorFlow / PyTorch  
  - NLTK  
  - Transformers  
  - Scikit-Learn
  - re
  - Matplotlib
  - Collections
  - Numpy
  - Pandas

## Installation 
1. Clone the repository
   ```bash
   git clone https://github.com/ajitashwathr10/Sarcasm-Detection.git
   cd Sarcasm-Detection
   ```
2. Set up a virtual environment (Optional)
   ```bash
   python -m venv venv
   source venv/bin/activate       # On Linux/Mac
   venv\Scripts\activate          # On Windows
   ```
3. Install Dependencies
   ```bash
   pip install -r requirements.txt
   ```
4. Download Pre-Trained BERT Model
   - This project uses the Hugging Face Transformers library. Ensure you have internet access when running the model to automatically download the pre-trained BERT weights.

5. Verify Installation
   ```bash
   python test_installation.py
   ```
### Additional Notes
- For GPU support, ensure you have CUDA installed and install the GPU-compatible version of TensorFlow or PyTorch
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
  ```
  ```bash
  pip install tensorflow-gpu
  ```
Now you're ready to start detecting sarcasm!

## Usage
1. Preprocess Data
   - Prepare your dataset using the `TextPreprocessor` class to clean and tokenize text.
2. Train Models
   - Train the traditional and deep learning models using the provided training pipelines.
3. Make Predictions
   - Use the ensemble or individual models to predict sarcasm in new textual data.
- Example:
  ```bash
  from sarcasm_detector import SarcasmDetector
  detector = SarcasmDetector()
  detector.train("path/to/dataset.csv")
  predictions = detector.predict(["This is such a great day!", "Oh, what a surprise..."])
  print(predictions)
  ```

## Contributing
We welcome contributions to enhance this project. Feel free to submit pull requests or raise issues!

## License
This project is licensed under the MIT License. See the LICENSE file for details.



