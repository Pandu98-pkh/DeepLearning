# Week 6 - RNN Models for Sarcasm Detection

## Overview
This assignment implements Recurrent Neural Network (RNN) models for sarcasm detection using the DeteksiSarkasme.json dataset. The implementation covers both **PyTorch** and **TensorFlow** frameworks with comprehensive evaluation metrics and hyperparameter tuning.

## Dataset
- **Source**: DeteksiSarkasme.json
- **Type**: Binary Classification (Sarcastic vs Non-Sarcastic)
- **Total Samples**: 26,709 headlines
- **Features**: Text headlines from news articles
- **Target**: Binary labels (0: Not Sarcastic, 1: Sarcastic)
- **Distribution**: 56.1% Non-Sarcastic, 43.9% Sarcastic

## Requirements Fulfilled ✅

### 1. Deep Learning Models (Both Frameworks)
**TensorFlow/Keras Models:**
- SimpleRNN
- LSTM
- GRU  
- Bidirectional LSTM (BiLSTM)
- Bidirectional GRU (BiGRU)
- Advanced Multi-layer LSTM
- Hyperparameter Tuned Model (using Keras Tuner)

**PyTorch Models:**
- LSTM
- GRU
- SimpleRNN
- Bidirectional LSTM (BiLSTM)
- Bidirectional GRU (BiGRU)

### 2. Comprehensive Evaluation Metrics
All models are evaluated using:
- **Accuracy** (Target: ≥70%)
- **Precision**
- **Recall**
- **F1-Score**
- **AUC (Area Under Curve)**
- **ROC Curves**

### 3. Visualizations
- Training/Validation **Accuracy** plots
- Training/Validation **Loss** plots
- **Confusion Matrices** for all models
- **ROC Curves** comparison
- **Model Performance** comparison charts

### 4. Hyperparameter Tuning
- **Keras Tuner** implementation for TensorFlow models
- **RandomSearch** strategy with 10 trials
- Optimized parameters:
  - RNN type (LSTM, GRU, BiLSTM)
  - Hidden units (32-128)
  - Dropout rates (0.2-0.6)
  - Learning rates (1e-4 to 1e-2)
  - Embedding dimensions (50-200)

### 5. Performance Requirements
- Target: **≥70% accuracy** on both training and test sets
- Early stopping and learning rate reduction implemented
- Cross-validation with train/validation/test splits (60/20/20)

## Technical Implementation

### Data Preprocessing
- Text cleaning and normalization
- Tokenization with vocabulary size of 10,000
- Sequence padding to maximum length of 50
- Train/Validation/Test split: 60/20/20

### Model Architecture
- **Embedding Layer**: 100-dimensional word embeddings
- **RNN Layers**: Various architectures (LSTM, GRU, SimpleRNN)
- **Dense Layers**: Fully connected layers with dropout
- **Output Layer**: Sigmoid activation for binary classification

### Training Configuration
- **Optimizer**: Adam with learning rate scheduling
- **Loss Function**: Binary Cross-Entropy
- **Batch Size**: 32
- **Epochs**: 20-25 with early stopping
- **Callbacks**: Early stopping, learning rate reduction

## File Structure
```
06. Week 6/
├── Dataset/
│   └── DeteksiSarkasme.json
├── Notebook Assignment/
│   ├── RNN Models.ipynb          # Main implementation
│   └── README.md                 # This file
└── hyperparameter_tuning/        # Keras Tuner results
```

## Usage Instructions

1. **Setup Environment**:
   ```bash
   pip install tensorflow torch transformers keras-tuner scikit-learn matplotlib seaborn wordcloud
   ```

2. **Run the Notebook**:
   - Open `RNN Models.ipynb` in Jupyter/VSCode
   - Execute cells sequentially
   - Training may take 30-60 minutes depending on hardware

3. **View Results**:
   - Model comparison table
   - Training history plots
   - Confusion matrices
   - ROC curves
   - Classification reports

## Key Results

### Model Performance Summary
The notebook generates a comprehensive comparison table showing:
- Model accuracy rankings
- Performance metrics for all models
- Framework comparison (TensorFlow vs PyTorch)
- Best performing architecture identification

### Expected Outcomes
- Multiple models achieving >70% accuracy
- Detailed analysis of model strengths/weaknesses
- Visualization of training dynamics
- Hyperparameter optimization results

## Advanced Features

### Error Analysis
- False positive/negative analysis
- Sample misclassified examples
- Model confidence examination

### Comprehensive Visualizations
- Word clouds for sarcastic vs non-sarcastic text
- Text length distributions
- Sequence length analysis
- Multi-metric performance comparison

## Technical Notes

### GPU Acceleration
- Automatic GPU detection and usage
- CUDA support for PyTorch models
- TensorFlow GPU optimization

### Memory Optimization
- Efficient data loading with DataLoaders
- Batch processing for large datasets
- Model checkpointing for best weights

## Assignment Deliverables

✅ **Deep Learning Models**: Both PyTorch and TensorFlow implementations  
✅ **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, AUC, ROC  
✅ **Visualizations**: Training curves, confusion matrices, ROC curves  
✅ **Hyperparameter Tuning**: Keras Tuner implementation  
✅ **Performance Target**: ≥70% accuracy achievement  
✅ **Comprehensive Analysis**: Error analysis and model comparison  

## Author
**Deep Learning Course - Week 6 Assignment**  
**Topic**: RNN Models for Sarcasm Detection  
**Framework**: TensorFlow/Keras + PyTorch  
**Dataset**: DeteksiSarkasme.json

---
*This implementation demonstrates advanced RNN techniques for text classification with comprehensive evaluation and comparison across multiple architectures and frameworks.*