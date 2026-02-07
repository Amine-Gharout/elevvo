# Elevvo Machine Learning Projects

A collection of machine learning and deep learning projects covering computer vision, audio classification, and customer analytics. Each project includes comprehensive exploratory data analysis, multiple modeling approaches, and detailed evaluations.

---

## 📋 Projects Overview

### 1. 🚦 Traffic Sign Recognition (GTSRB)
**Location:** `Trafic_Recognition/`

**Objective:** Classify German traffic signs into 43 classes using deep learning.

**Dataset:** GTSRB (German Traffic Sign Recognition Benchmark)
- 39,209 training images
- 12,630 test images
- 43 traffic sign classes
- Varying image sizes and lighting conditions

**Key Features:**
- ⚡ **Low RAM Optimized** - Designed for devices with 4-8 GB RAM
- Chunked data loading to reduce memory usage
- Lightweight model architectures
- Batch size optimization (16 vs 64)

**Techniques:**
- Custom lightweight CNN (16→32→64 filters, ~50K parameters)
- Transfer learning with MobileNetV2 (α=0.35, ~400K parameters)
- Data augmentation (rotation, shift, zoom, brightness)
- Image preprocessing (32×32 and 64×64 resizing)

**Performance:**
- Custom CNN: ~93-96% accuracy
- MobileNetV2: ~94-97% accuracy

**Tools:** Python, TensorFlow/Keras, OpenCV, scikit-learn

---

### 2. 🎵 Music Genre Classification
**Location:** `Music_Genre_Classification/`

**Objective:** Multi-class classification of music into 10 genres.

**Dataset:** GTZAN Dataset
- 1,000 audio tracks (100 per genre)
- 10 genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
- 30-second audio clips
- Pre-extracted features and mel spectrograms

**Approaches:**

**1. Tabular Machine Learning**
- Uses pre-extracted audio features (MFCCs, spectral features, tempo, etc.)
- Models: Logistic Regression, KNN, SVM, Random Forest, Gradient Boosting
- Feature importance analysis
- Cross-validation for robust evaluation

**2. CNN on Spectrograms**
- Mel spectrogram images as input
- Custom CNN architecture
- Image augmentation techniques

**3. Transfer Learning**
- Pre-trained models (VGG16, ResNet50)
- Fine-tuning on spectrogram images

**Key Features:**
- Comprehensive EDA with correlation analysis
- MFCC feature visualization by genre
- Multiple modeling approaches comparison
- Balanced dataset (100 samples per genre)

**Tools:** Python, TensorFlow/Keras, scikit-learn, librosa, pandas, seaborn

---

### 3. 👥 Customer Segmentation
**Location:** `Costumer_Segmentation/`

**Objective:** Segment customers based on purchasing behavior for targeted marketing strategies.

**Dataset:** Mall Customers Dataset
- Customer demographics (Age, Gender, Income)
- Spending behavior metrics
- Annual income and spending scores

**Techniques:**
- K-Means clustering
- Hierarchical clustering
- PCA for dimensionality reduction
- Elbow method for optimal cluster selection
- Customer profiling and analysis

**Applications:**
- Targeted marketing campaigns
- Customer retention strategies
- Product recommendations
- Store layout optimization

**Tools:** Python, scikit-learn, pandas, matplotlib, seaborn

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.12+
- Virtual environment (recommended)

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd elevvo
```

2. **Create and activate virtual environment:**
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. **Install dependencies:**
```bash
# For Traffic Sign Recognition
cd Trafic_Recognition
pip install -r requirements.txt

# For Customer Segmentation
cd ../Costumer_Segmentation
pip install -r requirements.txt
```

4. **Download datasets:**

Each project includes a `data_download.py` script using `kagglehub`:

```bash
# Traffic Sign Recognition
python Trafic_Recognition/data_download.py

# Music Genre Classification
python Music_Genre_Classification/data_download.py

# Customer Segmentation
python Costumer_Segmentation/data_download.py
```

---

## 📁 Repository Structure

```
elevvo/
├── README.md
├── .gitignore
├── .python-version
├── env/                                    # Virtual environment
├── Trafic_Recognition/
│   ├── traffic_sign_recognition.ipynb     # Main notebook (Low RAM optimized)
│   ├── data_download.py                   # Dataset downloader
│   ├── requirements.txt
│   └── Data/                              # Dataset (downloaded)
├── Music_Genre_Classification/
│   ├── music_genre_classification.ipynb   # Main notebook
│   ├── data_download.py
│   └── Data/
│       ├── features_30_sec.csv
│       ├── features_3_sec.csv
│       ├── genres_original/               # Audio files
│       └── images_original/               # Spectrograms
└── Costumer_Segmentation/
    ├── customer_segmentation.ipynb        # Main notebook
    ├── data_download.py
    ├── Mall_Customers.csv
    └── requirements.txt
```

---

## 🚀 Usage

### Running Notebooks

1. **Start Jupyter:**
```bash
jupyter notebook
```

2. **Navigate to desired project folder**

3. **Open and run the notebook:**
   - Traffic Sign Recognition: `traffic_sign_recognition.ipynb`
   - Music Genre Classification: `music_genre_classification.ipynb`
   - Customer Segmentation: `customer_segmentation.ipynb`

### Memory Considerations

**For Traffic Sign Recognition on low RAM devices:**
- The notebook is optimized for 4-8 GB RAM
- Uses batch size of 16
- Implements chunked data loading
- Includes aggressive garbage collection

If you have more RAM (16+ GB), you can increase:
- `BATCH_SIZE` from 16 to 32 or 64
- `chunk_size` in loading functions
- Model capacity (more filters, larger dense layers)

---

## 📊 Key Technologies

- **Deep Learning:** TensorFlow, Keras
- **Computer Vision:** OpenCV, PIL
- **Machine Learning:** scikit-learn
- **Data Processing:** NumPy, pandas
- **Visualization:** Matplotlib, Seaborn
- **Audio Processing:** librosa (Music Genre project)

---

## 🎯 Project Highlights

### Traffic Sign Recognition
✅ Memory-efficient implementation for resource-constrained devices  
✅ Multiple model architectures (Custom CNN + Transfer Learning)  
✅ Comprehensive data augmentation  
✅ Detailed per-class accuracy analysis  

### Music Genre Classification
✅ Three different approaches (Tabular ML, CNN, Transfer Learning)  
✅ Extensive feature engineering and analysis  
✅ Multiple model comparison  
✅ MFCC and spectrogram visualization  

### Customer Segmentation
✅ Unsupervised learning techniques  
✅ Customer profiling and insights  
✅ Business-oriented analysis  
✅ Actionable recommendations  

---

## 📝 Notes

- All datasets are downloaded via `kagglehub` (requires Kaggle account)
- Notebooks include detailed markdown explanations
- Models are trained from scratch (no pre-trained weights required except for transfer learning)
- Each project is self-contained with its own requirements

---

## 🤝 Contributing

Feel free to fork this repository and submit pull requests for improvements or additional features.

---

## 📄 License

This repository is for educational and demonstration purposes.

---

## 👤 Author

Elevvo Machine Learning Projects

---

**Last Updated:** February 2026