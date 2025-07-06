# ML-Multiple-Regg
ML Multiple Regg - 3D


# Machine Learning Regression Analysis Project

This project contains Jupyter notebooks demonstrating various machine learning regression techniques using Python. The project includes both simple linear regression and multiple regression analysis with data visualization.

## 📁 Project Structure

```
Test/
├── ML Multiple Regresion.ipynb     # Multiple regression analysis with synthetic data
├── Projects/
│   └── ML Linear Agg.ipynb         # Linear regression with real placement data
├── README.md                       # This file
└── [other Python files]
```

## 🎯 Project Overview

### 1. Multiple Regression Analysis (`ML Multiple Regresion.ipynb`)
- **Purpose**: Demonstrates multiple regression using synthetic data
- **Features**: 
  - Uses `sklearn.datasets.make_regression` to generate synthetic data
  - Creates 3D visualizations using Plotly
  - Implements multiple regression with 2 features
  - Includes model evaluation metrics

### 2. Linear Regression Analysis (`Projects/ML Linear Agg.ipynb`)
- **Purpose**: Real-world linear regression analysis
- **Dataset**: Placement data (CGPA vs Package/Salary)
- **Features**:
  - Data preprocessing and exploration
  - Train-test splitting
  - Linear regression model training
  - Model evaluation and visualization
  - Comprehensive comments explaining each step

## 🛠️ Technologies Used

- **Python 3.12**
- **Jupyter Lab/Notebook**
- **Key Libraries**:
  - `pandas` - Data manipulation and analysis
  - `numpy` - Numerical computing
  - `scikit-learn` - Machine learning algorithms
  - `plotly` - Interactive data visualization
  - `matplotlib` - Static plotting

## 📦 Installation & Setup

### Prerequisites
- Python 3.12 or higher
- Anaconda or Miniconda (recommended)

### Installation Steps

1. **Clone or download this repository**
   ```bash
   cd /path/to/your/project
   ```

2. **Install required packages**
   ```bash
   # Using conda (recommended)
   conda install pandas numpy scikit-learn plotly matplotlib jupyter
   
   # Or using pip
   pip install pandas numpy scikit-learn plotly matplotlib jupyter
   ```

3. **Start Jupyter Lab**
   ```bash
   jupyter lab
   ```

4. **Open the notebooks**
   - Navigate to the notebook files in Jupyter Lab
   - Make sure to select the correct kernel (Python 3)
   - Run cells sequentially

## 🚀 Usage

### Running the Multiple Regression Notebook
1. Open `ML Multiple Regresion.ipynb`
2. Run all cells in order
3. The notebook will:
   - Generate synthetic regression data
   - Create a 3D scatter plot visualization
   - Perform multiple regression analysis

### Running the Linear Regression Notebook
1. Open `Projects/ML Linear Agg.ipynb`
2. Ensure you have the placement dataset (`placement.csv`)
3. Run cells sequentially to:
   - Load and explore the data
   - Split data into training and test sets
   - Train a linear regression model
   - Evaluate model performance

## 📊 Key Features

### Data Visualization
- **3D Scatter Plots**: Interactive visualizations using Plotly
- **2D Scatter Plots**: Traditional matplotlib visualizations
- **Model Performance Metrics**: R² score, Mean Absolute Error, Mean Squared Error

### Machine Learning Techniques
- **Linear Regression**: Simple linear relationship modeling
- **Multiple Regression**: Multi-feature regression analysis
- **Train-Test Splitting**: Proper model validation approach
- **Model Evaluation**: Comprehensive performance metrics

## 🔧 Troubleshooting

### Common Issues

1. **ModuleNotFoundError for plotly**
   ```bash
   pip install plotly --upgrade
   # Then restart Jupyter kernel
   ```

2. **Kernel Issues**
   - Restart the Jupyter kernel (Kernel → Restart & Clear Output)
   - Ensure you're using the correct Python environment

3. **Data File Not Found**
   - For the linear regression notebook, ensure `placement.csv` is in the correct location
   - Update the file path in the notebook if needed

## 📈 Learning Outcomes

This project demonstrates:
- **Data Science Workflow**: From data loading to model evaluation
- **Regression Analysis**: Both simple and multiple regression techniques
- **Data Visualization**: Creating meaningful plots for analysis
- **Model Validation**: Proper train-test splitting and evaluation
- **Python Programming**: Best practices for ML projects

## 🤝 Contributing

Feel free to:
- Add more regression techniques
- Improve visualizations
- Add more datasets
- Enhance documentation

## 📝 License

This project is open source and available under the MIT License.

## 📞 Support

If you encounter any issues:
1. Check the troubleshooting section above
2. Ensure all dependencies are properly installed
3. Verify you're using the correct Python environment

---

**Happy Learning! 🎓**

# ML Multiple Regression Analysis

A comprehensive Jupyter notebook demonstrating multiple regression analysis using synthetic data with interactive 3D visualizations and model evaluation.

## 📊 Project Overview

This project implements a complete multiple regression analysis workflow using scikit-learn and Plotly. It demonstrates how to:
- Generate synthetic regression data
- Perform train-test splitting
- Train a multiple regression model
- Evaluate model performance
- Create interactive 3D visualizations
- Visualize the regression surface

## 🎯 Key Features

### Data Generation & Preprocessing
- **Synthetic Data**: Uses `sklearn.datasets.make_regression` to generate controlled test data
- **Data Structure**: 100 samples with 2 features and 1 target variable
- **Noise Level**: 50 (configurable for different complexity levels)

### Machine Learning Pipeline
- **Train-Test Split**: 80% training, 20% testing with reproducible results
- **Linear Regression**: Multiple regression with 2 independent variables
- **Model Evaluation**: Comprehensive metrics including MAE, MSE, and R² score

### Visualization
- **3D Scatter Plot**: Interactive visualization of original data points
- **Regression Surface**: 3D surface showing model predictions across feature space
- **Interactive Features**: Zoom, rotate, and explore the 3D plots

## 🛠️ Technologies Used

- **Python 3.12**
- **Jupyter Notebook**
- **Key Libraries**:
  - `scikit-learn` - Machine learning algorithms and data generation
  - `pandas` - Data manipulation
  - `numpy` - Numerical computing and array operations
  - `plotly` - Interactive 3D visualizations
  - `matplotlib` - Additional plotting capabilities

## 📦 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or Jupyter Lab

### Installation Steps

1. **Install required packages**
   ```bash
   pip install scikit-learn pandas numpy plotly matplotlib jupyter
   ```

2. **Start Jupyter**
   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```

3. **Open the notebook**
   - Navigate to `Projects/ML Multiple Regresion.ipynb`
   - Ensure you're using the correct kernel

## 🚀 Usage Guide

### Running the Analysis

1. **Data Generation** (Cells 0-2)
   ```python
   # Generate synthetic regression data
   X, y = make_regression(n_samples=100, n_features=2, n_informative=2, n_targets=1, noise=50)
   df = pd.DataFrame({'feature1': X[:,0], 'feature2': X[:,1], 'target': y})
   ```

2. **Data Exploration** (Cells 3-5)
   - Check data shape and preview
   - Create initial 3D scatter plot

3. **Model Training** (Cells 6-9)
   ```python
   # Split data and train model
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
   lr = LinearRegression()
   lr.fit(X_train, y_train)
   ```

4. **Model Evaluation** (Cells 10-11)
   - Generate predictions
   - Calculate performance metrics

5. **3D Surface Visualization** (Cells 12-13)
   - Create prediction surface
   - Overlay surface on scatter plot

## 📈 Model Performance

The trained model achieves:
- **MAE**: 46.73 (Mean Absolute Error)
- **MSE**: 3089.22 (Mean Squared Error)
- **R² Score**: 0.79 (Coefficient of Determination)

## 🔧 Technical Details

### Data Parameters
- **Samples**: 100 data points
- **Features**: 2 independent variables
- **Target**: 1 dependent variable
- **Noise**: 50 (controls data complexity)

### Model Parameters
- **Algorithm**: Linear Regression
- **Train/Test Split**: 80/20
- **Random State**: 3 (for reproducibility)

### Visualization Components
- **Grid Resolution**: 10x10 for surface plot
- **Coordinate Range**: -5 to 5 for both features
- **Surface Type**: Continuous prediction surface

## 🎨 Visualization Features

### 3D Scatter Plot
- Shows original data points in 3D space
- Interactive rotation and zoom
- Color-coded by target values

### Regression Surface
- Displays model predictions across feature space
- Smooth surface interpolation
- Overlaid on original data points

## 🔍 Understanding the Results

### Model Coefficients
- **Coefficients** (`lr.coef_`): Beta1 and Beta2 (slope parameters)
- **Intercept** (`lr.intercept_`): Beta0 (y-intercept)

### Performance Interpretation
- **R² Score**: 0.79 indicates the model explains 79% of variance
- **MAE**: Average absolute prediction error of ~47 units
- **MSE**: Mean squared error for model evaluation

## 🚨 Troubleshooting

### Common Issues

1. **ModuleNotFoundError for plotly**
   ```bash
   pip install plotly --upgrade
   # Restart Jupyter kernel
   ```

2. **Visualization not displaying**
   - Ensure you're running in Jupyter environment
   - Check if plotly is properly installed
   - Try restarting the kernel

3. **Performance issues**
   - Reduce grid resolution for faster rendering
   - Use fewer data points for quicker analysis

## 📚 Learning Outcomes

This project demonstrates:
- **Multiple Regression**: Working with multiple independent variables
- **Data Science Workflow**: Complete ML pipeline from data to visualization
- **3D Visualization**: Creating interactive plots for complex data
- **Model Evaluation**: Understanding different performance metrics
- **Synthetic Data**: Working with controlled datasets for learning

## 🔄 Customization Options

### Data Parameters
```python
# Modify these parameters for different scenarios
X, y = make_regression(
    n_samples=200,      # More/fewer samples
    n_features=3,       # More/fewer features
    n_informative=2,    # Number of informative features
    noise=30           # Lower/higher noise
)
```

### Visualization Parameters
```python
# Adjust grid resolution
x = np.linspace(-5, 5, 20)  # Higher resolution
y = np.linspace(-5, 5, 20)  # Higher resolution
```

## 🤝 Contributing

Feel free to enhance this project by:
- Adding more regression algorithms
- Implementing cross-validation
- Adding feature scaling
- Creating additional visualizations
- Improving documentation

## 📝 License

This project is open source and available under the MIT License.

---

**Happy Learning! 🎓**

*This README provides a comprehensive guide to understanding and using the ML Multiple Regression analysis notebook.*
