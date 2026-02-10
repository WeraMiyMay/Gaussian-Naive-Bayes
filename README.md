<div align="center">

#  Candy Classification with Gaussian Naive Bayes  
### *Machine Learning Model for Candy Brand Prediction*

</div>

---

##  Overview

This project demonstrates a **multiclass classification** task using the **Gaussian Naive Bayes** algorithm.  
The model predicts the **candy brand** based on a set of features such as:

- chocolate  
- fruity  
- caramel  
- nougat  
- crisped rice wafer  
- whether it is a bar  
- sugar percent  
- price percent  
- and more…

The dataset used is `candy_data.csv`, containing real candy characteristics.

The project also includes:

- evaluation of classification accuracy  
- counting misclassification errors  
- graphical visualization of data  
- visualization of classification errors  

---

##  Purpose of the Model

The goal is to teach a model how different candy properties relate to their brand.  
Gaussian Naive Bayes is chosen because:

- it works well with continuous numerical features  
- it is fast and simple  
- it handles independent features efficiently  
- it is commonly used for classification tasks in early machine learning studies  

This project helps understand:

- how probabilistic models work  
- how to train and evaluate classifiers  
- how to visualize multidimensional data  
- how model errors can be interpreted  

---

##  Visualizations

The script generates:

###  Class Distribution Plot  
Shows how candies are distributed by **sugar percent** and **price percent**, using a unique color and marker for each brand.

###  Error Visualization Plot  
Highlights:

-  correctly classified samples  
-  misclassified samples  

This helps evaluate model behavior and potential weaknesses.

---

##  Technologies Used

- **Python 3.x**  
- **Pandas** — data processing  
- **NumPy** — numerical operations  
- **Scikit-learn** — Naive Bayes classifier  
- **Matplotlib** — data visualization  

---

##  Running the Project

1. Place `candy_data.csv` in the project folder.  
2. Install required libraries:

```bash
pip install numpy pandas matplotlib scikit-learn
