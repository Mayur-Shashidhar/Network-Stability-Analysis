# 📡 Network Stability Analysis Using a Linear Algebra Pipeline

## 📖 Course Information

- Course Name : Linear Algebra and Its Applications
- Course Code : UE24MA241B

---

## 🚀 Project Overview

This project analyzes the stability of a communication/power network using a structured **linear algebra pipeline**.

The network is modeled as a matrix, and various linear algebra techniques are applied to:

- Identify independent and redundant links  
- Predict missing data  
- Analyze traffic patterns  
- Compute a stability score  

---

## 🎯 Objectives

- Convert network data into matrix form  
- Identify rank and nullity  
- Remove redundant links  
- Perform orthogonalization  
- Predict missing values using projection  
- Estimate link weights using least squares  
- Discover dominant patterns using eigenvalues  
- Compute overall network stability  

---

## 🧠 Pipeline

Matrix Representation  
→ RREF (Gaussian Elimination)  
→ Rank & Nullity  
→ Basis Selection  
→ Gram-Schmidt Orthogonalization  
→ Projection  
→ Least Squares  
→ Eigenvalue Analysis  
→ Diagonalization  

---

## 📊 Dataset

- Nodes: 6 (A–F)  
- Links: 7 (L1–L7)  
- Matrix size: 6 × 7  

Each value represents load on a link passing through a node.

---

## 🔍 Key Results

- Rank = 6  
- Nullity = 1 (1 redundant link: L7)  
- Stability Index = 0.3556  
- Condition Number = Very High  
- Network Status = UNSTABLE  

---

## ⚠️ Insights

- L7 is redundant → can be removed  
- Top 3 eigenmodes explain ~98% of behavior  
- Load is concentrated in one dominant mode  
- Network is highly sensitive to small changes  

---

## 🔥 Critical Links

- L3  
- L1  
- L5  

These links dominate the failure mode.

---

## 🛠️ Features

- Custom RREF implementation  
- Null space using SVD  
- Gram-Schmidt orthogonalization  
- Projection for missing data  
- Least squares prediction  
- Eigenvalue analysis  
- Stability metrics calculation  

---

## 📈 Concepts Used

- Matrices  
- Gaussian Elimination  
- Rank & Nullity  
- Vector Spaces  
- Gram-Schmidt  
- Projection  
- Least Squares  
- Eigenvalues & Eigenvectors  
- Diagonalization  

---

## 💻 Tech Stack

- Python  
- NumPy  

---

## ▶️ How to Run

```bash
pip install numpy
python network_stability.py
```

---

## 📌 Output

- RREF and pivot columns  
- Rank & nullity  
- Null space  
- Orthonormal basis  
- Projection results  
- Least squares solution  
- Eigenvalues and eigenvectors  
- Stability index  
- Network status  

---

## 🧾 Conclusion

The network is **unstable** due to:

- High load concentration in one mode  
- High sensitivity to perturbations  
- Presence of redundant link  

### Recommendations:

- Redistribute traffic  
- Strengthen critical links (L3, L1, L5)  
- Use reduced model for monitoring  
- Treat L7 as backup  
