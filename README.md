# Facebook Live Sellers Dataset Analysis

This repository contains the **Facebook Marketplace Dataset (Live Sellers in Thailand)** along with code and analysis for exploring engagement patterns and applying clustering techniques.

---

## 📂 Repository Contents

- `Facebook_Marketplace_data.csv` → Dataset file  
- `notebook` → Jupyter Notebook with full analysis  
- `code` → Python script(s) for data preprocessing and modeling  
- `ML 01.pdf` → Notebook-style PDF with code and explanations  

---

## 📊 Dataset Overview

- **Source**: UCI Machine Learning Repository  
- **Instances (rows)**: 7,050  
- **Attributes (columns)**: 14 after preprocessing  
- **Attributes include**:
  - `status_id`: Unique identifier for each post  
  - `status_published`: Date and time of post  
  - `status_type`: Type of post (photo, video, link, status)  
  - `num_reactions`, `num_comments`, `num_shares`  
  - Reaction breakdown: `num_likes`, `num_loves`, `num_wows`, `num_hahas`, `num_sads`, `num_angrys`

---

## 🔍 Analysis & Research Questions

1. **Effect of posting time**  
   - How does the time of upload (`status_published`) affect engagement (`num_reactions`)?  
   - *Finding:* Posts made in the evening hours generally receive higher average reactions.  

2. **Correlation of engagement metrics**  
   - Relationship between reactions, comments, and shares.  
   - *Finding:* Strong positive correlations exist between these metrics, meaning popular posts tend to get more of all engagement types.  

3. **Clustering analysis (K-Means)**  
   - Features used: `status_type`, `num_reactions`, `num_comments`, `num_shares`, `num_likes`, `num_loves`, `num_wows`, `num_hahas`, `num_sads`, `num_angrys`  
   - Elbow method used to determine optimal clusters (≈ 3).  
   - *Finding:* Clusters represent **high**, **medium**, and **low engagement** posts.  

4. **Post type distribution**  
   - Count of different `status_type` categories.  
   - *Finding:* Photo posts dominate the dataset, followed by videos.  

5. **Average engagement per post type**  
   - Compared average `num_reactions`, `num_comments`, `num_shares` by post type.  
   - *Finding:* Videos typically achieve the highest average engagement across metrics.  

---


## 🚀 How to Use

- [1. Clone the repository]()
  ```bash
  git clone https://github.com/manishlakhode/Facebook-Dataset.git
  cd Facebook-Dataset

2. Install dependencies

pip install pandas matplotlib seaborn scikit-learn


3. Run the notebook

jupyter notebook notebook


4. 📈 Key Insights

Evening posts tend to maximize reactions.

Engagement metrics (reactions, comments, shares) are highly correlated.

Three clusters of posts exist: low, medium, and high engagement.

Photo posts dominate in frequency, but videos drive stronger engagement.

📜 License

This project is licensed under the MIT License.

👤 Author

Manish Lakhode