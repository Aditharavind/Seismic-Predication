# Earthquake Building Response Predictor

This repository contains a Streamlit application that simulates the response of a single-degree-of-freedom (SDOF) building to ground motions (synthetic or real) and uses an LSTM neural network with transfer learning to predict structural displacement.

## 📂 Project Structure
```
├── app.py
├── data/
│   └── new.rar            # Compressed dataset of ground motion records
├── assets/
│   └── seismic_response_prediction.png   # Example output plot
├── requirements.txt        # List of Python dependencies
└── README.md              # This file
```

## 🚀 Installation & Setup
1. **Clone the repository**:
   ```bash
   git clone <YOUR_GIT_URL>.git
   cd <REPO_NAME>
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Extract the dataset**:
   ```bash
   mkdir data
   mv new.rar data/
   cd data
   unrar x new.rar    # or use `unzip` if it's actually a zip
   cd ..
   ```
   Ensure your dataset files (e.g., ground motion CSVs) are now in `data/`.

## 🎯 Running the App
From the project root, run:
```bash
streamlit run app.py
```

Open the displayed URL (usually `http://localhost:8501`) in your browser.

## 🗓️ Dataset (new.rar)
- `new.rar` contains the repository of ground motion records used for training and testing.
- After extraction, place all `.csv` or time-series files in `data/`.

## 📈 Example Output
Below is an example of the output showing the actual vs. predicted displacement:

### 🔍 Seismic Response Prediction
<img src="assets/seismic_response_prediction.png" alt="Seismic Response Prediction" width="600"/>

The blue curve shows the **actual structural displacement** calculated using the mathematical model, while the orange curve shows the **predicted displacement** using the trained LSTM neural network. Ideally, the predicted curve should closely follow the actual one, which indicates good model performance.

## 📜 GitHub Upload
1. **Initialize a local repository** (if not already):
   ```bash
   git init
   git add .
   git commit -m "Initial commit: add Streamlit app, dataset archive, and example output"
   ```

2. **Add remote and push**:
   ```bash
   git remote add origin https://github.com/yourusername/your-repo.git
   git push -u origin main
   ```

> **Tip**: If `new.rar` is larger than GitHub’s 100 MB limit, consider using [Git LFS](https://git-lfs.github.com/) to track large binary files.

## 📝 License
This project is released under the MIT License. See [LICENSE](LICENSE) for details.
