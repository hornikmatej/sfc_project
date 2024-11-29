# 🤖 GPT-2 Visualization Tool

## 🎯 Overview
This project provides a comprehensive implementation of a GPT-based language model complemented by interactive visualization tools built with Streamlit. It enables users to generate text from custom prompts and offers insightful visualizations of the model's internal mechanisms, including attention matrices and embeddings.

## ✨ Features

### 🧠 Model Implementation
- **GPT-2 Architecture**: Custom implementation of the GPT model in `model.py`
- **Key Components**: GPT, CausalSelfAttention, and LayerNorm classes

### 🎮 User Interface
- **Interactive Dashboard**: Built with Streamlit in `sample.py`
- **User-Friendly**: Easy-to-use interface for text generation and visualization

### 📊 Visualizations
- **Attention Matrix**: Explore self-attention mechanisms through Plotly
- **Embedding Analysis**: Dynamic token and positional embedding visualizations
- **Activation Functions**: Interactive GELU function plots

## 🚀 Installation

### Prerequisites
Make sure you have Poetry installed on your system.

### Setup Steps
1. **Clone the Repository**:
    ```bash
    git clone https://github.com/hornikmatej/sfc_project
    cd sfc_project
    ```

2. **Install Dependencies**:
    ```bash
    poetry install
    ```

3. **Activate Environment**:
    ```bash
    poetry shell
    ```

## 💻 Usage

1. **Launch the App**:
    ```bash
    poetry run streamlit run sample.py
    ```

2. **Using the Interface**:
    - 📝 Enter your prompt in the sidebar
    - 🎲 Click "Generate" to create text
    - 🔍 Explore various visualizations

## 📁 Project Structure

### Core Files
- `model.py`: Main GPT-2 implementation
- `helpers.py`: Visualization utilities
- `sample.py`: Streamlit interface
- `poetry.toml` & `poetry.lock`: Dependency management

### Key Components
- **GPT Model**: Core language model implementation
- **Visualization Tools**: Attention matrices, embeddings, and activation functions
- **Helper Functions**: Utility functions for data processing and visualization