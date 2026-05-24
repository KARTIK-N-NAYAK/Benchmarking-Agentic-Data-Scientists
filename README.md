# ADSB

Brief description of your research project and its main objectives.

## Overview

Provide a more detailed explanation of:
- What problem this research addresses
- Your approach or methodology
- Key findings or expected outcomes

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management.

### Prerequisites
- Python 3.8+ 
- uv (install with `curl -LsSf https://astral.sh/uv/install.sh | sh`)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/your-project.git
cd your-project

# Create virtual environment and install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

## Usage

```bash
# Example command to run main analysis
uv run python src/main.py
```

## Dependencies

To add new dependencies:
```bash
uv add package-name
uv add --dev package-name  # for development dependencies
```

Here is a formatted snippet you can copy directly into your `README.md` file. It covers the prerequisites, authentication, and rule acceptance required to use the Kaggle API.

-----

## Data Acquisition

To reproduce the dataset usage, you must download the data directly from Kaggle using their API.

### 1\. API Credentials (`kaggle.json`)

You need to authenticate to download the dataset:

1.  Log in to your Kaggle account.
2.  Go to **Settings** -\> **API** section.
3.  Click **Create New Token**. This will download a `kaggle.json` file.
4.  Place this file in the correct location for your operating system:
      * **Linux/Mac:** `~/.kaggle/kaggle.json`
      * **Windows:** `C:\Users\<Windows-Username>\.kaggle\kaggle.json`

> **Note for Linux/Mac users:** Ensure your key is readable only by you by running:
> `chmod 600 ~/.kaggle/kaggle.json`

### 3\. Accept Competition Rules (Crucial)

Before the API allows you to download files, you must accept the competition rules on the website:

1.  Navigate to the specific competition page on Kaggle.
2.  Click on the **"Rules"** tab.
3.  Scroll to the bottom and click **"I Understand and Accept"**.
4.  *If you do not do this, the API will return a `403 Forbidden` error.*

The exceptions thrown inside of `src/download_data.py` will point you to the URLs.

### 4\. Download Data

Once set up, run the following command to download the data to the current directory:

```bash
uv run src/download_data.py
```

If you use this research, please cite:
```bibtex
@article{yourname2024,
  title={Your Research Title},
  author={Your Name and Collaborators},
  journal={Journal Name},
  year={2024}
}
```

## License

[Choose appropriate license - MIT, Apache 2.0, GPL, etc.]

## Contact

- **Author**: Your Name (your.email@domain.com)
- **Institution**: Your Institution
.