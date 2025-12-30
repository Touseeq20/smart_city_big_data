import json
import os

NOTEBOOK_PATH = os.path.join("notebooks", "exploratory_analysis.ipynb")

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Smart City Big Data: Exploratory Data Analysis (EDA)\n",
    "**Researcher:** Muhammad Touseeq\n",
    "\n",
    "This notebook documents the data exploration and mining phase of the Smart City Traffic Analysis project. We investigate the UCI Metro Traffic dataset to understand underlying patterns before modeling."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import os\n",
    "\n",
    "# Load Processed Data\n",
    "DATA_PATH = os.path.join(\"..\", \"data\", \"processed\", \"traffic_processed.csv\")\n",
    "if not os.path.exists(DATA_PATH):\n",
    "    print(\"Data not found! Please run preprocessing.py first.\")\n",
    "else:\n",
    "    df = pd.read_csv(DATA_PATH)\n",
    "    df['date_time'] = pd.to_datetime(df['date_time'])\n",
    "    print(\"Data Loaded Successfully. Shape:\", df.shape)\n",
    "    display(df.head())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Distribution of Traffic Volume\n",
    "We analyze the target variable to check for skewness and typical volume ranges."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(12, 6))\n",
    "sns.histplot(df['traffic_volume'], bins=50, kde=True, color='teal')\n",
    "plt.title('Distribution of Hourly Traffic Volume')\n",
    "plt.xlabel('Traffic Volume (Cars/Hour)')\n",
    "plt.ylabel('Frequency')\n",
    "plt.grid(True, linestyle='--', alpha=0.5)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Temporal Analysis (Time-Series Patterns)\n",
    "Traffic is highly dependent on time factors. We examine the 'Rush Hour' phenomenon."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(14, 6))\n",
    "sns.lineplot(data=df, x='hour', y='traffic_volume', hue='day_of_week', palette='viridis', errorbar=None)\n",
    "plt.title('Average Traffic Volume by Hour of Day (0=Mon, 6=Sun)')\n",
    "plt.xlabel('Hour of Day')\n",
    "plt.ylabel('Average Volume')\n",
    "plt.xticks(range(0, 24))\n",
    "plt.legend(title='Day of Week')\n",
    "plt.grid(True)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Observation:**\n",
    "- **Weekdays (0-4):** Clear double-peak pattern (Morning ~7-8 AM, Evening ~4-5 PM).\n",
    "- **Weekends (5-6):** Single broad peak around midday, significantly lower volume in mornings."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Environmental Impact Analysis\n",
    "Does weather affect traffic? We look at the correlation between weather attributes and volume."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(10, 8))\n",
    "corr_cols = ['traffic_volume', 'temp', 'rain_1h', 'snow_1h', 'clouds_all', 'hour']\n",
    "corr_matrix = df[corr_cols].corr()\n",
    "\n",
    "sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=\".2f\", linewidths=0.5)\n",
    "plt.title('Correlation Heatmap: Traffic vs. Weather')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Anomaly Detection (Unsupervised)\n",
    "We used Isolation Forest in our pipeline. Here we visualize outliers with high/low volume relative to the time."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Simple visual inspection of potential outliers\n",
    "plt.figure(figsize=(12, 6))\n",
    "sns.scatterplot(data=df.sample(2000), x='hour', y='traffic_volume', hue='weather_main', palette='tab10', alpha=0.6)\n",
    "plt.title('Traffic Volume vs Hour (Colored by Weather)')\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

if not os.path.exists("notebooks"):
    os.makedirs("notebooks")

with open(NOTEBOOK_PATH, 'w') as f:
    json.dump(notebook_content, f, indent=1)
    
print(f"Professional Notebook generated at {NOTEBOOK_PATH}")
