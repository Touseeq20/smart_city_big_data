import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io

class TrafficVisualizer:
    def __init__(self, df):
        self.df = df
        
    def plot_hourly_traffic(self):
        """Average traffic volume by hour."""
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=self.df, x='hour', y='traffic_volume', hue='day_of_week', palette='viridis', errorbar=None, ax=ax)
        ax.set_title("Average Hourly Traffic Volume by Day of Week")
        ax.set_ylabel("Traffic Volume")
        ax.set_xlabel("Hour of Day")
        ax.grid(True, linestyle='--', alpha=0.7)
        return fig
    
    def plot_weather_impact(self):
        """Traffic volume distribution by weather."""
        fig, ax = plt.subplots(figsize=(12, 6))
        # Top 10 weather descriptions by frequency for readability
        top_weather = self.df['weather_description'].value_counts().nlargest(10).index
        sns.boxplot(data=self.df[self.df['weather_description'].isin(top_weather)], 
                    x='traffic_volume', y='weather_description', hue='weather_description', legend=False, ax=ax, palette="coolwarm")
        ax.set_title("Traffic Volume Distribution by Top Weather Conditions")
        return fig

    def plot_anomalies(self):
        """Scatter plot of Anomalies vs Normal traffic."""
        if 'is_anomaly' not in self.df.columns: return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        # Sample for performance if too large
        plot_data = self.df.sample(n=min(5000, len(self.df)), random_state=42)
        
        sns.scatterplot(data=plot_data, x='hour', y='traffic_volume', hue='is_anomaly', 
                        style='is_anomaly', palette={False: 'blue', True: 'red'}, alpha=0.6, ax=ax)
        ax.set_title("Detected Traffic Anomalies (Sampled)")
        return fig
    
    def plot_cluster_segments(self):
        if 'cluster' not in self.df.columns: return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=self.df.sample(n=min(5000, len(self.df)), random_state=42), 
                        x='hour', y='traffic_volume', hue='cluster', palette='tab10', alpha=0.5, ax=ax)
        ax.set_title("Traffic Clusters: Volume vs Hour")
        return fig
