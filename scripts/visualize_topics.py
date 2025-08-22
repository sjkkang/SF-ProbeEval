#!/usr/bin/env python3
"""
Topic visualization script - matplotlib-based scatter plots
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# English font settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Load data
result_dir = Path("output/kmeans_20_topics")
vis_df = pd.read_csv(result_dir / "pca2d_points.csv")

print(f"Visualization data loaded: {len(vis_df)} points")

# 토픽별 색상 맵핑
colors = plt.cm.tab20(np.linspace(0, 1, 20))

# Graph creation
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Topic Modeling Results Visualization', fontsize=16, fontweight='bold')

# 1. Overall topic scatter plot
ax1 = axes[0, 0]
scatter = ax1.scatter(vis_df['x'], vis_df['y'], c=vis_df['topic'], 
                     cmap='tab20', alpha=0.6, s=1)
ax1.set_title('Overall Topic Distribution (PCA 2D)')
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.grid(True, alpha=0.3)

# 2. Topic document count bar chart
ax2 = axes[0, 1]
topic_counts = vis_df['topic'].value_counts().sort_index()
bars = ax2.bar(topic_counts.index, topic_counts.values, 
               color=[colors[i] for i in topic_counts.index])
ax2.set_title('Documents per Topic')
ax2.set_xlabel('Topic ID')
ax2.set_ylabel('Document Count')
ax2.grid(True, alpha=0.3)

# Display values
for bar, count in zip(bars, topic_counts.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
             str(count), ha='center', va='bottom', fontsize=8)

# 3. Major topics highlight
ax3 = axes[1, 0]
# Show only top 5 topics in color
top_topics = topic_counts.head(5).index
for topic in top_topics:
    topic_data = vis_df[vis_df['topic'] == topic]
    ax3.scatter(topic_data['x'], topic_data['y'], 
               label=f'Topic {topic} ({len(topic_data)} docs)', 
               alpha=0.7, s=2)

# Others in gray
other_data = vis_df[~vis_df['topic'].isin(top_topics)]
ax3.scatter(other_data['x'], other_data['y'], 
           color='lightgray', alpha=0.3, s=1, label='Others')

ax3.set_title('Major Topics Highlight (Top 5)')
ax3.set_xlabel('PC1')
ax3.set_ylabel('PC2')
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax3.grid(True, alpha=0.3)

# 4. Topic size distribution histogram
ax4 = axes[1, 1]
ax4.hist(topic_counts.values, bins=10, alpha=0.7, 
         color='skyblue', edgecolor='black')
ax4.set_title('Topic Size Distribution')
ax4.set_xlabel('Topic Size (Document Count)')
ax4.set_ylabel('Frequency')
ax4.grid(True, alpha=0.3)

# Statistics display
mean_size = topic_counts.mean()
std_size = topic_counts.std()
ax4.axvline(mean_size, color='red', linestyle='--', 
           label=f'Mean: {mean_size:.0f}')
ax4.axvline(mean_size + std_size, color='orange', linestyle='--', 
           label=f'Mean+1σ: {mean_size + std_size:.0f}')
ax4.legend()

plt.tight_layout()
plt.savefig(result_dir / "topic_visualization.png", dpi=300, bbox_inches='tight')
plt.show()

print(f"Visualization completed! Results saved to {result_dir}/topic_visualization.png")

# Topic statistics summary
print("\nTopic Statistics Summary:")
print(f"Total topics: {len(topic_counts)}")
print(f"Average topic size: {mean_size:.1f} documents")
print(f"Standard deviation: {std_size:.1f}")
print(f"Largest topic size: {topic_counts.max()} documents (Topic {topic_counts.idxmax()})")
print(f"Smallest topic size: {topic_counts.min()} documents (Topic {topic_counts.idxmin()})")

# Top 5 topics info
print("\nTop 5 Topics:")
for i, (topic, count) in enumerate(topic_counts.head(5).items(), 1):
    print(f"{i}. Topic {topic}: {count} documents")

# Save statistics to file
stats_report = []
stats_report.append("TOPIC VISUALIZATION STATISTICS")
stats_report.append("=" * 40)
stats_report.append(f"Total topics: {len(topic_counts)}")
stats_report.append(f"Average topic size: {mean_size:.1f} documents")
stats_report.append(f"Standard deviation: {std_size:.1f}")
stats_report.append(f"Largest topic size: {topic_counts.max()} documents (Topic {topic_counts.idxmax()})")
stats_report.append(f"Smallest topic size: {topic_counts.min()} documents (Topic {topic_counts.idxmin()})")
stats_report.append("")
stats_report.append("TOP 5 TOPICS:")
for i, (topic, count) in enumerate(topic_counts.head(5).items(), 1):
    stats_report.append(f"{i}. Topic {topic}: {count} documents")

with open(result_dir / "visualization_stats.txt", "w", encoding='utf-8') as f:
    f.write("\n".join(stats_report))

print(f"\nStatistics saved to {result_dir}/visualization_stats.txt")
