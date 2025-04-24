# import libaraies
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("udemy_courses.csv")

# Cleaning the dataset
df['content_duration'] = df['content_duration'].fillna(method='ffill')
df.drop_duplicates(inplace=True)
df['price'] = df['price'].replace('Free', 0).astype(int)
df['published_timestamp'] = pd.to_datetime(df['published_timestamp'], errors='coerce')

#Exploratory Data analysis
print("Summary of dataset:\n")
print(df.info())
print("Summary Statistics:\n")
print(df.describe())

print(" Dataset Shape:", df.shape)
print("\n Column Names:\n", df.columns.tolist())
print("\n Data Types:\n", df.dtypes)
print("\n Missing Values:\n", df.isnull().sum())
print("\n Duplicate Rows:", df.duplicated().sum())

#correlaation
correlation_matrix = df.corr(numeric_only=True)
sns.set(style="white", font_scale=1.1)
plt.figure(figsize=(10, 7))
sns.heatmap(
    correlation_matrix,
    annot=True,
    linewidths=0.5,
)
plt.title("Correlation Heatmap of Numeric Features", fontsize=14, weight='bold')
plt.tight_layout()
plt.show()

#outliers
numeric_cols = df.select_dtypes(include=[np.number]).columns
z_scores = df[numeric_cols].apply(zscore)
z_threshold = 3
outliers_by_column = {}

for col in numeric_cols:
    outliers_count = (np.abs(z_scores[col]) > z_threshold).sum()
    if outliers_count > 0:
        outliers_by_column[col] = outliers_count

print("Outliers detected using Z-score method:")
for col, count in outliers_by_column.items():
    print(f"{col}: {count} outliers")

# Objective:- 1 Free vs Paid Courses by Subject
free_paid_subject = df.groupby(['subject', 'is_paid']).size().unstack().fillna(0)
free_paid_subject.plot(kind='bar', figsize=(10, 6), stacked=True, colormap='Set2')
plt.title(' Free vs Paid Courses by Subject')
plt.xlabel('Subject')
plt.ylabel('Number of Courses')
plt.xticks(rotation=45)
plt.legend(title='Is Paid')
plt.tight_layout()
plt.show()

# objective 2:-Optimal Course Duration
duration_vs_subscribers = df.groupby('content_duration')['num_subscribers'].sum().sort_index()

plt.figure(figsize=(10,6))
plt.plot(duration_vs_subscribers.index, duration_vs_subscribers.values, color='green', linewidth=2)
plt.title('Number of Subscribers vs Course Duration')
plt.xlabel('Course Duration (in hours)')
plt.ylabel('Number of Subscribers')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

#objective 3:- Most Popular Course Titles
top_courses = df[['course_title', 'num_subscribers']].sort_values(by='num_subscribers', ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(y='course_title', x='num_subscribers', data=top_courses, palette='viridis')
plt.title(' Top 10 Most Subscribed Courses')
plt.xlabel('Number of Subscribers')
plt.ylabel('Course Title')
plt.tight_layout()
plt.show()


#objective 4:- Course Reviews by Difficulty Level
filtered_df = df[df['level'] != 'All Levels']
reviews_by_level = filtered_df.groupby('level')['num_reviews'].sum()
labels = ['Beginner Level', 'Intermediate Level', 'Expert Level']
skyblue_colors = ['#87CEEB', '#ADD8E6', '#B0E0E6']  
plt.figure(figsize=(8,6))
wedges, texts, autotexts = plt.pie(
    reviews_by_level,
    labels=labels,
    autopct='%1.1f%%',
    startangle=140,
    colors=skyblue_colors,
    wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
    textprops={'fontsize': 12, 'color': 'black'}
)
plt.title('Review Distribution by Course Level', fontsize=14, weight='bold', pad=30)
plt.axis('equal')
plt.tight_layout()
plt.show()

# Trends in Course Creation Over Time
df['publish_year'] = pd.to_datetime(df['published_timestamp']).dt.year
yearly_counts = df['publish_year'].value_counts().sort_index()
full_years = pd.Series(index=range(df['publish_year'].min(), df['publish_year'].max() + 1), dtype=int)
yearly_counts_full = full_years.add(yearly_counts, fill_value=0)
plt.figure(figsize=(10, 6))
plt.plot(yearly_counts_full.index, yearly_counts_full.values, marker='o', color='darkorange', linewidth=2)
plt.title(' Course Upload Trend by Year', fontsize=14)
plt.xlabel('Year')
plt.ylabel('Number of Courses Published')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# Courses Published by Day of the Week
df['published_day'] = pd.to_datetime(df['published_timestamp']).dt.day_name()
day_counts = df['published_day'].value_counts().reindex([
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
])
plt.figure(figsize=(8, 5))
sns.barplot(x=day_counts.index, y=day_counts.values, palette='pastel')
plt.title(' Courses Published by Day of the Week')
plt.xlabel('Day of Week')
plt.ylabel('Number of Courses')
plt.tight_layout()
plt.show()

