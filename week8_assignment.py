# 1.1. Load Data

import pandas as pd

# Load the metadata.csv file
df = pd.read_csv("datasets/metadata.csv")
# Show first 5 records
print(df.head())

# 1.2. Explore Data Structure

# Shape of dataset
print("Shape:", df.shape)
# Data types and non-null counts
print(df.info())
# Column names
print(df.columns.tolist())

# 1.3. Missing Values & Basic Stats

# Count of missing values per column
print(df.isnull().sum())
# Basic stats (mostly for numeric columns)
print(df.describe(include='all'))

# 2.1. Identify & Handle Missing Data

# Identify columns with high % of missing data
missing_percent = df.isnull().mean().sort_values(ascending=False)
print(missing_percent)
# Optional: drop columns with too much missing data
df_clean = df.drop(columns=missing_percent[missing_percent > 0.5].index)

# 2.2. Convert Dates & Extract Year

# Convert publish_time to datetime
df_clean['publish_time'] = pd.to_datetime(df_clean['publish_time'], errors='coerce')
# Extract year
df_clean['year'] = df_clean['publish_time'].dt.year

# 2.3. Feature Engineering: Abstract Word Count

# Word count from abstract
df_clean['abstract_word_count'] = df_clean['abstract'].fillna("").apply(lambda x: len(x.split()))

# 3.1. Papers per Year

import matplotlib.pyplot as plt

year_counts = df_clean['year'].value_counts().sort_index()
plt.figure(figsize=(10, 6))
plt.bar(year_counts.index, year_counts.values)
plt.title("Publications per Year")
plt.xlabel("Year")
plt.ylabel("Number of Papers")
plt.show()

# 3.2. Top Journals

top_journals = df_clean['journal'].value_counts().head(10)

top_journals.plot(kind='barh', figsize=(10, 6), color='skyblue')
plt.title("Top 10 Journals")
plt.xlabel("Number of Publications")
plt.ylabel("Journal")
plt.gca().invert_yaxis()
plt.show()

# 3.3. Word Frequency in Titles

from collections import Counter
import re

# Clean titles and count words
titles = df_clean['title'].dropna().tolist()
words = [word.lower() for title in titles for word in re.findall(r'\b\w+\b', title)]
common_words = Counter(words).most_common(20)

# Plot
words, counts = zip(*common_words)
plt.figure(figsize=(12, 6))
plt.bar(words, counts)
plt.xticks(rotation=45)
plt.title("Most Common Words in Titles")
plt.show()

# 3.4. Word Cloud

from wordcloud import WordCloud

text = ' '.join(titles)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(15, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Titles")
plt.show()


# 3.5. Source Distribution
df_clean['source_x'].value_counts().head(10).plot(kind='bar', figsize=(10, 6))
plt.title("Top Sources")
plt.xlabel("Source")
plt.ylabel("Paper Count")
plt.show()

# Part 4: Streamlit Application

# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("CORD-19 Data Explorer")
st.write("Simple exploration of COVID-19 research papers")

@st.cache_data
def load_data():
    df = pd.read_csv("metadata.csv")
    df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
    df['year'] = df['publish_time'].dt.year
    return df

df = load_data()

# Filter by year
years = df['year'].dropna().astype(int).unique()
years.sort()
start_year, end_year = st.slider("Select Year Range", int(min(years)), int(max(years)), (2020, 2021))

filtered_df = df[(df['year'] >= start_year) & (df['year'] <= end_year)]

# Show sample data
st.subheader("Sample Data")
st.write(filtered_df.head())

# Visualization: Publications per Year
st.subheader("Publications per Year")
year_counts = filtered_df['year'].value_counts().sort_index()
fig, ax = plt.subplots()
ax.bar(year_counts.index, year_counts.values)
ax.set_title("Publications per Year")
st.pyplot(fig)

# Run the app:
# streamlit run app.py



