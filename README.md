## Swiggy Restaurant Recommendation System

A Machine Learning–based Restaurant Recommendation System built using Cosine Similarity, Scikit-Learn, and Streamlit that helps users find the best restaurants based on City and Cuisine.

## Project Overview

This project uses content-based filtering to recommend restaurants by analyzing their:

    Cuisine type

    Ratings

    Popularity

    Cost

Users select a city and a cuisine, and the system suggests the top-rated and most relevant restaurants in that city.

The entire system is deployed as a Streamlit Web App.

## How It Works

1. Data Preprocessing

    Categorical columns (like cuisine, city) are One-Hot Encoded

    Numerical columns (rating, cost, rating_count) are scaled

    Final ML dataset → final_df.pkl

    Original readable dataset → cleaned_df.csv

2. Vector Similarity

    Each restaurant is represented as a numerical vector

    When a user selects a cuisine, a query vector is created

    Cosine Similarity finds restaurants most similar to the query

3. Filtering

Only restaurants from the selected city

Only restaurants matching the selected cuisine

4. Ranking
Results are sorted by:

    Rating

    Number of Ratings (Popularity)

    Similarity Score

## Web App Features

✔ City selection
✔ Cuisine selection
✔ Top-K restaurant recommendation
✔ Rating-based ranking
✔ Visual insights:

    Ratings bar chart

    Cost distribution

    Rating vs popularity

    Cuisine composition


## Installation

Clone the Repository

git clone https://github.com/yourusername/swiggy-recommender.git
cd swiggy-recommender

Install Dependencies

pip install streamlit pandas numpy scikit-learn matplotlib seaborn

Run the Application

streamlit run app.py

## Technologies Used
| Technology          | Purpose           |
| ------------------- | ----------------- |
| Python              | Core programming  |
| Pandas, NumPy       | Data processing   |
| Scikit-Learn        | Cosine Similarity |
| Streamlit           | Web app           |
| Matplotlib, Seaborn | Visualizations    |

