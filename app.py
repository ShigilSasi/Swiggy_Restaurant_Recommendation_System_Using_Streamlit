import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Global chart size (SAME FOR ALL)
CHART_SIZE = (6, 4)

# -----------------------------------
# Load Data
# -----------------------------------
@st.cache_data
def load_data():
    cleaned = pd.read_csv("cleaned_df.csv")
    encoded = pd.read_pickle("final_df.pkl")
    return cleaned, encoded

cleaned_df, final_df = load_data()

# Safety
cleaned_df = cleaned_df.reset_index(drop=True)
final_df = final_df.reset_index(drop=True).fillna(0)

# -----------------------------------
# UI
# -----------------------------------
st.set_page_config(page_title="🍽 Restaurant Recommender", layout="wide")
st.title("Swiggy Restaurant Recommendation System")
st.markdown("Find the **best restaurants** by selecting a city and cuisine.")

# -----------------------------------
# Sidebar
# -----------------------------------
st.sidebar.header("Filters")

cities = sorted(cleaned_df["city"].dropna().astype(str).unique())
selected_city = st.sidebar.selectbox("Select City", cities)

city_df = cleaned_df[cleaned_df["city"] == selected_city]

city_cuisines = set()
for val in city_df["cuisine"].dropna():
    for c in val.split(","):
        city_cuisines.add(c.strip())

selected_cuisine = st.sidebar.selectbox("Select Cuisine", sorted(city_cuisines))
top_k = st.sidebar.slider("Number of Restaurants", 5, 20, 10)

# -----------------------------------
# Recommendation Engine
# -----------------------------------
def recommend_by_city_and_cuisine(city, cuisine, k=10):

    col_name = f"cuisine_{cuisine}"
    if col_name not in final_df.columns:
        return None

    city_indices = cleaned_df[cleaned_df["city"] == city].index
    city_vectors = final_df.loc[city_indices]

    query = np.zeros(final_df.shape[1])
    query[final_df.columns.get_loc(col_name)] = 1

    similarities = cosine_similarity([query], city_vectors)[0]

    top_indices = similarities.argsort()[-200:][::-1]
    selected_indices = city_indices[top_indices]

    results = cleaned_df.loc[selected_indices].copy()
    results["similarity"] = similarities[top_indices]

    results = results[results["cuisine"].str.contains(cuisine, case=False, na=False)]

    results = results.sort_values(
        by=["rating", "rating_count", "similarity"],
        ascending=[False, False, False]
    )

    return results.head(k)

# -----------------------------------
# Run Recommendation
# -----------------------------------
if st.sidebar.button("Recommend"):

    results = recommend_by_city_and_cuisine(selected_city, selected_cuisine, top_k)

    if results is None or results.empty:
        st.warning("No restaurants found.")
    else:
        st.success(f"Top {len(results)} {selected_cuisine} Restaurants in {selected_city}")

        st.dataframe(
            results[["name", "city", "rating", "rating_count", "cost", "cuisine"]],
            use_container_width=True
        )

        # -----------------------------------
        # Visualizations
        # -----------------------------------
        st.subheader("Restaurant Insights")

        col1, col2 = st.columns(2)

        # Ratings Chart
        with col1:
            fig, ax = plt.subplots(figsize=CHART_SIZE)
            sns.barplot(data=results, x="rating", y="name", ax=ax)
            ax.set_title("Restaurant Ratings")
            ax.set_xlabel("Rating")
            ax.set_ylabel("")
            st.pyplot(fig)

        # Cost Distribution
        with col2:
            fig, ax = plt.subplots(figsize=CHART_SIZE)
            sns.histplot(results["cost"], bins=8, kde=True, ax=ax)
            ax.set_title("Cost Distribution")
            ax.set_xlabel("Cost for Two")
            ax.legend(["Cost"])
            st.pyplot(fig)

        col3, col4 = st.columns(2)

# Footer
st.markdown("---")

