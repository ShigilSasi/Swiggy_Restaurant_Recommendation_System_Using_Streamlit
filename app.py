import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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
# Streamlit UI
# -----------------------------------
st.set_page_config(page_title="🍽 Restaurant Recommender", layout="wide")

st.title("🍕 Swiggy Restaurant Recommendation System")
st.markdown("Find the **best restaurants** by selecting a city and cuisine.")

# -----------------------------------
# Sidebar Filters
# -----------------------------------
st.sidebar.header("🔍 Filters")

# City
cities = sorted(cleaned_df["city"].dropna().astype(str).unique().tolist())
selected_city = st.sidebar.selectbox("Select City", cities)

# Get cuisines only for selected city
city_df = cleaned_df[cleaned_df["city"] == selected_city]

city_cuisines = set()
for val in city_df["cuisine"].dropna():
    for c in val.split(","):
        city_cuisines.add(c.strip())

city_cuisines = sorted(city_cuisines)

selected_cuisine = st.sidebar.selectbox("Select Cuisine", city_cuisines)

top_k = st.sidebar.slider("Number of Restaurants", 5, 20, 10)

# -----------------------------------
# Recommendation Engine
# -----------------------------------
def recommend_by_city_and_cuisine(city, cuisine, k=10):

    col_name = f"cuisine_{cuisine}"

    if col_name not in final_df.columns:
        return None

    # Get restaurants in selected city
    city_indices = cleaned_df[cleaned_df["city"] == city].index

    # Subset ML vectors
    city_vectors = final_df.loc[city_indices]

    # Create cuisine query
    query = np.zeros(final_df.shape[1])
    query[final_df.columns.get_loc(col_name)] = 1

    # Cosine similarity
    similarities = cosine_similarity([query], city_vectors)[0]

    # Larger candidate pool
    top_indices = similarities.argsort()[-200:][::-1]
    selected_indices = city_indices[top_indices]

    results = cleaned_df.loc[selected_indices].copy()
    results["similarity"] = similarities[top_indices]

    # HARD filter by cuisine
    results = results[results["cuisine"].str.contains(cuisine, case=False, na=False)]

    # Rank best restaurants
    results = results.sort_values(
        by=["rating", "rating_count", "similarity"],
        ascending=[False, False, False]
    )

    return results.head(k)

# -----------------------------------
# Run Recommendation
# -----------------------------------
if st.sidebar.button("🔎 Recommend"):

    results = recommend_by_city_and_cuisine(selected_city, selected_cuisine, top_k)

    if results is None or results.empty:
        st.warning("No restaurants found.")
    else:
        st.success(f"Top {len(results)} {selected_cuisine} Restaurants in {selected_city}")

        st.dataframe(
            results[["name", "city", "rating", "rating_count", "cost", "cuisine"]],
            use_container_width=True
        )

st.markdown("---")
st.markdown("Built with ❤️ using Machine Learning & Streamlit")
