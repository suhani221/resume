import sqlite3
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np

# Load SentenceTransformer model
model_name = "bert-base-nli-mean-tokens"
model = SentenceTransformer(model_name)

# Function to calculate matching score
def get_matching_score(jd, resume):
    jd_embedding = model.encode(jd, convert_to_tensor=True)
    resume_embedding = model.encode(resume, convert_to_tensor=True)
    
    # Calculate cosine similarity
    cos_scores = np.dot(jd_embedding, resume_embedding.T) / (np.linalg.norm(jd_embedding) * np.linalg.norm(resume_embedding))
    mean_score = np.mean(cos_scores)
    return mean_score

# Connect to SQLite database
conn = sqlite3.connect('resumes.db')

# Fetch data from 'resumes' table into DataFrame
df = pd.read_sql_query("SELECT name, email, phone_number, previous_job_history, education, skills FROM resumes", conn)

# Close connection
conn.close()

# Page layout
page = st.sidebar.radio("Navigate", ["Show Entire DataFrame", "Apply Filters", "Rank Candidates"])

# Show entire DataFrame
if page == "Show Entire DataFrame":
    st.title('Resume Data - Entire DataFrame')
    st.write(df)

# Apply filters
elif page == "Apply Filters":
    st.title('Resume Data - Apply Filters')

    # Get unique values for selected columns
    selected_columns = ['education', 'previous_job_history']
    unique_values = {}
    for column in selected_columns:
        unique_values[column] = df[column].unique()

    # Split skills into individual items
    all_skills = [skill.strip() for sublist in df['skills'].str.split(',') for skill in sublist if skill.strip()]
    unique_values['skills'] = sorted(set(all_skills))

    # Create filters based on unique values
    selected_filters = {}
    for column, values in unique_values.items():
        selected_filters[column] = st.multiselect(f"Filter by {column}", values)

    # Apply filters to the DataFrame
    filtered_df = df.copy()
    for column, values in selected_filters.items():
        if values:
            if column == 'skills':
                filtered_df = filtered_df[filtered_df['skills'].apply(lambda x: any(skill.strip() in values for skill in x.split(',')))]
            else:
                filtered_df = filtered_df[filtered_df[column].isin(values)]

    st.write(filtered_df)

# Rank candidates
elif page == "Rank Candidates":
    st.title('Rank Candidates')

    # Input job description
    job_description = st.text_area("Enter Job Description")

    # Rank candidates based on job description
    if job_description:
        scores = []
        for idx, row in df.iterrows():
            # Get matching scores for each section
            job_history_score = get_matching_score(job_description, row['previous_job_history'])
            education_score = get_matching_score(job_description, row['education'])
            skills_score = get_matching_score(job_description, row['skills'])

            # Combine scores with optional weights for each section
            total_score = (0.4 * job_history_score) + (0.3 * education_score) + (0.3 * skills_score)

            scores.append((row['name'], total_score))

        # Sort candidates based on total score
        ranked_candidates = sorted(scores, key=lambda x: x[1], reverse=True)

        # Display ranked candidates
        st.write("Ranked Candidates:")
        for idx, (candidate, score) in enumerate(ranked_candidates, start=1):
            st.write(f"{idx}. {candidate} - Score: {score:.2f}")
