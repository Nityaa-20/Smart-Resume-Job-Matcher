#pip install scikit-learn nltk
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# --- Sample Resume Text ---
resume_text = """
John Doe
Email: johndoe@example.com
Skills: Python, Data Analysis, Machine Learning, Pandas, NumPy
Experience: 2 years as Data Analyst at ABC Corp
Education: B.Tech in Computer Science
"""

# --- Sample Job Description Text ---
job_description = """
We are looking for a Data Analyst with strong experience in Python, SQL, and Machine Learning.
The candidate should be able to work with large datasets using Pandas and visualization tools like Matplotlib or Tableau.
"""

# --- Clean and Preprocess Text ---
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    return ' '.join(tokens)

resume_clean = preprocess(resume_text)
job_clean = preprocess(job_description)

# --- TF-IDF + Cosine Similarity ---
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([resume_clean, job_clean])
similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100

# --- Skill Suggestion ---
resume_words = set(resume_clean.split())
job_words = set(job_clean.split())
missing_skills = job_words - resume_words

# --- Results ---
print(f"✅ ATS Match Score: {similarity_score:.2f}%")

if similarity_score >= 70:
    print("🎉 Great! Your resume is a good match for the job.")
else:
    print("⚠️ Your resume could be improved for this job.")

print("\n🧠 Suggested Skills to Add:")
print(", ".join(missing_skills))
