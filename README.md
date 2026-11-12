# Candidate to Job Matching Pipeline

This project provides an automated matching pipeline to align job applicants with open job descriptions based on skill overlap, experience, and custom rules.

## 📂 Project Structure
- `candidate_profiles.csv`: Raw candidate profiles detailing their IDs, experience level, portfolio type, and lists of skills (separated by `|`).
- `job_descriptions.csv`: Standardized open roles specifying requirements, IDs, and key technical skills (space-separated).
- `main.py`: The executable pipeline script that normalizes text, maps skills dynamically, ranks matching profiles, and exports the findings.
- `results.csv`: The finalized output containing the top 5 qualified candidates ranked for each role.

## ⚙️ Matching Logic
1. **Skill Standardization**: To reconcile differences in delimiters (candidates use `|` while job specifications use spaces), skills are normalized, stripped of whitespace, lowercased, and stored in sets.
2. **Similarity Scoring**: Calculates intersection size over total requirements to yield a match percentage (0% - 100%).
3. **Ranking Priority**: Candidates matching a job are sorted first by their **match percentage** (descending) and subsequently by their **years of experience** (descending).
4. **Selection**: Retrieves exactly the top 5 matches per job code.
