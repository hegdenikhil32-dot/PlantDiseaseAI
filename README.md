PlantDiseaseAI

AI-powered tomato leaf disease detection and treatment recommendations.

Features

- Tomato leaf disease classification
- Detection of Early Blight, Late Blight, Leaf Mold, and Healthy leaves
- Confidence scores and quick recommendations
- Optional Gemini-powered detailed treatment plans

Run the application

Install the required dependencies using the project's existing dependency file, then run:

python -m streamlit run app/app.py

Gemini API configuration

Create a ".env" file in the project root and add:

GEMINI_API_KEY=your_api_key_here

Keep your API key private. Never commit ".env" to a public repository.

Disclaimer

Predictions are educational decision-support results and should be verified with a qualified plant-health professional.