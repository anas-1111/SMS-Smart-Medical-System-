---
noteId: "33b2f420a29811f0a851631ea7e3a6c6"
tags: []

---


# Smart Medicine Information Assistant

## Overview

This Streamlit web application provides detailed bilingual (English + Arabic) reports about medicines. Users can either upload a photo of a medicine package or type the medicine name, and the app fetches verified information from Drugs.com or generates a summary using Gemini AI if the medicine is not found.

## Features

-   **Image Upload:** Users can upload an image of a medicine package, and the app extracts the medicine name using OCR.
-   **Text Input:** Users can directly type the medicine name.
-   **Bilingual Reports:** Provides medicine information in both English and Arabic.
-   **Data Source:** Fetches verified information from Drugs.com.
-   **AI-Powered Summary:** If the medicine is not found on Drugs.com, the app generates a summary using Gemini AI.

## Dependencies

-   streamlit
-   google-generativeai
-   beautifulsoup4
-   fake-useragent
-   requests
-   tqdm
-   Pillow (PIL)
-   ipywidgets

To install all dependencies, run:
