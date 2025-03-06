# Automated Content Moderation System

## Text and Image Content Moderator

This project provides a simple web application to classify text and images as either **Appropriate** or **Inappropriate**. It leverages:

- A fine-tuned **BERT** model for text classification.
- A **Hugging Face NSFW Image Detection** model for image classification.

The Flask-based web interface allows users to enter text or upload an image, then displays the classification results along with probability scores. The UI dynamically colors the results:
- **Green** for **Appropriate** content.
- **Red** for **Inappropriate** content.

![UI Screenshot](https://github.com/user-attachments/assets/3b997cf6-f12a-4470-b348-cb6a7a5ea9d7)

---

## Features

- **Text Classification:** Uses a fine-tuned BERT model to predict if input text is appropriate.
- **Image Classification:** Utilizes a Hugging Face NSFW model to detect if an image is appropriate.
- **Interactive Web Interface:** Built with Flask, HTML, CSS, and JavaScript for a user-friendly experience.
- **Visual Feedback:** Results are presented with color-coded indicators and progress bars that reflect the confidence levels of predictions.

---

## Installation

### Prerequisites

- Python 3.7+
- `pip` for package management
- (Optional) A virtual environment (e.g., using `venv` or `conda`)

### Steps to Install

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/Automated_Content_Moderation_System.git
   cd Automated_Content_Moderation_System
