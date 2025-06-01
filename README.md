# AI Analyst

A Django-based AI data analysis platform designed to analyze data, generate insights, and visualize results through various AI-powered analysis techniques.

![alt text](image.png)

## Features

- **Data Source Management**: Upload CSV, Excel, JSON, Parquet, Feather, and other file formats or connect to databases (MySQL, PostgreSQL, Supabase)
- **AI-Powered Analysis**: Run various analyses including clustering, classification, regression, and enhanced statistical analysis
- **Customizable Clustering**: Choose between different clustering algorithms (K-Means, DBSCAN) and configure their parameters
- **Natural Language Interface**: Ask questions about your data in plain English
- **Data Management**: Delete data sources with confirmation and remove individual analyses
- **User-Friendly Statistical Analysis**: Improved visualizations and highlighted insights
- **Dynamic Visualizations**: Automatically generate appropriate visualizations based on data analysis

## Installation

### Prerequisites

- Python 3.8+
- Django 5.2.1
- OpenAI API key

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/alex-kazos/ai-analyst.git
   cd ai-analyst
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables in a `.env` file:
   ```
   DEBUG=True
   SECRET_KEY=your-secret-key
   DATABASE_URL=your-database-url
   OPENAI_API_KEY=your-openai-api-key
   ```

5. Run migrations:
   ```bash
   python manage.py migrate
   ```

6. Create a superuser (admin):
   ```bash
   python manage.py createsuperuser
   ```

7. Start the development server:
   ```bash
   python manage.py runserver
   ```

8. Visit `http://127.0.0.1:8000/` in your browser

## Usage

### Uploading Data

1. Navigate to the Data Sources section
2. Click "Add Data Source"
3. Upload your CSV, Excel, or JSON file, or connect to a database
4. Your data will be processed and made available for analysis

### Running Analysis

1. Select a data source
2. Choose an analysis type (clustering, classification, etc.)
3. Configure analysis parameters
4. Run the analysis
5. View the results, including AI-generated insights and visualizations

### Asking Questions About Your Data

1. Select a data source
2. Click on the "Ask" button
3. Enter your question in natural language
4. View the AI-generated response with relevant visualizations

## Technologies Used

- **Backend**: Django 5.2.1, Python, pandas 2.2.3, numpy 2.2.6, scikit-learn 1.6.1
- **Frontend**: HTML, CSS, JavaScript, Bootstrap, django-crispy-forms 2.4
- **Visualization**: matplotlib 3.10.3, plotly 6.1.2
- **AI/ML**: OpenAI API 1.82.1, langchain 0.3.25
- **Database**: MySQL, PostgreSQL, SQLAlchemy 2.0.41, Supabase 2.15.2
- **File Formats**: pyarrow 14.0.2, fastparquet 2023.10.1, openpyxl 3.1.2, tables 3.9.1 (HDF5), pyorc 0.8.0 (ORC)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Django](https://www.djangoproject.com/)
- [Chart.js](https://www.chartjs.org/)
- [OpenAI](https://openai.com/)
- [Bootstrap](https://getbootstrap.com/)
