import os
import pandas as pd
import json
import openai
import logging
import numpy as np
import hashlib
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure OpenAI API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
org_id = os.getenv("OPENAI_ORG_ID")

# Initialize OpenAI client
client = None

try:
    if api_key:
        logger.info("OpenAI API key loaded successfully")
        
        # Set the API key for the legacy openai library
        openai.api_key = api_key
        
        # Set organization if available
        if org_id:
            openai.organization = org_id
            logger.info("OpenAI organization ID set successfully")
        
        # Initialize the client
        client_kwargs = {"api_key": api_key}
        
        # Add base URL if configured
        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url:
            client_kwargs["base_url"] = base_url
            
        client = openai.OpenAI(**client_kwargs)
        logger.info("OpenAI client initialized successfully")
    else:
        logger.error("Failed to load OpenAI API key from environment")
        raise ValueError("OpenAI API key not found in environment variables")
except Exception as e:
    logger.error(f"Error initializing OpenAI client: {e}")
    logger.info("Will continue execution and use mock analysis when needed")

def generate_mock_analysis(df, analysis_type=None, query=None):
    """
    Generate mock AI analysis results when OpenAI API is not available
    """
    logger.info("Using mock AI analysis implementation")
    
    # Get basic statistics from the dataframe
    num_rows, num_cols = df.shape
    column_names = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Create mock insights
    insights = [
        f"Dataset contains {num_rows} rows and {num_cols} columns.",
        f"There are {len(numeric_cols)} numeric columns available for analysis.",
    ]
    
    if numeric_cols:
        # Add some basic statistics for the first numeric column
        col = numeric_cols[0]
        mean_val = df[col].mean()
        median_val = df[col].median()
        insights.append(f"The average {col} is {mean_val:.2f} with a median of {median_val:.2f}.")
    
    # Create mock recommendations
    recommendations = [
        "Consider exploring the correlations between numeric columns.",
        "Analyze the distribution of key variables to identify patterns.",
        "Investigate any missing values in the dataset for potential data quality issues."
    ]
    
    # Create mock visualizations
    visualizations = []
    if numeric_cols:
        for col in numeric_cols[:2]:  # Just use the first two numeric columns
            visualizations.append({
                "title": f"Distribution of {col}",
                "type": "bar",
                "x_axis": col,
                "y_axis": "count"
            })
    
    # Create mock correlations
    correlations = []
    if len(numeric_cols) >= 2:
        correlations.append({
            "var1": numeric_cols[0],
            "var2": numeric_cols[1],
            "strength": 0.7,
            "description": f"Moderate positive correlation between {numeric_cols[0]} and {numeric_cols[1]}"
        })
    
    # Create the mock result
    mock_result = {
        "summary": f"Analysis of your dataset with {num_rows} records and {num_cols} variables.",
        "key_insights": insights,
        "visualizations": visualizations,
        "correlations": correlations,
        "recommendations": recommendations,
        "analysis_type": analysis_type,
        "model_used": "Mock AI (OpenAI unavailable)"
    }
    
    return mock_result

def perform_ai_analysis(df, analysis_type=None, query=None):
    """
    Perform AI analysis on a dataframe:
    - Use OpenAI for insights/recommendations if available, fallback to local if not.
    - Always return two visualizations (distribution and value counts for selected column).
    - Always return numeric_columns and selected_column.
    """
    result = {
        'key_insights': [],
        'recommendations': [],
        'visualizations': [],
        'numeric_columns': [],
        'selected_column': None
    }
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        result['key_insights'].append('No numeric columns available for visualization.')
        return result
    
    # Use query as selected column if valid, else fallback
    selected_column = query if query in numeric_cols else numeric_cols[0]
    result['numeric_columns'] = numeric_cols
    result['selected_column'] = selected_column

    # Visualizations for all numeric columns
    result['visualizations'] = {}
    for col in numeric_cols:
        col_viz = {}
        # Value counts bar chart
        value_counts = df[col].value_counts().sort_index().head(20)
        col_viz['value_counts'] = {
            'title': f'Value Counts of {col}',
            'type': 'bar',
            'labels': [str(idx) for idx in value_counts.index],
            'data': value_counts.values.tolist(),
            'background_color': 'rgba(255, 206, 86, 0.7)',
            'border_color': 'rgba(255, 206, 86, 1)'
        }
        
        # Histogram
        hist, bins = np.histogram(df[col].dropna(), bins=10)
        col_viz['histogram'] = {
            'title': f'Distribution of {col}',
            'type': 'bar',
            'labels': [f"{round(bins[i],2)}-{round(bins[i+1],2)}" for i in range(len(bins)-1)],
            'data': hist.tolist(),
            'background_color': 'rgba(54, 162, 235, 0.7)',
            'border_color': 'rgba(54, 162, 235, 1)'
        }
        
        # Scatter plot (index vs value, if numeric)
        if pd.api.types.is_numeric_dtype(df[col]):
            scatter_x = df[col].dropna().index.tolist()
            scatter_y = df[col].dropna().values.tolist()
            col_viz['scatter'] = {
                'title': f'Scatter Plot of {col}',
                'type': 'scatter',
                'labels': scatter_x,
                'data': scatter_y,
                'background_color': 'rgba(75, 192, 192, 0.7)',
                'border_color': 'rgba(75, 192, 192, 1)'
            }
            
        result['visualizations'][col] = col_viz

    # Check if OpenAI client is available
    if client is None:
        logger.info("OpenAI client not available, falling back to basic insights")
        result['key_insights'].append(f"Showing distribution and value counts for column: {selected_column}.")
        result['recommendations'].append("No AI recommendations available.")
        return result
        
    # Try OpenAI for insights/recommendations with caching
    cache_dir = os.path.join(os.path.dirname(__file__), '.cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create a cache key based on dataset shape, columns, and head
    df_hash = hashlib.md5((str(df.shape) + str(list(df.columns)) + df.head(10).to_csv(index=False)).encode('utf-8')).hexdigest()
    cache_file = os.path.join(cache_dir, f"insights_{df_hash}.json")
    
    # Check for cached results
    cache_hit = False
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached = json.load(f)
                result['key_insights'] = cached.get('key_insights', [])
                result['recommendations'] = cached.get('recommendations', [])
                cache_hit = True
                logger.info("Using cached AI insights")
        except Exception as e:
            logger.error(f"Failed to load OpenAI insights cache: {e}")
    
    # Generate new insights if needed
    if not cache_hit:
        try:
            logger.info("Generating new insights with OpenAI API")
            
            # Prepare dataset information
            sample_data = df.head(5).to_string()
            data_info = df.describe().to_string()
            column_info = "\n".join([f"{col} ({df[col].dtype}): {df[col].nunique()} unique values" for col in df.columns])
            
            # Create the system prompt
            system_prompt = "You are an expert data analyst. Given a dataset, generate 3 key insights and 3 recommendations. Return a JSON with keys: key_insights, recommendations."
            
            # Create the user prompt
            user_prompt = f"Sample data:\n{sample_data}\n\nData summary:\n{data_info}\n\nColumns:\n{column_info}"
            
            # Call the OpenAI API
            response = client.chat.completions.create(
                model="gpt-4o",  # Use newer model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}  # Request JSON response
            )
            
            # Process the response
            result_text = response.choices[0].message.content.strip()
            logger.info(f"Received OpenAI response with length: {len(result_text)}")
            
            # Remove code block markers if present
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
                
            # Parse the JSON response
            ai_json = json.loads(result_text)
            result['key_insights'] = ai_json.get('key_insights', [])
            result['recommendations'] = ai_json.get('recommendations', [])
            
            # Save to cache
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump({'key_insights': result['key_insights'], 'recommendations': result['recommendations']}, f)
        except Exception as e:
            logger.error(f"OpenAI analysis failed: {e}")
            # Fallback if API call fails
            result['key_insights'].append(f"Showing distribution and value counts for column: {selected_column}.")
            result['recommendations'].append("No AI recommendations available.")
    
    return result

# Simple test function to verify the file works
if __name__ == "__main__":
    # Create a small test dataframe
    test_df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50]
    })
    
    print("Testing OpenAI integration...")
    # Try to check if client is initialized
    if client is not None:
        print("OpenAI client initialized successfully")
    else:
        print("OpenAI client initialization failed, will use mock analysis")
    
    # Try a simple analysis
    result = perform_ai_analysis(test_df)
    print("Analysis result keys:", result.keys())
    print("Key insights:", result['key_insights'])
