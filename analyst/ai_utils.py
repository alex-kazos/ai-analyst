import os
import pandas as pd
import json
import openai
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure OpenAI - use Client for the latest API version
api_key = os.getenv("OPENAI_API_KEY")

# OpenAI client initialization
try:
    if api_key:
        logger.info("OpenAI API key loaded successfully")
        # Initialize without any proxy settings
        import inspect
        # Check the parameters the OpenAI constructor accepts
        init_params = inspect.signature(openai.OpenAI.__init__).parameters
        # Only pass the API key parameter
        client_kwargs = {"api_key": api_key}
        client = openai.OpenAI(**client_kwargs)
        logger.info("OpenAI client initialized successfully")
    else:
        logger.error("Failed to load OpenAI API key from environment")
        raise ValueError("OpenAI API key not found in environment variables")
except Exception as e:
    logger.error(f"Error initializing OpenAI client: {e}")
    # We'll continue execution and use mock analysis when needed

def generate_mock_analysis(df, analysis_type=None, query=None):
    """
    Generate mock AI analysis results for demonstration purposes
    when OpenAI API is not available
    """
    logger.info("Using mock AI analysis implementation")
    
    # Get basic statistics from the dataframe
    num_rows, num_cols = df.shape
    column_names = df.columns.tolist()
    
    # Create sample column pairs for correlations
    correlations = []
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if len(numeric_cols) >= 2:
        for i in range(min(3, len(numeric_cols)-1)):
            correlations.append({
                "var1": numeric_cols[i],
                "var2": numeric_cols[i+1],
                "strength": "0.75",
                "description": f"Moderate to strong correlation between {numeric_cols[i]} and {numeric_cols[i+1]}."
            })
    
    # Create visualizations suggestions
    visualizations = []
    if len(numeric_cols) >= 1:
        # Bar chart suggestion
        visualizations.append({
            "title": f"Distribution of {numeric_cols[0]}",
            "type": "bar",
            "x_axis": column_names[0] if column_names else "Category",
            "y_axis": numeric_cols[0] if numeric_cols else "Value"
        })
        
        # If we have at least 2 numeric columns, suggest a scatter plot
        if len(numeric_cols) >= 2:
            visualizations.append({
                "title": f"Relationship between {numeric_cols[0]} and {numeric_cols[1]}",
                "type": "scatter",
                "x_axis": numeric_cols[0],
                "y_axis": numeric_cols[1]
            })
        
        # Add a line chart if we have enough columns
        if len(numeric_cols) >= 3:
            visualizations.append({
                "title": f"Trend of {numeric_cols[2]} over time",
                "type": "line",
                "x_axis": "Time",
                "y_axis": numeric_cols[2]
            })
    
    # Analyze column types
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()
    
    # Generate insights based on data characteristics
    insights = [
        f"The dataset contains {num_rows} rows and {num_cols} columns.",
        f"Key columns for analysis include: {', '.join(column_names[:3])}.",
        f"Numeric columns: {', '.join(numeric_cols) if numeric_cols else 'None'}.",
        f"Text columns: {', '.join(text_cols) if text_cols else 'None'}.",
        f"Date columns: {', '.join(date_cols) if date_cols else 'None'}."
    ]
    # Add missing values insight
    missing_report = []
    for col in df.columns:
        pct_missing = df[col].isnull().mean()*100
        if pct_missing > 0:
            missing_report.append(f"{col} ({pct_missing:.1f}% missing)")
    if missing_report:
        insights.append(f"Columns with missing values: {', '.join(missing_report)}.")
    # Outlier detection for numeric columns
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower) | (df[col] > upper)][col]
        if not outliers.empty:
            insights.append(f"Potential outliers detected in {col}.")
    if numeric_cols:
        # Add some insight about the range of a numeric column
        col = numeric_cols[0]
        min_val = df[col].min()
        max_val = df[col].max()
        avg_val = df[col].mean()
        insights.append(f"The {col} ranges from {min_val:.2f} to {max_val:.2f} with an average of {avg_val:.2f}.")
    # Generate recommendations
    recommendations = [
        "Consider running a more detailed statistical analysis on the key numeric variables.",
        "Explore the relationships between variables with correlation analysis.",
        "Clean any missing or outlier values for more accurate results."
    ]
    
    if query:
        recommendations.append(f"To answer your specific question about '{query}', additional targeted analysis may be needed.")
    
    # Create the mock result
    mock_result = {
        "summary": f"Analysis of your dataset with {num_rows} records and {num_cols} variables shows several patterns and insights for consideration.",
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
    import numpy as np
    import pandas as pd
    import json
    result = {
        'key_insights': [],
        'recommendations': [],
        'visualizations': [],
        'numeric_columns': [],
        'selected_column': None
    }
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

    # Try OpenAI for insights/recommendations (only once for the dataset), with caching
    import hashlib, os
    cache_dir = os.path.join(os.path.dirname(__file__), '.cache')
    os.makedirs(cache_dir, exist_ok=True)
    # Create a cache key based on dataset shape, columns, and head
    df_hash = hashlib.md5((str(df.shape) + str(list(df.columns)) + df.head(10).to_csv(index=False)).encode('utf-8')).hexdigest()
    cache_file = os.path.join(cache_dir, f"insights_{df_hash}.json")
    cache_hit = False
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached = json.load(f)
                result['key_insights'] = cached.get('key_insights', [])
                result['recommendations'] = cached.get('recommendations', [])
                cache_hit = True
        except Exception as e:
            logger.error(f"Failed to load OpenAI insights cache: {e}")
    if not cache_hit and 'client' in globals():
        try:
            sample_data = df.head(5).to_string()
            data_info = df.describe().to_string()
            column_info = "\n".join([f"{col} ({df[col].dtype}): {df[col].nunique()} unique values" for col in df.columns])
            system_prompt = "You are an expert data analyst. Given a dataset, generate 3 key insights and 3 recommendations. Return a JSON with keys: key_insights, recommendations."
            user_prompt = f"Sample data:\n{sample_data}\n\nData summary:\n{data_info}\n\nColumns:\n{column_info}"
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            result_text = response.choices[0].message.content.strip()
            # Remove code block markers if present
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
            ai_json = json.loads(result_text)
            result['key_insights'] = ai_json.get('key_insights', [])
            result['recommendations'] = ai_json.get('recommendations', [])
            # Save to cache
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump({'key_insights': result['key_insights'], 'recommendations': result['recommendations']}, f)
        except Exception as e:
            logger.error(f"OpenAI analysis failed: {e}")
            if not result['key_insights']:
                result['key_insights'].append(f"Showing distribution and value counts for column: {selected_column}.")
            if not result['recommendations']:
                result['recommendations'].append("No AI recommendations available.")
    elif not cache_hit:
        # Fallback local insights
        result['key_insights'].append(f"Showing distribution and value counts for column: {selected_column}.")
        result['recommendations'].append("No AI recommendations available.")
    return result

    if not 'client' in globals():
        logger.info("OpenAI client not available, using mock analysis")
        return generate_mock_analysis(df, analysis_type, query)
    # Prepare sample data for the prompt
    sample_data = df.head(5).to_string()
    data_info = df.describe().to_string()
    column_info = "\n".join([f"{col} ({df[col].dtype}): {df[col].nunique()} unique values" 
                          for col in df.columns])
    
    # Create prompt for OpenAI based on analysis type
    if analysis_type == "quick_ai":
        system_prompt = (
            "You are an expert data analyst AI assistant. Your job is to analyze data and provide insights. "
            "Generate a comprehensive analysis of the data including: "
            "1. Summary statistics and key observations \n"
            "2. Trends and patterns \n"
            "3. Correlations between variables \n"
            "4. Actionable insights \n"
            "5. Recommendations for further analysis\n\n"
            "Format your response as a JSON object with the following structure:\n"
            "{ \n"
            "  \"summary\": \"Overall summary of the data\",\n"
            "  \"key_insights\": [\"insight 1\", \"insight 2\", ...],\n"
            "  \"visualizations\": [\n"
            "    {\n"
            "      \"title\": \"Chart title\",\n"
            "      \"type\": \"bar|line|scatter|pie\",\n"
            "      \"x_axis\": \"column_name\",\n"
            "      \"y_axis\": \"column_name\"\n"
            "    }\n"
            "  ],\n"
            "  \"correlations\": [{\"var1\": \"column1\", \"var2\": \"column2\", \"strength\": \"value\", \"description\": \"explanation\"}],\n"
            "  \"recommendations\": [\"recommendation 1\", \"recommendation 2\", ...]\n"
            "}\n"
        )
        
        user_prompt = f"Please analyze this dataset:\n\nColumn Information:\n{column_info}\n\nSample Data:\n{sample_data}\n\nSummary Statistics:\n{data_info}\n\n"
        
        if query:
            user_prompt += f"Additional Instructions: {query}\n\n"
    
    else:
        # Generate prompt for other analysis types (clustering, regression, etc.)
        system_prompt = (
            "You are an expert data analyst AI assistant specializing in statistical analysis, "
            f"machine learning, and {analysis_type} techniques. "
            "Generate a detailed analysis of the data and provide insights in JSON format."
        )
        
        user_prompt = (f"Perform a {analysis_type} analysis on this dataset:\n\n"
                      f"Column Information:\n{column_info}\n\n"
                      f"Sample Data:\n{sample_data}\n\n"
                      f"Summary Statistics:\n{data_info}\n\n")
        
        if query:
            user_prompt += f"Additional Instructions: {query}\n\n"
    
    # Call OpenAI API
    try:
        logger.info(f"Sending request to OpenAI with prompt length: {len(user_prompt)}")
        
        # Use client to create chat completions
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        logger.info("Received response from OpenAI API")
        
        # Extract and parse the JSON response
        result_text = response.choices[0].message.content
        import re
        # Remove code block markers if present
        if result_text.strip().startswith("```"):
            # Remove triple backticks and optional 'json' after them
            result_text = re.sub(r"^```(?:json)?", "", result_text.strip(), flags=re.IGNORECASE).strip()
            result_text = re.sub(r"```$", "", result_text).strip()
        try:
            result = json.loads(result_text)
        except Exception as e:
            logger.error(f"Failed to parse OpenAI response as JSON: {e}")
            logger.error(f"OpenAI raw response: {result_text}")
            return {
                "error": "Failed to parse OpenAI response as JSON.",
                "summary": "The AI did not return valid JSON. Please try again.",
                "key_insights": [],
                "recommendations": [],
                "visualizations": []
            }
        # Add metadata
        result["analysis_type"] = analysis_type
        result["model_used"] = "gpt-4"
        
        return result
    
    except Exception as e:
        logger.error(f"Error in OpenAI API call: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Check if it's a billing error and use mock analysis instead
        error_msg = str(e).lower()
        if 'billing' in error_msg or 'account is not active' in error_msg:
            logger.info("OpenAI account inactive. Using mock analysis instead.")
            return generate_mock_analysis(df, analysis_type, query)
        else:
            return {
                "error": str(e),
                "summary": "An error occurred during AI analysis.",
                "key_insights": ["Analysis failed due to an error."],
                "recommendations": ["Please try again with a different dataset or query."],
                "visualizations": []
            }
