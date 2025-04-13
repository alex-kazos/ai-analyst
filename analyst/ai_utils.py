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
    
    # Generate insights based on data characteristics
    insights = [
        f"The dataset contains {num_rows} rows and {num_cols} columns.",
        f"Key columns for analysis include: {', '.join(column_names[:3])}.",
    ]
    
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
    Perform AI analysis on a dataframe using OpenAI's API or mock implementation
    
    Args:
        df: pandas DataFrame containing the data to analyze
        analysis_type: type of analysis to perform (clustering, classification, etc.)
        query: specific query or instructions for the analysis
        
    Returns:
        dict: Analysis results containing insights, charts, and summary
    """
    
    # Check if the OpenAI client was properly initialized
    # If there was an error during initialization, use mock analysis
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
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        logger.info("Received response from OpenAI API")
        
        # Extract and parse the JSON response
        result_text = response.choices[0].message.content
        result = json.loads(result_text)
        
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
