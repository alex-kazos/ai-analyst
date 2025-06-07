import os
import pandas as pd
import json
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

# Load environment variables from .env if present (Docker passes them directly)
load_dotenv(override=True)

# Configure OpenAI API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.warning("No OpenAI API key found. Please set OPENAI_API_KEY in your .env file or environment.")
org_id = os.getenv("OPENAI_ORG_ID")
base_url = os.getenv("OPENAI_BASE_URL")

# Initialize OpenAI client
client = None
skip_api_calls = False
try:
    if api_key:
        logger.info("OpenAI API key found, attempting to initialize client")
        client_kwargs = {"api_key": api_key}
        if org_id:
            client_kwargs["organization"] = org_id
            logger.info("OpenAI organization ID set successfully")
        if base_url:
            client_kwargs["base_url"] = base_url
        client = openai.OpenAI(**client_kwargs)
        logger.info("OpenAI client initialized successfully")
    else:
        logger.error("Failed to load OpenAI API key from environment")
        skip_api_calls = True
except Exception as e:
    logger.error(f"Error initializing OpenAI client: {e}")
    logger.info("Will use mock analysis instead")
    skip_api_calls = True

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
        "dataset_description": f"This dataset contains {num_rows} records with {num_cols} variables including {', '.join(numeric_cols[:3]) if numeric_cols else 'no numeric columns'}. The data provides insights into business performance metrics and operational indicators.",
        "key_insights": insights,
        "visualizations": visualizations,
        "correlations": correlations,
        "recommendations": recommendations,
        "analysis_type": analysis_type,
        "model_used": "Mock AI (OpenAI unavailable)"
    }
    
    return mock_result

def perform_ai_analysis(df, analysis_type=None, query=None):
    global skip_api_calls
    """
    Perform AI analysis on a dataframe:
    - Use OpenAI for insights/recommendations if available, fallback to local if not.
    - Always return two visualizations (distribution and value counts for selected column).
    - Always return numeric_columns and selected_column.
    - Also return column type information for improved data statistics display.
    - Generate dataset description using OpenAI API
    """
    import numpy as np
    import pandas as pd
    import json
    result = {
        'key_insights': [],
        'recommendations': [],
        'visualizations': [],
        'numeric_columns': [],
        'selected_column': None,
        'column_info': {},  # Added column info dictionary
        'dataset_description': ''  # Added dataset description field
    }
    # Populate column_info with data types for each column
    for col in df.columns:
        result['column_info'][col] = {
            'dtype': str(df[col].dtype),
            'unique_count': df[col].nunique(),
            'missing_count': df[col].isna().sum()
        }
        
    # Create histograms for numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Add columns with numeric-looking names that might be stored as strings
    numeric_pattern = ['number', 'num', 'count', 'qty', 'price', 'amount', 'sales', 'order', 'id', 'revenue', 'profit']
    potential_numeric_cols = []
    
    # Check column names for numeric patterns
    for col in df.columns:
        col_lower = col.lower()
        if any(term in col_lower for term in numeric_pattern) and col not in numeric_cols:
            potential_numeric_cols.append(col)
    
    # Try converting potential numeric columns
    for col in potential_numeric_cols:
        try:
            # If column has numeric data but stored as string, convert and include it
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            # If at least 80% of values can be converted successfully, treat as numeric
            if numeric_series.notna().mean() >= 0.8:
                # Store converted values
                df[col] = numeric_series
                numeric_cols.append(col)
        except Exception as e:
            logger.warning(f"Error converting column {col}: {e}")
            pass  # Skip if conversion fails
    
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
        # Value counts bar chart - Improved formatting
        value_counts = df[col].value_counts().sort_index().head(20)
        
        # Format labels based on data type
        if pd.api.types.is_numeric_dtype(df[col]):
            # Format numeric labels to be more readable
            formatted_labels = []
            for idx in value_counts.index:
                if isinstance(idx, (int, float)):
                    # Use commas for thousands, limit decimal places
                    if idx == int(idx):
                        formatted_labels.append(f"{int(idx):,}")  # No decimals for integers
                    else:
                        formatted_labels.append(f"{idx:,.2f}")  # 2 decimals for floats
                else:
                    formatted_labels.append(str(idx))
        else:
            # For non-numeric columns, convert to string but truncate if too long
            formatted_labels = [str(idx)[:20] + '...' if len(str(idx)) > 20 else str(idx) for idx in value_counts.index]
        
        col_viz['value_counts'] = {
            'title': f'Most Common Values - {col}',
            'type': 'bar',
            'labels': formatted_labels,
            'data': value_counts.values.tolist(),
            'background_color': 'rgba(255, 206, 86, 0.7)',
            'border_color': 'rgba(255, 206, 86, 1)',
            'x_axis_label': f'{col} Values',
            'y_axis_label': 'Frequency'
        }
        
        # Histogram generation with improved handling of different column types
        try:
            # Check if column originally had a string dtype
            original_dtype = df[col].dtype
            if original_dtype == 'object':
                # For columns that were converted from string to numeric
                numeric_values = pd.to_numeric(df[col].dropna(), errors='coerce').dropna()
                if len(numeric_values) > 0:
                    hist, bins = np.histogram(numeric_values, bins=10)
                else:
                    # Fallback if conversion fails
                    hist, bins = np.array([0]), np.array([0, 1])
            else:
                # Standard numeric columns
                hist, bins = np.histogram(df[col].dropna(), bins=10)
        except Exception as e:
            logger.warning(f"Error creating histogram for {col}: {e}")
            # Safe fallback
            hist, bins = np.array([0]), np.array([0, 1])
        
        # Create more readable bin labels
        bin_labels = []
        for i in range(len(bins)-1):
            # Format with comma separators and fewer decimals
            if bins[i] == int(bins[i]) and bins[i+1] == int(bins[i+1]):
                # Integer ranges
                bin_labels.append(f"{int(bins[i]):,} - {int(bins[i+1]):,}")
            else:
                # Float ranges - limit decimals based on magnitude
                if abs(bins[i+1] - bins[i]) > 10:
                    # Larger ranges - no decimals needed
                    bin_labels.append(f"{bins[i]:.0f} - {bins[i+1]:.0f}")
                elif abs(bins[i+1] - bins[i]) > 1:
                    # Medium ranges - 1 decimal
                    bin_labels.append(f"{bins[i]:.1f} - {bins[i+1]:.1f}")
                else:
                    # Small ranges - 2 decimals
                    bin_labels.append(f"{bins[i]:.2f} - {bins[i+1]:.2f}")
        
        col_viz['histogram'] = {
            'title': f'Distribution of {col}',
            'type': 'bar',
            'labels': bin_labels,
            'data': hist.tolist(),
            'background_color': 'rgba(54, 162, 235, 0.7)',
            'border_color': 'rgba(54, 162, 235, 1)',
            'x_axis_label': f'{col} Ranges',
            'y_axis_label': 'Frequency'
        }
        # Scatter plot (index vs value, if numeric)
        if pd.api.types.is_numeric_dtype(df[col]):
            # Instead of using all rows, sample a manageable number to improve readability
            # and performance, especially for large datasets
            max_points = 100  # Limit points for better visualization
            
            if len(df) > max_points:
                # Use systematic sampling for more even distribution
                sample_interval = len(df) // max_points
                sample_indices = list(range(0, len(df), sample_interval))[:max_points]
                sample_df = df.iloc[sample_indices]
            else:
                sample_df = df
                
            scatter_x = sample_df.index.tolist()
            scatter_y = sample_df[col].dropna().tolist()
            
            # Create point labels for hover info (if there's a name/id column)
            point_labels = []
            # Look for likely label columns (id, name, etc.)
            potential_label_cols = [c for c in df.columns if any(term in c.lower() for term in ['id', 'name', 'key', 'title'])]
            
            if potential_label_cols and len(scatter_y) > 0:
                label_col = potential_label_cols[0]  # Use the first one found
                for idx in sample_indices if len(df) > max_points else df.index:
                    if idx < len(df) and pd.notna(df.loc[idx, col]):
                        label = f"Row {idx}: {col} = {df.loc[idx, col]:,.2f}"
                        if label_col in df.columns:
                            label += f", {label_col} = {df.loc[idx, label_col]}"
                        point_labels.append(label)
            else:
                # Simple labels with row number and value
                point_labels = [f"Row {idx}: {val:,.2f}" for idx, val in zip(scatter_x, scatter_y)]
                
            col_viz['scatter'] = {
                'title': f'Data Points - {col} Values',
                'type': 'scatter',
                'labels': scatter_x,  # X-axis is row index
                'data': scatter_y,     # Y-axis is column value
                'point_labels': point_labels,  # For tooltip/hover information
                'background_color': 'rgba(75, 192, 192, 0.7)',
                'border_color': 'rgba(75, 192, 192, 1)',
                'x_axis_label': 'Row Index',
                'y_axis_label': col
            }
        result['visualizations'][col] = col_viz

    # Check if OpenAI client is available
    if 'client' not in globals():
        logger.info("OpenAI client not available, falling back to basic insights")
        # Fallback local insights
        result['key_insights'].append(f"Showing distribution and value counts for column: {selected_column}.")
        result['recommendations'].append("No AI recommendations available.")
        return result
        
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
                result['dataset_description'] = cached.get('dataset_description', 'This dataset contains various metrics and indicators that have been analyzed to identify key patterns and trends.')
                cache_hit = True
                logger.info("Using cached AI insights")
        except Exception as e:
            logger.error(f"Failed to load OpenAI insights cache: {e}")
    
    if not cache_hit and client and not skip_api_calls:
        try:
            logger.info("Generating new insights with OpenAI API")
            sample_data = df.head(5).to_string()
            data_info = df.describe().to_string()
            column_info = "\n".join([f"{col} ({df[col].dtype}): {df[col].nunique()} unique values" for col in df.columns])
            system_prompt = "You are an expert data analyst. Given a dataset, generate a concise dataset description, 3 key insights, and 3 recommendations. Return a JSON with keys: dataset_description, key_insights, recommendations."
            user_prompt = f"Sample data:\n{sample_data}\n\nData summary:\n{data_info}\n\nColumns:\n{column_info}"

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            result_text = response.choices[0].message.content.strip()
            logger.info(result_text)
            logger.info(f"Received OpenAI response with length: {len(result_text)}")
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
            ai_json = json.loads(result_text)
            result['key_insights'] = ai_json.get('key_insights', [])
            result['recommendations'] = ai_json.get('recommendations', [])
            result['dataset_description'] = ai_json.get('dataset_description', 'This dataset contains various metrics and indicators that have been analyzed to identify key patterns and trends.')
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'key_insights': result['key_insights'],
                    'recommendations': result['recommendations'],
                    'dataset_description': result['dataset_description']
                }, f)
        except Exception as e:
            logger.error(f"OpenAI analysis failed: {e}")
            skip_api_calls = True
            mock_results = generate_mock_analysis(df, analysis_type, query)
            result['key_insights'] = mock_results['key_insights']
            result['recommendations'] = mock_results['recommendations']
            result['dataset_description'] = mock_results.get('dataset_description', 'This dataset contains various metrics and indicators that have been analyzed.')
    return result
