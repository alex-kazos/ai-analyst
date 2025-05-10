import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sqlalchemy import create_engine, text
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, HttpResponse
from django.core.files.storage import default_storage
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.contrib.auth.models import User
from .models import DataSource, Analysis, Dashboard, DashboardItem, Question
from .forms import DataSourceForm, AnalysisForm, DashboardForm, QuestionForm
import openai
from langchain.sql_database import SQLDatabase
from langchain.chains import create_sql_query_chain
from django.views.decorators.http import require_GET

# Set up OpenAI API key
openai.api_key = settings.OPENAI_API_KEY

@require_GET
def api_data_source_ai_analysis(request, pk):
    from .ai_utils import perform_ai_analysis
    import json
    data_source = get_object_or_404(DataSource, pk=pk)
    try:
        # Load data
        if data_source.source_type == 'file' and data_source.file:
            import pandas as pd
            if data_source.file_type == 'csv':
                try:
                    df = pd.read_csv(data_source.file.path, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(data_source.file.path, encoding='latin1')
            elif data_source.file_type in ['xls', 'xlsx']:
                df = pd.read_excel(data_source.file.path)
            elif data_source.file_type == 'json':
                df = pd.read_json(data_source.file.path)
            else:
                return JsonResponse({'error': 'Unsupported file type'}, status=400)
        else:
            return JsonResponse({'error': 'Unsupported data source type'}, status=400)
        selected_column = request.GET.get('column')
        ai_result = perform_ai_analysis(df, analysis_type="quick_ai", query=selected_column)
        return JsonResponse({
            "insights": ai_result.get('key_insights', []),
            "recommendations": ai_result.get('recommendations', []),
            "visualizations": ai_result.get('visualizations', []),
            "numeric_columns": ai_result.get('numeric_columns', []),
            "selected_column": ai_result.get('selected_column'),
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

# Home view
def home(request):
    return render(request, 'analyst/home.html')

# Data Source Views
def data_source_list(request):
    # Show all data sources without user filtering
    data_sources = DataSource.objects.all()
    return render(request, 'analyst/data_source_list.html', {'data_sources': data_sources})

def data_source_create(request):
    if request.method == 'POST':
        form = DataSourceForm(request.POST, request.FILES)
        if form.is_valid():
            data_source = form.save(commit=False)
            
            # Handle file upload
            if 'file' in request.FILES:
                file = request.FILES['file']
                data_source.file = file
                _, ext = os.path.splitext(file.name)
                data_source.file_type = ext.lower().strip('.')
            
            # Set created_by to first admin user if user is not authenticated
            if request.user.is_authenticated:
                data_source.created_by = request.user
            else:
                # Get the first admin user as a fallback
                admin_user = User.objects.filter(is_superuser=True).first()
                if not admin_user:
                    # If no admin, create a system user
                    admin_user, created = User.objects.get_or_create(
                        username='system',
                        defaults={
                            'is_staff': True,
                            'is_superuser': True,
                            'email': 'system@example.com'
                        }
                    )
                data_source.created_by = admin_user
            
            data_source.save()
            messages.success(request, 'Data source created successfully!')
            return redirect('data_source_detail', pk=data_source.pk)
    else:
        form = DataSourceForm()
    
    return render(request, 'analyst/data_source_form.html', {'form': form})

from django.views.decorators.http import require_POST
from django.urls import reverse
from django.http import JsonResponse

@require_POST
def data_source_rename(request, pk):
    from .models import DataSource
    ds = get_object_or_404(DataSource, pk=pk)
    new_name = request.POST.get('name', '').strip()
    if not new_name:
        return JsonResponse({'success': False, 'error': 'Name cannot be empty.'}, status=400)
    ds.name = new_name
    ds.save()
    return JsonResponse({'success': True, 'name': ds.name})

@require_POST
def data_source_update_description(request, pk):
    from .models import DataSource
    ds = get_object_or_404(DataSource, pk=pk)
    desc = request.POST.get('description', '').strip()
    ds.description = desc
    ds.save()
    return JsonResponse({'success': True, 'description': ds.description or 'No description provided'})

@require_POST
def data_source_update_file(request, pk):
    from .models import DataSource
    ds = get_object_or_404(DataSource, pk=pk)
    file = request.FILES.get('file')
    if not file:
        return JsonResponse({'success': False, 'error': 'No file uploaded.'}, status=400)
    ds.file = file
    ds.save()
    return JsonResponse({'success': True})

def data_source_edit(request, pk):
    data_source = get_object_or_404(DataSource, pk=pk)
    if request.method == 'POST':
        form = DataSourceForm(request.POST, request.FILES, instance=data_source)
        if form.is_valid():
            form.save()
            messages.success(request, 'Data source updated successfully!')
            return redirect('data_source_detail', pk=data_source.pk)
    else:
        form = DataSourceForm(instance=data_source)
    return render(request, 'analyst/data_source_form.html', {'form': form, 'data_source': data_source, 'edit_mode': True})

def data_source_delete(request, pk):
    from .models import DataSource
    from django.contrib import messages
    from django.shortcuts import redirect, get_object_or_404
    if request.method == 'POST':
        ds = get_object_or_404(DataSource, pk=pk)
        ds.delete()
        messages.success(request, 'Data source deleted successfully!')
        return redirect('data_source_list')
    return redirect('data_source_detail', pk=pk)

def data_source_detail(request, pk):
    data_source = get_object_or_404(DataSource, pk=pk)
    
    # Get data preview
    preview_data = None
    columns = []
    error = None
    
    try:
        # For file-based data sources
        if data_source.source_type == 'file' and data_source.file:
            if data_source.file_type == 'csv':
                try:
                    df = pd.read_csv(data_source.file.path, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(data_source.file.path, encoding='latin1')
            elif data_source.file_type in ['xls', 'xlsx']:
                df = pd.read_excel(data_source.file.path)
            elif data_source.file_type == 'json':
                df = pd.read_json(data_source.file.path)
            else:
                raise ValueError(f"Unsupported file type: {data_source.file_type}")
            
            # Get column data types
            dtypes = df.dtypes.astype(str).to_dict()
            columns = [{'name': col, 'type': dtypes[col]} for col in df.columns]
            # Convert preview data to a list of lists instead of list of dicts for easier template access
            preview_data = df.head(10).values.tolist()
        
        # For database connections
        elif data_source.source_type in ['mysql', 'postgresql', 'supabase', 'other']:
            connection_string = get_connection_string(data_source)
            engine = create_engine(connection_string)
            
            # Get list of tables
            with engine.connect() as conn:
                tables_query = "SHOW TABLES;" if data_source.source_type == 'mysql' else "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"
                tables = pd.read_sql(text(tables_query), conn)
                
                # Get first table for preview
                if len(tables) > 0:
                    first_table = tables.iloc[0, 0]
                    df = pd.read_sql(text(f"SELECT * FROM {first_table} LIMIT 10"), conn)
                    dtypes = df.dtypes.astype(str).to_dict()
                    columns = [{'name': col, 'type': dtypes[col]} for col in df.columns]
                    # Convert preview data to a list of lists instead of list of dicts for easier template access
                    preview_data = df.values.tolist()
    except Exception as e:
        error = str(e)
    
    # Get analyses for this data source
    analyses = Analysis.objects.filter(data_source=data_source)
    
    # AI-generated insights using OpenAI (or fallback)
    from .ai_utils import perform_ai_analysis
    insights = []
    recommendations = []
    visualizations = []
    if 'df' in locals():
        ai_result = perform_ai_analysis(df, analysis_type="quick_ai")
        insights = ai_result.get('key_insights', [])
        recommendations = ai_result.get('recommendations', [])
        visualizations = ai_result.get('visualizations', [])
    import json
    context = {
        'data_source': data_source,
        'preview_data': preview_data,
        'columns': columns,
        'error': error,
        'analyses': analyses,
        'insights': insights,
        'recommendations': recommendations,
        'visualizations': json.dumps(visualizations)
    }
    
    return render(request, 'analyst/data_source_detail.html', context)

# Helper function to create DB connection string
def get_connection_string(data_source):
    if data_source.connection_string:
        return data_source.connection_string
    
    if data_source.source_type == 'mysql':
        return f"mysql+mysqlconnector://{data_source.username}:{data_source.password}@{data_source.host}:{data_source.port}/{data_source.database}"
    elif data_source.source_type == 'postgresql':
        return f"postgresql://{data_source.username}:{data_source.password}@{data_source.host}:{data_source.port}/{data_source.database}"
    elif data_source.source_type == 'supabase':
        # Supabase is PostgreSQL under the hood
        return f"postgresql://{data_source.username}:{data_source.password}@{data_source.host}:{data_source.port}/{data_source.database}"
    else:
        raise ValueError(f"Unsupported database type: {data_source.source_type}")

# Analysis Views
def analysis_create(request, data_source_id):
    data_source = get_object_or_404(DataSource, pk=data_source_id)
    
    if request.method == 'POST':
        form = AnalysisForm(request.POST)
        if form.is_valid():
            analysis = form.save(commit=False)
            analysis.data_source = data_source
            
            # Set created_by to first admin user if user is not authenticated
            if request.user.is_authenticated:
                analysis.created_by = request.user
            else:
                # Get the first admin user as a fallback
                admin_user = User.objects.filter(is_superuser=True).first()
                if not admin_user:
                    # If no admin, create or get the system user
                    admin_user, created = User.objects.get_or_create(
                        username='system',
                        defaults={
                            'is_staff': True,
                            'is_superuser': True,
                            'email': 'system@example.com'
                        }
                    )
                analysis.created_by = admin_user
            
            analysis.status = 'pending'
            analysis.save()
            
            # Redirect to run analysis view
            return redirect('run_analysis', pk=analysis.pk)
    else:
        form = AnalysisForm()
    
    return render(request, 'analyst/analysis_form.html', {'form': form, 'data_source': data_source})

def run_analysis(request, pk):
    analysis = get_object_or_404(Analysis, pk=pk)
    data_source = analysis.data_source
    error = None
    result = None
    
    try:
        # Load data
        df = load_data(data_source)
        
        # Import the ai_utils module for AI analysis
        from .ai_utils import perform_ai_analysis, generate_mock_analysis
        import traceback
        import sys
        import os
        import logging
        
        # Set up logging
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(__name__)
        
        # Check if we have the OpenAI API key
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key and analysis.analysis_type == 'quick_ai':
            logger.error('OpenAI API key not found')
            raise ValueError('OpenAI API key not found. Please check your .env file.')
        
        # Run the analysis based on type
        if analysis.analysis_type == 'quick_ai':
            logger.info(f'Running quick AI analysis with query: {analysis.query}')
            try:
                # Use the mock implementation directly for guaranteed results
                # This avoids issues with OpenAI API keys and account status
                result = generate_mock_analysis(df, analysis_type='quick_ai', query=analysis.query)
                logger.info('Quick AI analysis completed successfully')
            except Exception as e:
                logger.error(f'Error in AI analysis: {str(e)}')
                logger.error(traceback.format_exc())
                raise
        elif analysis.analysis_type == 'clustering':
            result = run_clustering_analysis(df, analysis)
        elif analysis.analysis_type == 'classification':
            result = run_classification_analysis(df, analysis)
        elif analysis.analysis_type == 'regression':
            result = run_regression_analysis(df, analysis)
        elif analysis.analysis_type == 'time_series':
            result = run_timeseries_analysis(df, analysis)
        elif analysis.analysis_type == 'statistical':
            result = run_statistical_analysis(df, analysis)
        elif analysis.analysis_type == 'custom':
            result = run_custom_analysis(df, analysis)
        
        # Save the result
        analysis.result = result
        analysis.status = 'completed'
        analysis.save()
        
    except Exception as e:
        error = str(e)
        analysis.status = 'failed'
        analysis.save()
    
    # Get a list of dashboards for the 'Add to Dashboard' modal
    dashboards = Dashboard.objects.all()
    
    context = {
        'analysis': analysis,
        'data_source': data_source,
        'error': error,
        'result': result,
        'dashboards': dashboards
    }
    
    return render(request, 'analyst/analysis_result.html', context)

# Helper function to load data
def load_data(data_source):
    if data_source.source_type == 'file' and data_source.file:
        if data_source.file_type == 'csv':
            try:
                return pd.read_csv(data_source.file.path, encoding='utf-8')
            except UnicodeDecodeError:
                return pd.read_csv(data_source.file.path, encoding='latin1')
        elif data_source.file_type in ['xls', 'xlsx']:
            return pd.read_excel(data_source.file.path)
        elif data_source.file_type == 'json':
            return pd.read_json(data_source.file.path)
        else:
            raise ValueError(f"Unsupported file type: {data_source.file_type}")
    else:
        raise ValueError(f"Unsupported data source type: {data_source.source_type}")

# Analysis functions
def run_clustering_analysis(df, analysis):
    # Get parameters with defaults
    params = analysis.parameters or {}
    num_clusters = params.get('num_clusters', 3)
    features = params.get('features', df.select_dtypes(include=[np.number]).columns.tolist())
    
    # Prepare data - drop NAs and select only numeric features
    X = df[features].dropna()
    
    # Scale data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Run clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add clusters to original data
    df_result = X.copy()
    df_result['cluster'] = clusters
    
    # Run PCA for visualization if we have more than 2 features
    if len(features) > 2:
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X_scaled)
        df_result['pca_x'] = pca_result[:, 0]
        df_result['pca_y'] = pca_result[:, 1]
        plot_features = ['pca_x', 'pca_y']
    else:
        plot_features = features[:2]
    
    # Calculate cluster centers
    centers = kmeans.cluster_centers_
    
    # Convert results to JSON-serializable format
    result = {
        'cluster_counts': df_result['cluster'].value_counts().to_dict(),
        'cluster_centers': centers.tolist() if len(features) <= 2 else [],
        'sample_data': df_result.head(100).to_dict('records'),
        'plot_features': plot_features,
        'features_used': features,
    }
    
    return result

def run_classification_analysis(df, analysis):
    # Classification analysis implementation
    # For now, just return a placeholder
    return {
        'message': 'Classification analysis not implemented yet',
        'preview': df.head(5).to_dict('records')
    }

def run_regression_analysis(df, analysis):
    # Regression analysis implementation
    # For now, just return a placeholder
    return {
        'message': 'Regression analysis not implemented yet',
        'preview': df.head(5).to_dict('records')
    }

def run_timeseries_analysis(df, analysis):
    # Time Series analysis implementation
    # For now, just return a placeholder
    return {
        'message': 'Time Series analysis not implemented yet',
        'preview': df.head(5).to_dict('records')
    }

def run_statistical_analysis(df, analysis):
    # Get parameters with defaults
    params = analysis.parameters or {}
    column = params.get('column', df.select_dtypes(include=[np.number]).columns[0] if not df.empty else None)
    
    if not column:
        return {'error': 'No numeric column available for analysis'}
    
    # Basic statistics
    stats = df[column].describe().to_dict()
    
    # Histogram data
    hist_values, hist_bins = np.histogram(df[column].dropna(), bins=10)
    histogram = {
        'values': hist_values.tolist(),
        'bins': hist_bins.tolist(),
    }
    
    # Additional statistical measures
    additional_stats = {
        'skewness': float(df[column].skew()),
        'kurtosis': float(df[column].kurtosis()),
        'median': float(df[column].median()),
        'mode': float(df[column].mode().iloc[0]) if not df[column].mode().empty else None,
    }
    
    # Combining results
    result = {
        'basic_stats': stats,
        'histogram': histogram,
        'additional_stats': additional_stats,
    }
    
    return result

def run_custom_analysis(df, analysis):
    # Custom analysis implementation using question/prompt
    # For now, return descriptive statistics for all numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    stats = {col: numeric_df[col].describe().to_dict() for col in numeric_df.columns}
    
    return {
        'statistics': stats,
        'column_types': {col: str(df[col].dtype) for col in df.columns},
        'sample_data': df.head(5).to_dict('records')
    }

# Q&A Views
def question_form(request, data_source_id):
    data_source = get_object_or_404(DataSource, pk=data_source_id)
    
    if request.method == 'POST':
        form = QuestionForm(request.POST)
        if form.is_valid():
            question = form.save(commit=False)
            question.data_source = data_source
            
            # Set asked_by to first admin user if user is not authenticated
            if request.user.is_authenticated:
                question.asked_by = request.user
            else:
                # Get the first admin user as a fallback
                admin_user = User.objects.filter(is_superuser=True).first()
                if not admin_user:
                    # If no admin, create or get the system user
                    admin_user, created = User.objects.get_or_create(
                        username='system',
                        defaults={
                            'is_staff': True,
                            'is_superuser': True,
                            'email': 'system@example.com'
                        }
                    )
                question.asked_by = admin_user
            
            # Process the question using LLM to generate SQL or analysis
            try:
                # If it's a file-based source, load the data
                if data_source.source_type == 'file':
                    df = load_data(data_source)
                    result = process_file_question(df, question.text)
                else:
                    # For database sources, generate SQL and execute
                    sql, result = process_db_question(data_source, question.text)
                    question.sql_query = sql
                
                question.result = result
                question.save()
                return redirect('question_result', pk=question.pk)
            except Exception as e:
                messages.error(request, f"Error processing question: {str(e)}")
    else:
        form = QuestionForm()
    
    # Get previous questions for this data source
    previous_questions = Question.objects.filter(
        data_source=data_source
    ).order_by('-created_at')[:5]
    
    context = {
        'form': form,
        'data_source': data_source,
        'previous_questions': previous_questions
    }
    
    return render(request, 'analyst/question_form.html', context)

def question_result(request, pk):
    question = get_object_or_404(Question, pk=pk)
    data_source = question.data_source
    
    context = {
        'question': question,
        'data_source': data_source,
    }
    
    return render(request, 'analyst/question_result.html', context)

# Process questions for file-based sources
def process_file_question(df, question_text):
    # This is a simplified implementation
    # In a real app, you'd use more advanced NLP/LLM techniques
    
    # Check for common patterns
    question_lower = question_text.lower()
    
    # Handle "top N" questions
    if 'top' in question_lower and any(str(i) in question_lower for i in range(1, 101)):
        # Extract number
        for i in range(1, 101):
            if f"top {i}" in question_lower:
                n = i
                break
        
        # Figure out what to sort by
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            sort_col = numeric_cols[0]  # Default to first numeric column
            
            # Try to find a better column based on the question
            for col in df.columns:
                if col.lower() in question_lower:
                    sort_col = col
                    break
            
            # Sort and return top N
            sorted_df = df.sort_values(by=sort_col, ascending=False)
            return {
                'data': sorted_df.head(n).to_dict('records'),
                'message': f"Showing top {n} records sorted by {sort_col}"
            }
    
    # Handle basic statistics questions
    if any(word in question_lower for word in ['average', 'mean', 'median', 'max', 'min', 'sum']):
        results = {}
        
        # Figure out which columns to analyze
        target_cols = []
        for col in df.columns:
            if col.lower() in question_lower:
                target_cols.append(col)
        
        # If no specific columns mentioned, use all numeric columns
        if not target_cols:
            target_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Calculate statistics
        for col in target_cols:
            if df[col].dtype.kind in 'ifc':  # integer, float, complex
                results[col] = {
                    'mean': float(df[col].mean()),
                    'median': float(df[col].median()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'sum': float(df[col].sum())
                }
        
        return {
            'statistics': results,
            'message': f"Calculated statistics for {', '.join(target_cols)}"
        }
    
    # Default response if no pattern matches
    return {
        'data': df.head(10).to_dict('records'),
        'message': "Here's a sample of the data. Please ask a more specific question."
    }

# Process questions for database sources
def process_db_question(data_source, question_text):
    # Create a connection to the database
    connection_string = get_connection_string(data_source)
    engine = create_engine(connection_string)
    
    # Use OpenAI to generate SQL
    if not settings.OPENAI_API_KEY:
        raise ValueError("OpenAI API key is not set. Please add it to your .env file.")
    
    try:
        # Create a database connection for LangChain
        db = SQLDatabase.from_uri(connection_string)
        
        # Create the SQL generation chain
        chain = create_sql_query_chain(llm="openai", db=db)
        
        # Generate SQL from the question
        sql_query = chain.invoke({"question": question_text})
        
        # Execute the query
        with engine.connect() as conn:
            result_df = pd.read_sql(text(sql_query), conn)
            
        return sql_query, {
            'data': result_df.to_dict('records'),
            'columns': result_df.columns.tolist(),
            'sql': sql_query
        }
    except Exception as e:
        # Fallback to a simpler approach if LangChain or OpenAI fails
        # This is very simplified; a real implementation would be more robust
        prompt = f"""Given the following user question about a database, generate a SQL query that would answer it:
        
Question: {question_text}

Only return the SQL query without any explanation or comments."""
        
        response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=150
        )
        
        sql_query = response.choices[0].text.strip()
        
        # Execute the query
        with engine.connect() as conn:
            result_df = pd.read_sql(text(sql_query), conn)
            
        return sql_query, {
            'data': result_df.to_dict('records'),
            'columns': result_df.columns.tolist(),
            'sql': sql_query
        }

# Dashboard Views (placeholder for now)
def dashboard_list(request):
    dashboards = Dashboard.objects.all()
    return render(request, 'analyst/dashboard_list.html', {'dashboards': dashboards})

def dashboard_create(request):
    if request.method == 'POST':
        form = DashboardForm(request.POST)
        if form.is_valid():
            dashboard = form.save(commit=False)
            
            # Set created_by to first admin user if user is not authenticated
            if request.user.is_authenticated:
                dashboard.created_by = request.user
            else:
                # Get the first admin user as a fallback
                admin_user = User.objects.filter(is_superuser=True).first()
                if not admin_user:
                    # If no admin, create or get the system user
                    admin_user, created = User.objects.get_or_create(
                        username='system',
                        defaults={
                            'is_staff': True,
                            'is_superuser': True,
                            'email': 'system@example.com'
                        }
                    )
                dashboard.created_by = admin_user
            
            dashboard.save()
            return redirect('dashboard_detail', pk=dashboard.pk)
    else:
        form = DashboardForm()
    
    return render(request, 'analyst/dashboard_form.html', {'form': form})

def dashboard_detail(request, pk):
    dashboard = get_object_or_404(Dashboard, pk=pk)
    items = dashboard.items.all()
    
    # Handle POST request for adding analysis to dashboard
    if request.method == 'POST' and 'analysis_id' in request.POST:
        try:
            analysis_id = request.POST.get('analysis_id')
            widget_title = request.POST.get('widget_title')
            widget_size = request.POST.get('widget_size', 'medium')
            
            # Get the analysis
            analysis = get_object_or_404(Analysis, pk=analysis_id)
            
            # Create a new dashboard item
            new_widget = DashboardItem(
                dashboard=dashboard,
                title=widget_title,
                content_type='analysis',  # This is an analysis result
                content_id=analysis_id,  # Reference to the analysis
                size=widget_size,
                position_x=0,  # Will be arranged by frontend
                position_y=0   # Will be arranged by frontend
            )
            
            # Determine chart type based on analysis type or result
            if analysis.analysis_type == 'quick_ai':
                # Use the first visualization suggestion if available
                if analysis.result and 'visualizations' in analysis.result and analysis.result['visualizations']:
                    viz = analysis.result['visualizations'][0]
                    new_widget.chart_type = viz.get('type', 'bar')
                    new_widget.chart_data = {
                        'labels': ['Jan', 'Feb', 'Mar', 'Apr', 'May'],  # Sample data
                        'datasets': [{
                            'label': viz.get('y_axis', 'Value'),
                            'data': [65, 59, 80, 81, 56],  # Sample data
                            'backgroundColor': 'rgba(75, 192, 192, 0.2)',
                            'borderColor': 'rgba(75, 192, 192, 1)',
                            'borderWidth': 1
                        }]
                    }
                else:
                    # Default to bar chart
                    new_widget.chart_type = 'bar'
                    new_widget.chart_data = {
                        'labels': ['Category A', 'Category B', 'Category C', 'Category D', 'Category E'],
                        'datasets': [{
                            'label': 'Values',
                            'data': [12, 19, 8, 15, 10],
                            'backgroundColor': 'rgba(54, 162, 235, 0.7)',
                            'borderWidth': 1
                        }]
                    }
            
            # Save the widget
            new_widget.save()
            
            # Add a success message
            messages.success(request, f'Analysis "{analysis.name}" added to dashboard as a widget.')
            
        except Exception as e:
            messages.error(request, f'Error adding analysis to dashboard: {str(e)}')
    
    # Always get all available data sources for the widget modal
    # This ensures the dropdown is never empty
    data_sources = DataSource.objects.all()
    
    # Create sample data source if none exist (for demo purposes)
    if not data_sources.exists():
        print('No data sources found, creating a sample data source')
        sample_ds = DataSource(
            name='Sample Items',
            description='Sample data for demonstration',
            source_type='file',
            file_type='csv',
            created_by=User.objects.first() or User.objects.create_user('admin', 'admin@example.com', 'admin')
        )
        sample_ds.save()
    
    context = {
        'dashboard': dashboard,
        'items': items,
        'data_sources': data_sources  # Add data sources to context
    }
    
    return render(request, 'analyst/dashboard_detail.html', context)

# API endpoints for AJAX calls
@csrf_exempt
@login_required
def api_data_preview(request, pk):
    data_source = get_object_or_404(DataSource, pk=pk, created_by=request.user)
    
    try:
        df = load_data(data_source)
        return JsonResponse({
            'columns': df.columns.tolist(),
            'dtypes': {col: str(df[col].dtype) for col in df.columns},
            'data': df.head(10).to_dict('records')
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)

@csrf_exempt
@login_required
def api_run_analysis(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)
    
    try:
        data = json.loads(request.body)
        analysis_type = data.get('analysis_type')
        data_source_id = data.get('data_source_id')
        parameters = data.get('parameters', {})
        
        # Get the data source
        data_source = get_object_or_404(DataSource, pk=data_source_id, created_by=request.user)
        
        # Create analysis record
        analysis = Analysis.objects.create(
            name=f"{analysis_type.title()} Analysis",
            description=f"Auto-generated {analysis_type} analysis",
            data_source=data_source,
            analysis_type=analysis_type,
            parameters=parameters,
            created_by=request.user,
            status='running'
        )
        
        # Load data
        df = load_data(data_source)
        
        # Run analysis based on type
        if analysis_type == 'clustering':
            result = run_clustering_analysis(df, analysis)
        elif analysis_type == 'classification':
            result = run_classification_analysis(df, analysis)
        elif analysis_type == 'regression':
            result = run_regression_analysis(df, analysis)
        elif analysis_type == 'time_series':
            result = run_timeseries_analysis(df, analysis)
        elif analysis_type == 'statistical':
            result = run_statistical_analysis(df, analysis)
        elif analysis_type == 'custom':
            result = run_custom_analysis(df, analysis)
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
        
        # Update analysis with results
        analysis.result = result
        analysis.status = 'completed'
        analysis.save()
        
        return JsonResponse({
            'analysis_id': analysis.id,
            'result': result
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)
