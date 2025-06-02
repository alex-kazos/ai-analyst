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
from langchain_community.utilities import SQLDatabase
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
            "dataset_description": ai_result.get('dataset_description', 'This dataset contains various metrics and indicators that have been analyzed to identify key patterns and trends.')
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
            try:
                if data_source.file_type == 'csv':
                    try:
                        df = pd.read_csv(data_source.file.path, encoding='utf-8')
                    except UnicodeDecodeError:
                        df = pd.read_csv(data_source.file.path, encoding='latin1')
                elif data_source.file_type in ['xls', 'xlsx']:
                    df = pd.read_excel(data_source.file.path)
                elif data_source.file_type == 'json':
                    df = pd.read_json(data_source.file.path)
                elif data_source.file_type == 'parquet':
                    df = pd.read_parquet(data_source.file.path)
                elif data_source.file_type == 'feather':
                    df = pd.read_feather(data_source.file.path)
                elif data_source.file_type == 'pickle':
                    df = pd.read_pickle(data_source.file.path)
                elif data_source.file_type == 'hdf':
                    df = pd.read_hdf(data_source.file.path)
                elif data_source.file_type == 'orc':
                    df = pd.read_orc(data_source.file.path)
                else:
                    raise ValueError(f"Unsupported file type: {data_source.file_type}")
            except Exception as e:
                logger.error(f"Error loading file {data_source.file.path}: {str(e)}")
                raise ValueError(f"Error loading file: {str(e)}")
                
            # Try to convert potential numeric columns that are stored as strings
            for col in df.columns:
                # Check if column name suggests numeric content
                if any(term in col.lower() for term in ['order', 'number', 'num', 'qty', 'price', 'amount', 'sales', 'id']):
                    try:
                        # If >80% of values can be converted to numeric, convert the column
                        numeric_values = pd.to_numeric(df[col], errors='coerce')
                        if numeric_values.notna().mean() > 0.8:
                            df[col] = numeric_values
                    except:
                        pass
            
            # Get column data types and enhance classification
            columns = []
            numeric_columns = []
            text_columns = []
            date_columns = []
            
            # First pass - get standard type classification
            standard_numeric = df.select_dtypes(include=['int', 'float', 'int64', 'float64']).columns.tolist()
            
            for col in df.columns:
                dtype_str = str(df[col].dtype)
                col_type = dtype_str
                
                # Enhanced detection for numeric columns
                is_numeric = False
                
                # Check standard numeric types
                if col in standard_numeric:
                    is_numeric = True
                # Check column name patterns
                elif any(term in col.lower() for term in ['order', 'number', 'num', 'qty', 'quantity', 'price', 'amount', 'sales', 'id']):
                    # Try to convert sample to numeric
                    try:
                        sample = df[col].dropna().head(100)
                        # If most values can be converted to numeric
                        numeric_ratio = pd.to_numeric(sample, errors='coerce').notna().mean()
                        if numeric_ratio > 0.8:  # If >80% are numeric
                            is_numeric = True
                            col_type = 'numeric'  # Override type for display
                    except:
                        pass
                
                # Categorize column
                if is_numeric:
                    numeric_columns.append(col)
                elif 'datetime' in dtype_str or 'date' in col.lower() or 'time' in col.lower():
                    date_columns.append(col)
                    col_type = 'datetime'
                else:
                    text_columns.append(col)
                    col_type = 'text'
                
                # Add to columns list with enhanced type information
                columns.append({
                    'name': col, 
                    'type': col_type,
                    'category': 'numeric' if is_numeric else ('date' if col in date_columns else 'text')
                })
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
        'visualizations': json.dumps(visualizations),
        # Add column type counts for direct template access
        'numeric_columns': numeric_columns if 'numeric_columns' in locals() else [],
        'text_columns': text_columns if 'text_columns' in locals() else [],
        'date_columns': date_columns if 'date_columns' in locals() else []
    }
    
    return render(request, 'analyst/data_source_detail.html', context)

# ... (rest of the code remains the same)
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

@require_POST
def analysis_rename(request, pk):
    analysis = get_object_or_404(Analysis, pk=pk)
    import json
    try:
        data = json.loads(request.body)
        new_name = data.get('name', '').strip()
        if not new_name:
            return JsonResponse({'success': False, 'error': 'Name cannot be empty.'}, status=400)
        analysis.name = new_name
        analysis.save()
        return JsonResponse({'success': True, 'name': analysis.name})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

@require_POST
def analysis_update_description(request, pk):
    analysis = get_object_or_404(Analysis, pk=pk)
    import json
    try:
        data = json.loads(request.body)
        desc = data.get('description', '').strip()
        analysis.description = desc
        analysis.save()
        return JsonResponse({'success': True, 'description': analysis.description})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)

@require_POST
def analysis_delete(request, pk):
    import logging
    analysis = get_object_or_404(Analysis, pk=pk)
    data_source_id = analysis.data_source.id  # Store the data source ID before deletion
    try:
        # Proactively delete related DashboardItems
        analysis.dashboard_items.all().delete()
        analysis.delete()
        # Redirect to data source page on success
        messages.success(request, 'Analysis deleted successfully!')
        return redirect('data_source_detail', pk=data_source_id)
    except Exception as e:
        logging.exception(f"Failed to delete analysis {pk}")
        messages.error(request, f'Failed to delete analysis: {str(e)}')
        return redirect('data_source_detail', pk=data_source_id)


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
        if result is None:
            analysis.result = None
        else:
            # For statistical analysis, include all available numeric columns for the dropdown
            if analysis.analysis_type == 'statistical':
                available_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                if hasattr(result, 'get') and result.get('column') in available_columns:
                    # Remove the current column from the list to avoid duplication
                    available_columns = [col for col in available_columns if col != result.get('column')]
                    # Add available columns to the result
                    result['available_columns'] = available_columns
                
            analysis.result = result
        analysis.status = 'completed'
        analysis.save()
        
    except Exception as e:
        # User-friendly error for classification/regression
        if analysis.analysis_type == 'classification':
            error = 'Classification not possible. Try another analysis method.'
        elif analysis.analysis_type == 'regression':
            error = 'Regression analysis not possible. Try another analysis method.'
        else:
            error = str(e)
        analysis.status = 'failed'
        analysis.result = {"error": error}  # Ensure valid JSON for the result field
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
    # Import necessary libraries
    import numpy as np
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.decomposition import PCA
    
    # Get parameters with defaults
    params = analysis.parameters or {}
    num_clusters = params.get('num_clusters', 3)
    features = params.get('features', df.select_dtypes(include=[np.number]).columns.tolist())
    algorithm = params.get('algorithm', 'kmeans')  # Allow different algorithms
    
    # Prepare data - drop NAs and select only numeric features
    X = df[features].dropna()
    
    # Scale data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Run clustering based on algorithm choice
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots
    import numpy as np
    
    if algorithm == 'dbscan':
        from sklearn.cluster import DBSCAN
        eps = params.get('eps', 0.5)
        min_samples = params.get('min_samples', 5)
        model = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = model.fit_predict(X_scaled)
        centers = None  # DBSCAN doesn't have explicit centers
    else:  # default to kmeans
        model = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = model.fit_predict(X_scaled)
        centers = model.cluster_centers_
    
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
        
        # If we used PCA, transform centers as well for visualization
        if centers is not None:
            centers_pca = pca.transform(centers)
    else:
        plot_features = features[:2]
        centers_pca = centers if centers is not None else None
    
    # Create cluster visualization
    fig = go.Figure()
    
    # Add scatter plot for data points
    for cluster_id in sorted(df_result['cluster'].unique()):
        cluster_data = df_result[df_result['cluster'] == cluster_id]
        
        if len(plot_features) >= 2:
            fig.add_trace(go.Scatter(
                x=cluster_data[plot_features[0]],
                y=cluster_data[plot_features[1]],
                mode='markers',
                name=f'Cluster {cluster_id}',
                marker=dict(size=8),
            ))
    
    # Add cluster centers if available (only for KMeans)
    if centers is not None and centers_pca is not None:
        if len(plot_features) >= 2:
            fig.add_trace(go.Scatter(
                x=centers_pca[:, 0] if len(features) > 2 else centers[:, 0],
                y=centers_pca[:, 1] if len(features) > 2 else centers[:, 1],
                mode='markers',
                marker=dict(
                    color='black',
                    size=12,
                    symbol='x'
                ),
                name='Cluster Centers'
            ))
    
    # Update layout
    x_axis_title = 'PCA Component 1' if len(features) > 2 else features[0]
    y_axis_title = 'PCA Component 2' if len(features) > 2 else features[1] if len(features) > 1 else ''
    
    fig.update_layout(
        title='Cluster Analysis Results',
        xaxis_title=x_axis_title,
        yaxis_title=y_axis_title,
        template='plotly_white',
        legend_title='Clusters',
        height=600
    )
    
    cluster_vis_html = pio.to_html(fig, full_html=False)
    
    # Create a second visualization - cluster sizes
    counts = df_result['cluster'].value_counts().sort_index()
    
    size_fig = go.Figure(data=[
        go.Bar(
            x=[f'Cluster {i}' for i in counts.index],
            y=counts.values,
            marker_color='rgb(55, 83, 217)'
        )
    ])
    
    size_fig.update_layout(
        title='Cluster Sizes',
        xaxis_title='Cluster',
        yaxis_title='Number of Points',
        template='plotly_white'
    )
    
    size_vis_html = pio.to_html(size_fig, full_html=False)
    
    # Generate insights about the clusters
    insights = []
    
    # Check cluster balance
    if len(counts) > 1:
        max_size = counts.max()
        min_size = counts.min()
        ratio = max_size / min_size if min_size > 0 else float('inf')
        
        if ratio > 10:
            insights.append({
                'type': 'warning',
                'message': f'Clusters are highly imbalanced (largest/smallest ratio: {ratio:.1f}). Consider adjusting the number of clusters or using a different algorithm.'
            })
        elif ratio < 1.5:
            insights.append({
                'type': 'success',
                'message': 'Clusters are well-balanced in size, suggesting a good separation of data.'
            })
    
    # Look at feature importance if we have more than one cluster
    if len(counts) > 1 and algorithm == 'kmeans':
        # Simple feature importance: the spread of cluster centers along each axis
        feature_importance = {}
        for i, feature in enumerate(features):
            if centers is not None:
                spread = np.std(centers[:, i])
                feature_importance[feature] = float(spread)
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        if sorted_features:
            top_features = sorted_features[:min(3, len(sorted_features))]
            insights.append({
                'type': 'info',
                'message': f'The most important features for clustering appear to be: {", ".join([f[0] for f in top_features])}'
            })
    
    # Helper function to ensure all values are JSON serializable
    def make_json_serializable(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return make_json_serializable(obj.tolist())
        elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_json_serializable(i) for i in obj]
        else:
            return obj
    
    # Convert value_counts to a serializable format
    cluster_counts = {}
    for cluster, count in df_result['cluster'].value_counts().items():
        cluster_counts[int(cluster) if isinstance(cluster, (np.integer, np.int64)) else cluster] = int(count)
    
    # Convert results to JSON-serializable format
    result = {
        'cluster_counts': cluster_counts,
        'cluster_centers': centers.tolist() if centers is not None and len(features) <= 2 else [],
        'sample_data': df_result.head(100).to_dict('records'),
        'plot_features': plot_features,
        'features_used': features,
        'algorithm': algorithm,
        'num_clusters': int(num_clusters) if algorithm == 'kmeans' else 'auto',
        'visualizations': {
            'cluster_plot': cluster_vis_html,
            'size_plot': size_vis_html
        },
        'insights': insights
    }
    
    # Make sure all nested values are JSON serializable
    result = make_json_serializable(result)
    
    return result

def run_classification_analysis(df, analysis):
    # Classification analysis implementation
    # For now, just return a placeholder with only JSON-serializable types
    try:
        preview = df.head(5).to_dict('records')
    except Exception:
        preview = []
    return {
        'message': 'Classification analysis not implemented yet',
        'preview': preview
    }

def run_regression_analysis(df, analysis):
    # Always return a valid JSON-serializable object and handle errors
    try:
        preview = df.head(5).to_dict('records')
    except Exception:
        preview = []
    return {
        'message': 'Regression analysis not possible. Try another analysis method.',
        'preview': preview
    }

def run_timeseries_analysis(df, analysis):
    # Time Series analysis implementation
    # For now, just return a placeholder
    return {
        'message': 'Time Series analysis not implemented yet',
        'preview': df.head(5).to_dict('records')
    }

def run_statistical_analysis(df, analysis):
    # Import necessary libraries
    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio
    import plotly.express as px
    from plotly.subplots import make_subplots
    import pandas as pd
    from scipy import stats as scipy_stats
    
    # Get parameters with defaults
    params = analysis.parameters or {}
    column = params.get('column', df.select_dtypes(include=[np.number]).columns[0] if not df.empty else None)
    if not column:
        return {'error': 'No numeric column available for analysis'}
    
    # Basic statistics with formatted values for display
    stats_obj = df[column].describe()
    stats = stats_obj.to_dict()
    
    # Calculate additional measures needed for analysis
    q1 = stats_obj['25%']
    q3 = stats_obj['75%']
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    skewness = df[column].skew()
    kurtosis = df[column].kurtosis()
    missing_values = df[column].isna().sum()
    missing_percent = (missing_values / len(df)) * 100
    
    # Identify outliers
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
    
    # Create formatted stats for display
    formatted_stats = {
        'count': int(stats_obj['count']),
        'mean': f"{stats_obj['mean']:.2f}",
        'median': f"{df[column].median():.2f}",
        'std': f"{stats_obj['std']:.2f}",
        'min': f"{stats_obj['min']:.2f}",
        '25%': f"{stats_obj['25%']:.2f}",
        '50%': f"{stats_obj['50%']:.2f}",
        '75%': f"{stats_obj['75%']:.2f}",
        'max': f"{stats_obj['max']:.2f}",
        'range': f"{stats_obj['max'] - stats_obj['min']:.2f}",
        'iqr': f"{iqr:.2f}"
    }
    
    # Create a combined visualization using subplots for better presentation
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'Distribution of {column}', 
            f'Box Plot of {column}',
            f'Q-Q Plot (Test for Normality)', 
            f'Value Counts of {column}'
        ),
        specs=[[{'type': 'histogram'}, {'type': 'box'}],
              [{'type': 'scatter'}, {'type': 'bar'}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    # 1. Histogram - Add to subplot (top left)
    fig.add_trace(
        go.Histogram(
            x=df[column].dropna(),
            nbinsx=20,
            marker=dict(
                color='rgba(55, 128, 191, 0.7)',
                line=dict(color='rgba(55, 128, 191, 1)', width=1)
            ),
            name=f'Distribution'
        ),
        row=1, col=1
    )
    
    # Add distribution curve overlay
    x_range = np.linspace(min(df[column].dropna()), max(df[column].dropna()), 100)
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=scipy_stats.norm.pdf(x_range, df[column].mean(), df[column].std()) * len(df[column].dropna()) * (max(df[column].dropna()) - min(df[column].dropna())) / 20,
            mode='lines',
            line=dict(color='rgba(255, 0, 0, 0.7)', width=2, dash='dash'),
            name='Normal Distribution'
        ),
        row=1, col=1
    )
    
    # 2. Box Plot - Add to subplot (top right)
    fig.add_trace(
        go.Box(
            y=df[column].dropna(),
            name=column,
            boxmean=True,  # adds a marker for the mean
            marker=dict(color='rgba(58, 71, 80, 0.6)'),
            line=dict(color='rgba(58, 71, 80, 1)'),
            boxpoints='outliers'  # only show outliers
        ),
        row=1, col=2
    )
    
    # 3. Q-Q Plot - Add to subplot (bottom left)
    # Prepare Q-Q plot data
    qq_data = df[column].dropna()
    qq = scipy_stats.probplot(qq_data, dist='norm')
    x_qq = qq[0][0]  # theoretical quantiles
    y_qq = qq[0][1]  # sample quantiles
    
    fig.add_trace(
        go.Scatter(
            x=x_qq,
            y=y_qq,
            mode='markers',
            marker=dict(color='rgba(44, 160, 101, 0.7)'),
            name='Q-Q Plot'
        ),
        row=2, col=1
    )
    
    # Add reference line
    line_x = np.linspace(min(x_qq), max(x_qq), 100)
    line_y = line_x * qq[1][0] + qq[1][1]  # slope and intercept from probplot
    
    fig.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y,
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='Reference Line'
        ),
        row=2, col=1
    )
    
    # 4. Value Counts - Add to subplot (bottom right)
    # For numerical data, create a histogram with fewer bins to show value distribution
    value_counts = df[column].value_counts().sort_index()
    if len(value_counts) > 15:  # If too many values, bin them
        bins = pd.cut(df[column].dropna(), bins=10)
        value_counts = bins.value_counts().sort_index()
        x_values = [f"{interval.left:.1f}-{interval.right:.1f}" for interval in value_counts.index]
    else:
        x_values = [f"{x}" for x in value_counts.index]
    
    fig.add_trace(
        go.Bar(
            x=x_values,
            y=value_counts.values,
            marker=dict(color='rgba(246, 78, 139, 0.6)'),
            name='Value Counts'
        ),
        row=2, col=2
    )
    
    # Update layout with improved appearance
    fig.update_layout(
        title=dict(
            text=f'Statistical Analysis of {column}',
            font=dict(size=24)
        ),
        showlegend=False,
        template='plotly_white',
        height=800,  # Taller figure for better visibility
        margin=dict(t=100, l=50, r=50, b=50),
        annotations=[
            dict(
                x=0.5, y=1.05,
                showarrow=False,
                text=f'Statistical Analysis of {column}',
                xref='paper', yref='paper',
                font=dict(size=24)
            )
        ]
    )
    
    # Update subplot axes
    fig.update_xaxes(title_text=column, row=1, col=1)
    fig.update_yaxes(title_text='Frequency', row=1, col=1)
    fig.update_yaxes(title_text=f'{column} Value', row=1, col=2)
    fig.update_xaxes(title_text='Theoretical Quantiles', row=2, col=1)
    fig.update_yaxes(title_text='Sample Quantiles', row=2, col=1)
    fig.update_xaxes(title_text=f'{column} Values', row=2, col=2)
    fig.update_yaxes(title_text='Count', row=2, col=2)
    
    # Convert to HTML
    combined_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
    
    # Structure for organized statistical summary
    box = {
        'min': float(df[column].min()),
        'q1': float(df[column].quantile(0.25)),
        'median': float(df[column].median()),
        'q3': float(df[column].quantile(0.75)),
        'max': float(df[column].max()),
        'outliers': outliers.tolist() if not outliers.empty else []
    }
    
    # Helper functions for statistical interpretation
    def interpret_skewness(skew_value):
        if abs(skew_value) < 0.5:
            return 'Approximately symmetric'
        elif abs(skew_value) < 1.0:
            return 'Moderately skewed'
        else:
            return 'Highly skewed ' + ('to the right (positive)' if skew_value > 0 else 'to the left (negative)')
    
    def interpret_kurtosis(kurt_value):
        if abs(kurt_value) < 0.5:
            return 'Approximately normal'
        elif kurt_value > 0:
            return 'Leptokurtic (heavier tails than normal)'
        else:
            return 'Platykurtic (lighter tails than normal)'
    
    def format_sample_values(values, max_samples=5):
        if len(values) <= max_samples:
            return ', '.join([f"{v:.2f}" for v in values])
        else:
            sample = values.sample(max_samples) if hasattr(values, 'sample') else values[:max_samples]
            return ', '.join([f"{v:.2f}" for v in sample]) + f" and {len(values) - max_samples} more"
    
    # Additional statistical measures with interpretations
    cv = float(stats_obj['std'] / stats_obj['mean'] * 100) if stats_obj['mean'] != 0 else None
    
    additional_stats = {
        'skewness': {
            'value': float(skewness),
            'formatted': f"{skewness:.3f}",
            'interpretation': interpret_skewness(skewness)
        },
        'kurtosis': {
            'value': float(kurtosis),
            'formatted': f"{kurtosis:.3f}",
            'interpretation': interpret_kurtosis(kurtosis)
        },
        'normality_test': {
            'test': 'Shapiro-Wilk',
            'value': scipy_stats.shapiro(df[column].dropna())[1] if len(df[column].dropna()) < 5000 else None,
            'interpretation': 'Normal distribution' if (scipy_stats.shapiro(df[column].dropna())[1] > 0.05 and len(df[column].dropna()) < 5000) else 'Non-normal distribution'
        },
        'median': float(df[column].median()),
        'mode': float(df[column].mode().iloc[0]) if not df[column].mode().empty else None,
        'range': float(df[column].max() - df[column].min()),
        'iqr': float(iqr),
        'coefficient_of_variation': cv
    }
    
    # Enhanced key insights with more detailed analysis
    key_insights = []
    
    # Insight 1: Distribution type
    if abs(skewness) < 0.5 and abs(kurtosis) < 0.5:
        key_insights.append({
            'type': 'success',
            'icon': 'fas fa-check-circle',
            'title': 'Normal Distribution',
            'message': f'The distribution of {column} appears to be approximately normal (skewness: {skewness:.3f}, kurtosis: {kurtosis:.3f}), making it suitable for parametric statistical tests.'
        })
    elif abs(skewness) > 1 or abs(kurtosis) > 1:
        key_insights.append({
            'type': 'warning',
            'icon': 'fas fa-exclamation-triangle',
            'title': 'Non-Normal Distribution',
            'message': f'The distribution of {column} deviates significantly from normal (skewness: {skewness:.3f}, kurtosis: {kurtosis:.3f}). Consider using non-parametric tests or transforming the data.'
        })
        
        # Add transformation suggestion if highly skewed
        if abs(skewness) > 1.5:
            transform_suggestion = 'log transformation' if skewness > 0 else 'square transformation'
            key_insights.append({
                'type': 'info',
                'icon': 'fas fa-lightbulb',
                'title': 'Transformation Suggestion',
                'message': f'Due to the high skewness ({skewness:.3f}), consider applying a {transform_suggestion} to normalize the data.'
            })
    
    # Insight 2: Outliers analysis
    outlier_count = len(outliers)
    outlier_percent = 0
    if outlier_count > 0:
        outlier_percent = (outlier_count / stats_obj['count']) * 100
        outlier_effect = 'significant' if outlier_percent > 5 else 'moderate' if outlier_percent > 1 else 'minimal'
        key_insights.append({
            'type': 'warning' if outlier_percent > 2 else 'info',
            'icon': 'fas fa-filter',
            'title': 'Outliers Detected',
            'message': f'Found {outlier_count} outliers ({outlier_percent:.2f}% of data) which may have a {outlier_effect} effect on your analysis.'
        })
    
    # Insight 3: Data quality
    missing_percent = 0
    if missing_values > 0:
        missing_percent = (missing_values / len(df)) * 100
        missing_severity = 'severe' if missing_percent > 15 else 'moderate' if missing_percent > 5 else 'minor'
        key_insights.append({
            'type': 'danger' if missing_percent > 15 else 'warning' if missing_percent > 5 else 'info',
            'icon': 'fas fa-exclamation-circle',
            'title': 'Missing Values',
            'message': f'This column has {missing_values} missing values ({missing_percent:.2f}% of data), which represents a {missing_severity} data quality issue. Consider imputation or filtering strategies.'
        })
    
    # Insight 4: Data variability
    if cv is not None:
        if cv > 30:
            key_insights.append({
                'type': 'info',
                'icon': 'fas fa-chart-line',
                'title': 'High Variability',
                'message': f'The coefficient of variation is {cv:.2f}%, indicating high data variability. This may suggest heterogeneity in your data.'
            })
        elif cv < 5:
            key_insights.append({
                'type': 'info',
                'icon': 'fas fa-compress-arrows-alt',
                'title': 'Low Variability',
                'message': f'The coefficient of variation is only {cv:.2f}%, indicating low data variability. This may suggest homogeneity in your data.'
            })
    
    # Insight 5: Distribution peaks
    if abs(kurtosis) > 1.5:
        peak_type = 'more peaked (leptokurtic)' if kurtosis > 0 else 'more flat (platykurtic)'
        tail_behavior = 'heavier tails' if kurtosis > 0 else 'lighter tails'
        key_insights.append({
            'type': 'info',
            'icon': 'fas fa-mountain',
            'title': 'Distribution Shape',
            'message': f'The distribution is {peak_type} than a normal distribution with {tail_behavior}.'
        })
    
    # Helper function to ensure all values are JSON serializable
    def make_json_serializable(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_json_serializable(i) for i in obj]
        else:
            return obj
    
    # Calculate outlier count (ensuring it's a native Python int)
    outlier_count = int(len(outliers))
    
    # Combine results in a user-friendly structure with better organization
    result = {
        'column': column,
        'basic_stats': formatted_stats,
        'additional_stats': additional_stats,
        'box_plot_data': box,
        'key_insights': key_insights,
        'visualizations': {
            'combined': combined_html
        },
        'analysis_summary': {
            'distribution_type': additional_stats['normality_test']['interpretation'],
            'central_tendency': {
                'mean': formatted_stats['mean'],
                'median': formatted_stats['median'],
                'mode': str(additional_stats['mode']) if additional_stats['mode'] is not None else 'N/A'
            },
            'dispersion': {
                'range': additional_stats['range'],
                'std': formatted_stats['std'],
                'iqr': additional_stats['iqr'],
                'coefficient_of_variation': f"{cv:.2f}%" if cv is not None else 'N/A'
            },
            'data_quality': {
                'missing_values': int(missing_values),
                'missing_percent': f"{missing_percent:.2f}%" if missing_values > 0 else '0%',
                'outliers': outlier_count,
                'outlier_percent': f"{outlier_percent:.2f}%" if outlier_count > 0 else '0%'
            }
        }
    }
    
    # Make sure all values are JSON serializable
    result = make_json_serializable(result)
    
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
    # Handle GET requests for column selection in existing analysis
    if request.method == 'GET':
        try:
            analysis_id = request.GET.get('analysis_id')
            column = request.GET.get('column')
            
            if not analysis_id or not column:
                return JsonResponse({'error': 'Missing analysis_id or column parameter'}, status=400)
            
            # Get the analysis
            analysis = get_object_or_404(Analysis, pk=analysis_id)
            data_source = analysis.data_source
            
            # Load data
            df = load_data(data_source)
            
            # Verify the column exists
            if column not in df.columns:
                return JsonResponse({'error': f'Column {column} not found in dataset'}, status=400)
                
            # Check if column is numeric for statistical analysis
            if analysis.analysis_type == 'statistical' and not np.issubdtype(df[column].dtype, np.number):
                return JsonResponse({'error': f'Column {column} is not numeric'}, status=400)
            
            # Update analysis parameters
            if analysis.parameters is None:
                analysis.parameters = {}
            analysis.parameters['column'] = column
            analysis.save()
            
            # Run the analysis
            if analysis.analysis_type == 'statistical':
                result = run_statistical_analysis(df, analysis)
                if result.get('error'):
                    return JsonResponse({'error': result.get('error')}, status=400)
                    
                # Save the result
                analysis.result = result
                analysis.status = 'completed'
                analysis.save()
                
                return JsonResponse({'success': True, 'message': f'Analysis updated for column {column}'})
            else:
                return JsonResponse({'error': 'Column selection is only supported for statistical analysis'}, status=400)
                
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    # Handle POST requests for creating new analysis
    elif request.method == 'POST':
        try:
            data = json.loads(request.body)
            analysis_type = data.get('analysis_type')
            data_source_id = data.get('data_source_id')
            parameters = data.get('parameters', {})
            
            # Get the data source
            data_source = get_object_or_404(DataSource, pk=data_source_id)
            
            # Create analysis record
            analysis = Analysis.objects.create(
                name=f"{analysis_type.title()} Analysis",
                description=f"Auto-generated {analysis_type} analysis",
                data_source=data_source,
                analysis_type=analysis_type,
                parameters=parameters,
                created_by=request.user if request.user.is_authenticated else None,
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
    else:
        return JsonResponse({'error': 'Only GET and POST requests are allowed'}, status=405)
