from django.urls import path
from . import views

urlpatterns = [
    # Home
    path('', views.home, name='home'),
    
    # Data Source routes
    path('data-sources/', views.data_source_list, name='data_source_list'),
    path('data-sources/new/', views.data_source_create, name='data_source_create'),
    path('data-sources/<int:pk>/', views.data_source_detail, name='data_source_detail'),
    path('data-sources/<int:pk>/edit/', views.data_source_edit, name='data_source_edit'),
    path('data-sources/<int:pk>/delete/', views.data_source_delete, name='data_source_delete'),
    path('data-sources/<int:pk>/rename/', views.data_source_rename, name='data_source_rename'),
    path('data-sources/<int:pk>/update-description/', views.data_source_update_description, name='data_source_update_description'),
    path('data-sources/<int:pk>/update-file/', views.data_source_update_file, name='data_source_update_file'),
    
    # Analysis routes
    path('data-sources/<int:data_source_id>/analysis/new/', views.analysis_create, name='analysis_create'),
    path('analysis/<int:pk>/run/', views.run_analysis, name='run_analysis'),
    
    # Question routes
    path('data-sources/<int:data_source_id>/question/', views.question_form, name='question_form'),
    path('question/<int:pk>/result/', views.question_result, name='question_result'),
    
    # Dashboard routes
    path('dashboards/', views.dashboard_list, name='dashboard_list'),
    path('dashboards/new/', views.dashboard_create, name='dashboard_create'),
    path('dashboards/<int:pk>/', views.dashboard_detail, name='dashboard_detail'),
    
    # API routes
    path('api/data-sources/<int:pk>/preview/', views.api_data_preview, name='api_data_preview'),
    path('api/data-sources/<int:pk>/ai_analysis/', views.api_data_source_ai_analysis, name='api_data_source_ai_analysis'),
    path('api/analysis/run/', views.api_run_analysis, name='api_run_analysis'),
]
