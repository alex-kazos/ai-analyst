from django.db import models
from django.contrib.auth.models import User
import uuid
import os


def get_file_path(instance, filename):
    ext = filename.split('.')[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    return os.path.join('datasets', filename)


class DataSource(models.Model):
    TYPE_CHOICES = [
        ('file', 'File Upload (CSV, Excel, JSON, Parquet, Feather, etc)')
    ]
    
    FILE_TYPE_CHOICES = [
        ('csv', 'CSV'),
        ('xlsx', 'Excel'),
        ('json', 'JSON'),
        ('parquet', 'Parquet'),
        ('feather', 'Feather'),
        ('pickle', 'Pickle'),
        ('hdf', 'HDF5'),
        ('orc', 'ORC')
    ]
    
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    source_type = models.CharField(max_length=20, choices=TYPE_CHOICES)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name='data_sources')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # For file uploads
    file = models.FileField(upload_to=get_file_path, blank=True, null=True)
    file_type = models.CharField(max_length=20, choices=FILE_TYPE_CHOICES, blank=True, null=True)
    
    # For database connections

    def __str__(self):
        return self.name
    
    class Meta:
        verbose_name = 'Data Source'
        verbose_name_plural = 'Data Sources'


class Analysis(models.Model):
    TYPE_CHOICES = [
        ('quick_ai', 'Quick AI Analysis'),
        ('clustering', 'Clustering Analysis'),
        ('classification', 'Classification Analysis'),
        ('regression', 'Regression Analysis'),
        ('time_series', 'Time Series Analysis'),
        ('statistical', 'Statistical Analysis'),
        ('custom', 'Custom Analysis')
    ]
    
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    data_source = models.ForeignKey(DataSource, on_delete=models.CASCADE, related_name='analyses')
    analysis_type = models.CharField(max_length=20, choices=TYPE_CHOICES)
    query = models.TextField(blank=True, null=True)  # SQL query or question
    parameters = models.JSONField(blank=True, null=True)  # Store analysis params as JSON
    result = models.JSONField(blank=True, null=True)  # Store results as JSON
    status = models.CharField(max_length=20, default='pending')  # pending, running, completed, failed
    created_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name='analyses')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.name
    
    class Meta:
        verbose_name = 'Analysis'
        verbose_name_plural = 'Analyses'


class Dashboard(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    layout = models.JSONField(blank=True, null=True)  # Store dashboard layout as JSON
    created_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name='dashboards')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.name


class DashboardItem(models.Model):
    TYPE_CHOICES = [
        ('chart', 'Chart'),
        ('table', 'Table'),
        ('metric', 'Metric'),
        ('text', 'Text')
    ]
    
    dashboard = models.ForeignKey(Dashboard, on_delete=models.CASCADE, related_name='items')
    analysis = models.ForeignKey(Analysis, on_delete=models.CASCADE, related_name='dashboard_items', null=True, blank=True)
    item_type = models.CharField(max_length=20, choices=TYPE_CHOICES)
    name = models.CharField(max_length=255)
    configuration = models.JSONField(blank=True, null=True)  # Store item config as JSON
    position_x = models.IntegerField(default=0)
    position_y = models.IntegerField(default=0)
    width = models.IntegerField(default=1)
    height = models.IntegerField(default=1)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.name


class Question(models.Model):
    text = models.TextField()
    data_source = models.ForeignKey(DataSource, on_delete=models.CASCADE, related_name='questions')
    sql_query = models.TextField(blank=True, null=True)
    result = models.JSONField(blank=True, null=True)
    asked_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name='questions')
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.text[:50]
