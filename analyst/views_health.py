import os
import shutil
import psutil
from django.http import JsonResponse
from django.views import View
from django.conf import settings

class HealthCheckView(View):
    """
    Health check endpoint for the application.
    Returns 200 if the application is healthy, 500 otherwise.
    """
    def get(self, request, *args, **kwargs):
        # Check disk usage
        try:
            disk_usage = psutil.disk_usage('/')
            disk_usage_percent = disk_usage.percent
            if disk_usage_percent > settings.HEALTH_CHECK['DISK_USAGE_MAX']:
                return JsonResponse(
                    {'error': f'Disk usage is too high: {disk_usage_percent}%'},
                    status=500
                )
        except Exception as e:
            return JsonResponse(
                {'error': f'Error checking disk usage: {str(e)}'},
                status=500
            )
        
        # Check memory
        try:
            memory = psutil.virtual_memory()
            memory_available_mb = memory.available / (1024 * 1024)  # Convert to MB
            if memory_available_mb < settings.HEALTH_CHECK['MEMORY_MIN']:
                return JsonResponse(
                    {'error': f'Available memory is too low: {memory_available_mb:.2f}MB'},
                    status=500
                )
        except Exception as e:
            return JsonResponse(
                {'error': f'Error checking memory: {str(e)}'},
                status=500
            )
        
        # Check database connection
        from django.db import connection
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
        except Exception as e:
            return JsonResponse(
                {'error': f'Database connection failed: {str(e)}'},
                status=500
            )
        
        # If all checks pass
        return JsonResponse({
            'status': 'healthy',
            'disk_usage_percent': disk_usage_percent,
            'available_memory_mb': memory_available_mb,
            'database': 'connected'
        })
