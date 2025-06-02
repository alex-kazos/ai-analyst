from django.contrib import admin
from .models import DataSource, Analysis, Dashboard, DashboardItem, Question

# Register models
admin.site.register(DataSource)
admin.site.register(Analysis)
admin.site.register(Dashboard)
admin.site.register(DashboardItem)
admin.site.register(Question)
