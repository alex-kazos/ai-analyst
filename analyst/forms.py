from django import forms
from .models import DataSource, Analysis, Dashboard, DashboardItem, Question

class DataSourceForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput(), required=False, 
                            help_text="Password for database connection")
    
    class Meta:
        model = DataSource
        fields = ['name', 'description', 'source_type', 'file']
        widgets = {
            'description': forms.Textarea(attrs={'rows': 3}),
            'connection_string': forms.Textarea(attrs={'rows': 2}),
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['file'].required = False
        
        # Add a JavaScript trigger to show/hide fields based on source_type
        self.fields['source_type'].widget.attrs.update({
            'class': 'source-type-select',
            'onchange': 'toggleSourceFields(this.value)'
        })

class AnalysisForm(forms.ModelForm):
    class Meta:
        model = Analysis
        fields = ['name', 'description', 'analysis_type', 'query']
        widgets = {
            'description': forms.Textarea(attrs={'rows': 3}),
            'query': forms.Textarea(attrs={'rows': 4, 'placeholder': 'SQL query or analysis prompt...'}),
        }

class DashboardForm(forms.ModelForm):
    class Meta:
        model = Dashboard
        fields = ['name', 'description']
        widgets = {
            'description': forms.Textarea(attrs={'rows': 3}),
        }

class DashboardItemForm(forms.ModelForm):
    class Meta:
        model = DashboardItem
        fields = ['name', 'item_type', 'analysis', 'width', 'height']

class QuestionForm(forms.ModelForm):
    class Meta:
        model = Question
        fields = ['text']
        widgets = {
            'text': forms.Textarea(attrs={
                'rows': 3, 
                'placeholder': 'Ask a question about your data...', 
                'class': 'form-control'
            }),
        }
        labels = {
            'text': 'Your Question',
        }
