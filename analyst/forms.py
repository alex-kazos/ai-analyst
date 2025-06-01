from django import forms
from .models import DataSource, Analysis, Dashboard, DashboardItem, Question

class DataSourceForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput(), required=False, 
                            help_text="Password for database connection")
    file_type = forms.ChoiceField(choices=DataSource.FILE_TYPE_CHOICES, required=False,
                              help_text="Select the type of file you're uploading")
    
    class Meta:
        model = DataSource
        fields = ['name', 'description', 'source_type', 'file', 'file_type']
        widgets = {
            'description': forms.Textarea(attrs={'rows': 3}),
            'connection_string': forms.Textarea(attrs={'rows': 2}),
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['file'].required = False
        self.fields['file_type'].required = False
        
        # Add a JavaScript trigger to show/hide fields based on source_type
        self.fields['source_type'].widget.attrs.update({
            'class': 'source-type-select',
            'onchange': 'toggleSourceFields(this.value)'
        })
        
    def clean(self):
        cleaned_data = super().clean()
        source_type = cleaned_data.get('source_type')
        file = cleaned_data.get('file')
        file_type = cleaned_data.get('file_type')
        
        if source_type == 'file':
            if not file:
                raise forms.ValidationError("Please upload a file.")
            
            if not file_type:
                # Try to detect file type from extension
                filename = file.name.lower()
                if filename.endswith('.csv'):
                    cleaned_data['file_type'] = 'csv'
                elif filename.endswith(('.xls', '.xlsx')):
                    cleaned_data['file_type'] = 'xlsx'
                elif filename.endswith('.json'):
                    cleaned_data['file_type'] = 'json'
                elif filename.endswith('.parquet'):
                    cleaned_data['file_type'] = 'parquet'
                elif filename.endswith('.feather'):
                    cleaned_data['file_type'] = 'feather'
                elif filename.endswith('.pickle'):
                    cleaned_data['file_type'] = 'pickle'
                elif filename.endswith('.h5') or filename.endswith('.hdf5'):
                    cleaned_data['file_type'] = 'hdf'
                elif filename.endswith('.orc'):
                    cleaned_data['file_type'] = 'orc'
                else:
                    raise forms.ValidationError("Unable to detect file type. Please specify the file type.")
        
        return cleaned_data

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
