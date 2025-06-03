from django import template
from allauth.socialaccount.models import SocialApp

register = template.Library()

@register.simple_tag
def is_google_configured():
    """Check if Google provider is configured"""
    return SocialApp.objects.filter(provider='google').exists()
