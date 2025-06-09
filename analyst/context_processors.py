from django.conf import settings

def show_beta_tag(request):
    """Inject SHOW_BETA_TAG into all templates."""
    return {
        'SHOW_BETA_TAG': getattr(settings, 'SHOW_BETA_TAG', True)
    }
