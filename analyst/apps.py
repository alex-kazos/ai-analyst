from django.apps import AppConfig
import os

def ensure_google_socialapp():
    try:
        from allauth.socialaccount.models import SocialApp
        from django.contrib.sites.models import Site
        client_id = os.getenv('GOOGLE_CLIENT_ID')
        secret = os.getenv('GOOGLE_CLIENT_SECRET')
        if not client_id or not secret:
            return  # Do nothing if not set
        site = Site.objects.get_current()
        socialapp, created = SocialApp.objects.get_or_create(
            provider='google',
            defaults={
                'name': 'Google',
                'client_id': client_id,
                'secret': secret,
            }
        )
        # Update if needed
        updated = False
        if socialapp.client_id != client_id:
            socialapp.client_id = client_id
            updated = True
        if socialapp.secret != secret:
            socialapp.secret = secret
            updated = True
        if site not in socialapp.sites.all():
            socialapp.sites.add(site)
            updated = True
        if updated:
            socialapp.save()
    except Exception as e:
        # Log error if needed
        pass

class AnalystConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'analyst'

    def ready(self):
        ensure_google_socialapp()
