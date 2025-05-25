from django import template

register = template.Library()

@register.filter
def get_item(dictionary, key):
    """
    Template filter to get an item from a dictionary using a key
    Usage: {{ dictionary|get_item:key }}
    """
    if not dictionary:
        return None
    
    try:
        return dictionary.get(key, None)
    except (AttributeError, TypeError):
        # For non-dictionary objects that support item access
        try:
            return dictionary[key]
        except (KeyError, TypeError, IndexError):
            return None

@register.filter(name='abs')
def absolute_value(value):
    """
    Template filter to get the absolute value of a number
    Usage: {{ value|abs }}
    """
    try:
        return __builtins__['abs'](float(value))
    except (ValueError, TypeError):
        return value
