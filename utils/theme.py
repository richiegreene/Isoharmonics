import sys
from Foundation import NSUserDefaults

def is_dark_theme():
    try:
        defaults = NSUserDefaults.standardUserDefaults()
        style = defaults.stringForKey_("AppleInterfaceStyle")
        return style == "Dark"
    except Exception as e:
        return False
