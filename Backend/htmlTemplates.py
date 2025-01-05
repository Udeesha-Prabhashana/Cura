# frontend/htmlTemplates.py

bot_template = '''
<div style="text-align:left; background-color: #D1E7FF; padding: 10px; border-radius: 10px; margin-bottom: 10px;">
    <strong>Bot:</strong> {{MSG}}
</div>
'''

user_template = '''
<div style="text-align:right; background-color: #E8E8E8; padding: 10px; border-radius: 10px; margin-bottom: 10px;">
    <strong>You:</strong> {{MSG}}
</div>
'''

css = """
<style>
    .stApp {
        font-family: Arial, sans-serif;
    }
    .chat-box {
        max-width: 600px;
        margin: 0 auto;
    }
</style>
"""
