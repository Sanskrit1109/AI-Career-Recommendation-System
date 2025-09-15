"""
Image utilities for MY NEW CAREER system
Creates visual elements and placeholder images
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from io import BytesIO
import base64

def create_career_logo(career_title, industry, size=(200, 200)):
    """Create a simple logo/image for a career"""
    
    # Industry-based color schemes
    industry_colors = {
        'Technology': ['#1f77b4', '#ff7f0e'],
        'Healthcare': ['#2ca02c', '#d62728'],
        'Education': ['#9467bd', '#8c564b'],
        'Business': ['#e377c2', '#7f7f7f'],
        'Engineering': ['#17becf', '#bcbd22'],
        'Finance': ['#ff9896', '#aec7e8'],
        'Arts': ['#ffbb78', '#98df8a'],
        'Science': ['#c5b0d5', '#c49c94'],
        'Government': ['#f7b6d3', '#c7c7c7'],
        'Media': ['#dbdb8d', '#9edae5'],
        'Sports': ['#ff6b6b', '#4ecdc4'],
        'Default': ['#667eea', '#764ba2']
    }
    
    # Get colors for the industry
    colors = industry_colors.get(industry, industry_colors['Default'])
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(size[0]/100, size[1]/100))
    
    # Create gradient background
    gradient = np.linspace(0, 1, 256).reshape(256, -1)
    gradient = np.vstack((gradient, gradient))
    
    # Display gradient
    ax.imshow(gradient, aspect='auto', cmap='viridis', alpha=0.7, extent=[0, 1, 0, 1])
    
    # Add career initial or icon
    career_initial = career_title[0].upper() if career_title else '?'
    ax.text(0.5, 0.5, career_initial, fontsize=60, ha='center', va='center', 
            color='white', weight='bold', fontfamily='sans-serif')
    
    # Remove axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
    buffer.seek(0)
    
    # Convert to base64 string
    img_str = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    
    return f"data:image/png;base64,{img_str}"

def get_industry_icon(industry):
    """Get an emoji icon for different industries"""
    industry_icons = {
        'Technology': '💻',
        'Healthcare': '🏥',
        'Education': '🎓',
        'Business': '💼',
        'Engineering': '⚙️',
        'Finance': '💰',
        'Arts': '🎨',
        'Science': '🔬',
        'Government': '🏛️',
        'Media': '📺',
        'Sports': '⚽',
        'Food': '🍽️',
        'Transportation': '🚗',
        'Environment': '🌱',
        'Law': '⚖️',
        'Real Estate': '🏠',
        'Retail': '🛍️'
    }
    
    return industry_icons.get(industry, '💼')

def create_skill_badge_html(skill, level=None):
    """Create HTML for a skill badge"""
    if level:
        level_colors = {
            'Beginner': '#28a745',
            'Intermediate': '#ffc107', 
            'Advanced': '#dc3545',
            'Expert': '#6f42c1'
        }
        color = level_colors.get(level, '#6c757d')
    else:
        color = '#1f77b4'
    
    return f"""
    <span style="
        background: linear-gradient(135deg, {color}, {color}aa);
        color: white;
        padding: 0.3em 0.8em;
        border-radius: 15px;
        margin: 0.2em;
        font-size: 0.9em;
        font-weight: 500;
        display: inline-block;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    ">
    {skill}
    </span>
    """

def create_progress_bar_html(percentage, label="Progress", color="#1f77b4"):
    """Create an HTML progress bar"""
    return f"""
    <div style="margin: 1em 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5em;">
            <span style="font-weight: 500;">{label}</span>
            <span style="color: {color}; font-weight: bold;">{percentage:.1f}%</span>
        </div>
        <div style="
            width: 100%;
            height: 20px;
            background-color: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
        ">
            <div style="
                width: {percentage}%;
                height: 100%;
                background: linear-gradient(90deg, {color}, {color}aa);
                transition: width 0.3s ease;
            "></div>
        </div>
    </div>
    """

if __name__ == "__main__":
    # Test the functions
    print("Testing image utilities...")
    
    # Test industry icon
    print(f"Technology icon: {get_industry_icon('Technology')}")
    
    # Test skill badge
    skill_html = create_skill_badge_html("Python", "Advanced")
    print(f"Skill badge HTML created: {len(skill_html)} characters")
    
    # Test progress bar
    progress_html = create_progress_bar_html(75, "Skill Match")
    print(f"Progress bar HTML created: {len(progress_html)} characters")
    
    print("✅ Image utilities tested successfully!")