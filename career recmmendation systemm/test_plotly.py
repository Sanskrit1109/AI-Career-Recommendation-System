#!/usr/bin/env python3
"""
Test script to verify Plotly syntax
"""

import plotly.express as px
import pandas as pd

# Create test data
data = {'x': [1, 2, 3, 4], 'y': [10, 11, 12, 13]}
df = pd.DataFrame(data)

# Test 1: Create bar chart with correct syntax
print("Testing Plotly bar chart...")
fig = px.bar(x=df['x'], y=df['y'], title="Test Chart")
fig.update_layout(xaxis_title="X Axis", yaxis_title="Y Axis")
print("✅ Plotly bar chart with update_layout() works correctly!")

# Test 2: Try the incorrect syntax that should fail
print("\nTesting incorrect syntax...")
try:
    fig.update_xaxis(title="Test")
    print("❌ update_xaxis() should not work!")
except AttributeError as e:
    print(f"✅ Confirmed: {e}")

print("\n🎉 Plotly syntax test completed!")