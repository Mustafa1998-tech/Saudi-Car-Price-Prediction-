import os

# Create necessary directories
directories = [
    'data/raw',
    'data/processed',
    'models',
    'reports',
    'eda'
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"✅ Created directory: {directory}")

# Create .gitkeep files to keep empty directories
for directory in directories:
    with open(f"{directory}/.gitkeep", 'w') as f:
        pass
    print(f"✅ Created .gitkeep in: {directory}")

print("\n🎉 Directory structure setup complete!")
