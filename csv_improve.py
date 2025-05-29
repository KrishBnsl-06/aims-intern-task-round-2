import pandas as pd
import re

def enhance_instructions(row):
    full = row['full-instructions']
    title = row['title']
    
    # Extract critical parameters
    temps = re.findall(r'\b(\d{3,4}°?F?)\b', full)
    times = re.findall(r'(\d+[\s-]*(?:minute|hour|min|hr)s?)', full)
    prep_steps = re.findall(r'(?:dice|chop|slice|marinate|season|preheat|pat dry|rinse|sift|mix)\s[\w\s]+?(?=[A-Z]|\.|\,)', full)
    
    # Structure into 3 steps
    step1 = f"Prep & Start: {prep_steps[0] if prep_steps else ''}"
    step1 += f" at {temps[0]}" if temps else ""
    
    step2 = f"Cook & Combine: {prep_steps[1] if len(prep_steps)>1 else ''}"
    step2 += f" for {times[0]}" if times else ""
    
    step3 = f"Finish & Serve: {prep_steps[2] if len(prep_steps)>2 else ''}"
    step3 += f" ({times[1]})" if len(times)>1 else ""
    
    # Special handling for specific recipe patterns
    if 'Bake' in full:
        step2 += f" at {temps[0]}" if temps else ""
    if 'Broil' in title:
        step2 = step2.replace('Cook', 'Sear & Broil')
    
    return f"{step1}. {step2}. {step3}."

# Load and process data
df = pd.read_csv('recipes_2.csv')
df['enhanced_3_step'] = df.apply(enhance_instructions, axis=1)

# Save enhanced CSV
df.to_csv('enhanced_recipes.csv', index=False)
print("Enhanced CSV saved with 3-step improvements")

