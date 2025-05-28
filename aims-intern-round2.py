from huggingface_hub import login

login(token="enter your hf access token here")

from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import requests
import torch

model_id = "google/gemma-3-4b-it"

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, device_map="auto"
).eval()

processor = AutoProcessor.from_pretrained(model_id)

recipes = {
    "quick creamy lumps": {
        "file": "images/30-Minute_Chicken_and_Dumplings.jpg",
        "instruction": "1. Cook Chicken & Base: Sauté veggies, add broth and chicken, then boil. 2. Add Dumplings: Drop in dumpling mix and steam. 3. Finish: Stir in peas, thicken, and serve."
    },
    "buttery farm meat": {
        "file": "images/Amish_Chicken.jpg",
        "instruction": "1. Coat Chicken: Dredge chicken in seasoned flour and place in a dish. 2. Add Cream: Pour mixed cream and water over it. 3. Bake: Cook at 350°F until golden."
    },
    "soft creamy swirls": {
        "file": "images/Angel_Chicken_Pasta.jpg",
        "instruction": "1. Make Sauce: Melt butter, add dressing, wine, soup, cream cheese. 2. Bake Chicken: Cover chicken with sauce, bake at 325°F for 60 mins. 3. Serve: Cook pasta, serve chicken & sauce over it."
    },
    "soy noodle thing": {
        "file": "images/Asian_Noodle_Bowl.jpg",
        "instruction": "1. Prep & Toast: Boil water. Toast almonds until golden. 2. Cook Pasta & Veggies: Cook pasta; add broccoli, then snow peas & peppers towards the end. 3. Sauce & Serve: Whisk almond butter, soy, lime, sugar, chili-garlic sauce & pasta water. Toss with drained pasta/veggies; garnish with almonds & scallions."
    },
    "spicy bean mess": {
        "file": "images/Award_Winning_Chili.jpg",
        "instruction": "1. Heat oil, brown ground meat. 2. Add onions, peppers, jalapeno; cook until soft. Drain grease.3. Stir in tomatoes, sauce, spices, water, and beans. Simmer covered for 1-2 hours. Serve with your favorite toppings"
    },
    "pink baked slab": {
        "file": "images/Baked_Salmon.jpg",
        "instruction": "1. Mix garlic, oil, basil, salt, pepper, lemon juice, parsley. Marinate salmon in fridge for 1 hour. 2. Preheat oven to 375°F. 3. Seal salmon and marinade in foil. Place foil-sealed salmon in dish, bake 35-45 minutes until flaky."
    },
    "dark sweet thighs": {
        "file": "images/Balsamic_Chicken_Thighs.jpg",
        "instruction": "1. Prep & Brown: Preheat sprayed pan. Season chicken thighs and brown well on all sides. 2. Cook & Add Aromatics: Cover, reduce heat, cook ~25 mins until done. Add and soften shallots.3. Sauce & Serve: Stir in balsamic vinegar, coat chicken, and serve with sauce spooned over."
    },
    "beefy crunchy pie": {
        "file": "images/Beef_Nacho_Crescent_Bake.jpg",
        "instruction": "1. Preheat oven to 375°F, grease dish. Cook beef with onion, garlic, pepper, and spices; drain. 2. Mix soup & milk. Roll crescent dough, cut into squares, fill with beef, seal, and place in dish; drizzle with soup mix.3. Bake 20-25 mins until golden. Sprinkle with cheese, bake 5 more mins until melted."
    },
    "fluffy malt loaf": {
        "file": "images/Beer_Bread.jpg",
        "instruction": "1. Mix & Bake: Preheat oven to 375°F. Mix sifted dry ingredients with beer, pour into greased pan, top with melted butter, and bake for 1 hour. 2. Sift Flour Crucial: For best results (not a \"brick\"), sift flour or spoon it lightly into the measuring cup.... 3. Tips & Variations: For softer crust, mix butter into batter. If using non-alcoholic liquid, add yeast."
    },
    "tomato bread bites": {
        "file": "images/Best_Ever_Bruschetta.jpg",
        "instruction": "1. Mix Ingredients: Combine tomatoes, garlic, basil, vinegar, olive oil, cheese, salt, and pepper in a bowl. 2. Marinate: Let the mixture sit for at least 15 minutes to blend flavors.3. Serve: Slice bread and spoon the tomato mixture on top."
    },
    "garlicky red splash": {
        "file": "images/Bevs_Spaghetti_Sauce.jpg",
        "instruction": "1. Brown beef, onion, garlic with olive oil and all seasonings (bay leaves, oregano, basil, Italian seasoning, salt, pepper). 2. Stir in tomato paste, sauce, and diced tomatoes; simmer covered for 1.5 hours. 3. Top cooked spaghetti with sauce, sautéed mushrooms, and Parmesan."
    },
    "sweet sticky meat": {
        "file": "images/Bourbon_Chicken.jpg",
        "instruction": "1. Brown Chicken: Heat oil, lightly brown chicken pieces, then remove from skillet. 2. Make Sauce & Simmer: Add remaining ingredients to skillet, heat until dissolved. Return chicken, bring to a boil, then simmer for 20 minutes.3. Serve: Serve over hot rice."
    },
    "green cheesy bowl": {
        "file": "images/Broccoli_Cheese_Soup.jpg",
        "instruction": "1. Roux & Base: Sauté onions. Make a golden roux with butter/flour, then whisk in stock and simmer. 2. Veggies & Cream: Add broccoli, carrots, onions, milk, and half & half; cook on low (do not boil).3. Cheese & Serve: Season, blend if desired, then stir in cheese until melted. Serve immediately."
    },
    "juicy charred slab": {
        "file": "images/Broil_a_Perfect_Steak.jpg",
        "instruction": "1. Prep & Preheat: Bring steaks to room temp, pat dry. Preheat oven and cast iron skillet under broiler for 15-20 mins. Season steaks. 2. Sear & Bake: Carefully place steaks in hot skillet, broil 3 mins per side. Switch oven to 500°F, cook to desired doneness (see chart), flipping halfway.3. Rest & Serve: Rest steaks for 5 minutes before serving, ideally on warm plates. (Crucial: Don't pierce meat during cooking)."
    },
    "cheesy tuber meat": {
        "file": "images/Cheese_Potato_and_Sausage_Casserole.jpg",
        "instruction": "1. Prep Meat & Potatoes: Brown chopped smoked sausage. Combine with cooked, diced potatoes in a casserole dish. 2. Make Cheese Sauce: Mix and heat remaining ingredients (except cheddar/paprika) until smooth; pour over sausage/potatoes.3. Top & Bake: Sprinkle with cheddar and paprika. Bake at 350°F for 35-45 minutes until golden."
    }
}

image_dir = '/kaggle/input/intern-data-new/images (2)/'
messages = []


messages.append({
    "role": "system",
    "content": [{
        "type": "text",
        "text": """You are an Expert Culinary Assistant. Your sole task is to convert a food image and a vague or imprecise dish title into an ultra-concise, 2-3 step recipe summary.

**Your Response MUST Adhere Strictly to These Rules:**

1.  **Structure & Length:**
    * **Exactly 2 or 3 steps.** No more, no less.
    * Each step must be a **single, complete sentence.**

2.  **Content & Style:**
    * Describe essential cooking actions using **present-tense, imperative verbs** (e.g., "Mix ingredients...", "Sauté vegetables...", "Bake until golden...").
    * Focus entirely on the **core assembly and cooking method.** What does one *do* to make this?

3.  **What to OMIT (Critically Important):**
    * **NO introductory phrases, chit-chat, or concluding remarks.** Get straight to the steps.

4.  **Goal:** The recipe summary must be **clear, unambiguous, actionable,** and capture the fundamental essence of how the dish is prepared. Assume the user has basic cooking knowledge and can infer common sense details.
"""
    }]
})

for vague_title, data in recipes.items():
    img = Image.open(f"{image_dir}/{data['file']}")
    messages.append({
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": f'Vague Title: "{vague_title}"\nInstruction:'}
        ]
    })
    messages.append({
        "role": "assistant",
        "content": [
            {"type": "text", "text": data['instruction']}
        ]
    })


test_image_path = f"{image_dir}images/"+str(input("Enter Image Name with the extension"))
test_vague_title = str(input("Enter Vague Title of the dish"))

print()
print()

test_img = Image.open(test_image_path)
messages.append({
    "role": "user",
    "content": [
        {"type": "image", "image": test_img},
        {"type": "text", "text": f'Vague Title: "{test_vague_title}"\nInstruction:'}
    ]
})

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
).to(model.device, dtype=torch.bfloat16)

input_len = inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**inputs, max_new_tokens=1000, do_sample=True, top_p=0.8, top_k=50)
    generation = generation[0][input_len:]

decoded = processor.decode(generation, skip_special_tokens=True)
print(decoded)
