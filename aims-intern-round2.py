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
        "file": "images/train_images/30-Minute_Chicken_and_Dumplings.jpg",
        "instruction": "1. Dice chicken and set aside; wash hands; sauté vegetables, bay leaf, oil, and butter in a pot for 5 minutes, season with salt, pepper, and poultry seasoning, then add flour and cook for 2 minutes. 2. Add broth and bring to a boil; stir in chicken; mix biscuit mix with warm water and parsley, drop spoonfuls evenly spaced into the pot, cover tightly and steam on medium-low for 8–10 minutes. 3. Uncover, stir to thicken sauce, add peas, mix well, and serve hot in shallow bowls."
    },
    "buttery farm meat": {
        "file": "images/train_images/Amish_Chicken.jpg",
        "instruction": "Dredge chicken pieces in flour and spices, then place them skin side up in a baking dish. 2. Mix cream and water, pour over the chicken. 3. Bake at 350°F for 1.5 hours or until skin is golden brown."
    },
    "soft creamy swirls": {
        "file": "images/train_images/Angel_Chicken_Pasta.jpg",
        "instruction": "1. Preheat oven to 325°F; melt butter in saucepan, stir in Italian dressing, wine, soup, and cream cheese until smooth. 2. Arrange chicken in a foil-covered baking dish, pour sauce over chicken, cover with foil, and bake for 1 hour. 3. Cook angel hair pasta; serve chicken and sauce over pasta."
    },
    "soy noodle thing": {
        "file": "images/train_images/Asian_Noodle_Bowl.jpg",
        "instruction": "1. Boil water and cook noodles; toast almonds and sesame seeds separately until golden (about 3 minutes). 2. In a bowl, mix soy sauce, lime juice, peanut oil, and chili sauce to make dressing. 3. Drain noodles, toss with dressing, green onions, almonds, sesame seeds, and serve warm."
    },
    "spicy bean mess": {
        "file": "images/train_images/Award_Winning_Chili.jpg",
        "instruction": "1. Heat oil in Dutch oven, brown ground meat, add onions, peppers, and garlic, sauté until soft. 2. Stir in tomato paste, diced tomatoes, sugar, bouillon, beer, beans, and spices. 3. Cover and simmer for 2 hours, stirring occasionally; serve hot."
    },
    "pink baked slab": {
        "file": "images/train_images/Baked_Salmon.jpg",
        "instruction": "1. Mix garlic, oil, basil, salt, pepper, lemon juice, and parsley to prepare marinade. 2. Marinate salmon fillets skin-side down in fridge for 1 hour; preheat oven to 375°F. 3. Place salmon in baking dish, spoon marinade over, bake 35–45 minutes until flakes easily."
    },
    "dark sweet thighs": {
        "file": "images/train_images/Balsamic_Chicken_Thighs.jpg",
        "instruction": "Prep & Brown: Preheat sprayed pan. Season chicken thighs and brown well on all sides. Cook & Add Aromatics: Cover, reduce heat, cook ~25 mins until done. Add and soften shallots for 2-3 minutes. Sauce & Serve: Stir in balsamic vinegar, coat chicken, and serve with sauce spooned over."
    },
    "beefy crunchy pie": {
        "file": "images/train_images/Beef_Nacho_Crescent_Bake.jpg",
        "instruction": "1.Prepare the filling and dough: Cook ground beef with seasonings; whisk cheese soup and milk; cut crescent dough into squares. 2. Assemble and bake: Spoon beef onto dough, seal, place in a greased dish, and drizzle with soup mixture. Bake at 375°F for 20-25 minutes. 3. Finish and serve: Sprinkle with cheese and return to oven for 5 minutes until melted."
    },
    "fluffy malt loaf": {
        "file": "images/train_images/Beer_Bread.jpg",
        "instruction": "1. Mix & Bake: Preheat oven to 375°F and cool for 15 minutes. Mix sifted dry ingredients with beer, pour into greased pan, top with melted butter, and bake for 1 hour.2. Sift Flour Crucial: For best results (not a ""brick""), sift flour or spoon it lightly into the measuring cup.... 3. Tips & Variations: For softer crust, mix butter into batter. If using non-alcoholic liquid, add yeast."
    },
    "tomato bread bites": {
        "file": "images/train_images/Best_Ever_Bruschetta.jpg",
        "instruction": "1. Mix Ingredients: Combine tomatoes, garlic, basil, vinegar, olive oil, cheese, salt, and pepper in a bowl.2. Marinate: Let the mixture sit for at least 15 minutes to blend flavors.3. Serve: Slice bread and spoon the tomato mixture on top."
    },
    "garlicky red splash": {
        "file": "images/train_images/Bevs_Spaghetti_Sauce.jpg",
        "instruction": "Brown beef, onion, garlic with olive oil and all seasonings (bay leaves, oregano, basil, Italian seasoning, salt, pepper). Stir in tomato paste, sauce, and diced tomatoes; simmer covered for 1.5 hours. Top cooked spaghetti with sauce, sautéed mushrooms, and Parmesan."
    },
    "sweet sticky meat": {
        "file": "images/train_images/Bourbon_Chicken.jpg",
        "instruction": "1.Prepare the base: Sauté onion in butter, then make a roux with butter and flour, slowly adding chicken stock and simmering for 20 minutes. 2. Combine and simmer: Add broccoli, carrots, and sautéed onions, then milk and half-and-half, cooking on low heat for 20-25 minutes without boiling, and season. 3. Melt in cheese: Slowly stir in grated cheese until melted, then remove from heat and serve.."
    },
    "green cheesy bowl": {
        "file": "images/train_images/Broccoli_Cheese_Soup.jpg",
        "instruction": "1. Roux & Base: Sauté onions. Make a golden roux with butter/flour, then whisk in stock and simmer. 2. Veggies & Cream: Add broccoli, carrots, onions, milk, and half & half; cook on low (do not boil).3. Cheese & Serve: Season, blend if desired, then stir in cheese until melted. Serve immediately."
    },
    "juicy charred slab": {
        "file": "images/train_images/Broil_a_Perfect_Steak.jpg",
        "instruction": "Prep & Preheat: Bring steaks to room temp, pat dry. Preheat oven and cast iron skillet under broiler for 15-20 mins. Season steaks. Sear & Bake: Carefully place steaks in hot skillet, broil 3 mins per side. Switch oven to 500°F, cook to desired doneness (see chart), flipping halfway. Rest & Serve: Rest steaks for 5 minutes before serving, ideally on warm plates. (Crucial: Don't pierce meat during cooking)."
    },
    "cheesy tuber meat": {
        "file": "images/train_images/Cheese_Potato_and_Sausage_Casserole.jpg",
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


test_image_path = f"{image_dir}images/test_images/"+str(input("Enter Image Name with the extension"))
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
