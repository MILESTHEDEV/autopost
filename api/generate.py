import requests
import base64
import os
import time
from runwayml import RunwayML
from transformers import pipeline
from flask import Flask, request, jsonify  # Add Flask imports

app = Flask(__name__)  # Initialize Flask app

# Function to load the Llama 2 model
def load_llama_model():
    try:
        print("Loading Llama 2 model...")
        generator = pipeline("text-generation", model="meta-llama/Llama-2-7b-hf")
        print("Llama 2 loaded successfully.")
        return generator
    except Exception as e:
        print(f"Failed to load Llama 2: {e}")
        return None

# Step 1: Ask user if they want to use Llama
use_llama = input("Do you want to use Llama 2 for text generation? (yes/no): ").strip().lower()
generator = load_llama_model() if use_llama == "yes" else None

# Input: User's base prompt
base_prompt = input("Enter your base prompt: ")

# Step 2: Modify the prompt for image generation
if generator:
    enriched_prompt = generator(
        f"Enhance this prompt for image generation: {base_prompt}",
        max_length=50
    )
    modified_prompt = enriched_prompt[0]['generated_text']
else:
    modified_prompt = f"{base_prompt}, detailed, vibrant, high resolution"
    print("Using fallback prompt modification.")

print(f"Modified Prompt: {modified_prompt}")

# Step 3: Generate the image using Stability AI
def generate_image_stability(api_key, prompt):
    try:
        response = requests.post(
            "https://api.stability.ai/v2beta/stable-image/generate/ultra",
            headers={
                "authorization": f"Bearer {api_key}",
                "accept": "image/*",
            },
            files={
                "prompt": (None, prompt),
                "output_format": (None, "webp"),
            },
        )

        if response.status_code == 200:
            image_url = response.json().get('url')  # Get the image URL from the response
            print(f"Image generated and available at {image_url}")
            return image_url  # Return the image URL
        else:
            print(f"Failed to generate image: {response.status_code}")
            print(response.json())
            return None
    except Exception as e:
        print(f"Error during image generation: {e}")
        return None

# Step 4: Generate video from the image using Runway Gen-3 API
# Define the generate_video function
def generate_video(runway_api_key, image_url):
    client = RunwayML(api_key=runway_api_key)  # Initialize the RunwayML client

    # Specify the model and inputs
    model = "gen3a_turbo"
    prompt_image_url = image_url  # Replace with your image URL
    prompt_text = "The bunny is eating a carrot"  # Describe your video content
    duration = 5  # Video duration in seconds (accepted values: 5 or 10)

    # Create a video generation task
    try:
        task = client.image_to_video.create(
            model=model,
            prompt_image=prompt_image_url,
            prompt_text=prompt_text,
            duration=duration  # Set the duration
        )
        print(f"Task started successfully. Task ID: {task.id}")

        # Polling for task status
        while True:
            time.sleep(5)  # Wait for 5 seconds before checking the status
            task_status = client.tasks.retrieve(id=task.id)
            
            print(f"Current status: {task_status.status}")
            
            if task_status.status in ["SUCCEEDED", "FAILED", "CANCELLED"]:
                break
        
        if task_status.status == "SUCCEEDED":
            print("Video generation succeeded!")
            print("Output URLs:", task_status.output)
            return task_status.output[0]  # Return the video URL
        elif task_status.status == "FAILED":
            print("Video generation failed.")
            print("Failure reason:", task_status.failure)
        else:
            print("Video generation was cancelled.")

    except Exception as e:
        print(f"Error creating task: {e}")

# Step 5: Generate the image
api_key = os.getenv("STABILITY_API_KEY")  # Use an environment variable for the Stability API key
if not api_key:
    api_key = input("Stability API key is required. Please enter it: ")  # Prompt user for the API key

image_url = generate_image_stability(api_key, modified_prompt)

# Step 6: Generate video if image generation is successful
if image_url:
    runway_api_key = os.getenv("RUNWAY_API_KEY")  # Use an environment variable for the Runway API key
    if not runway_api_key:
        runway_api_key = input("Runway API key is required. Please enter it: ")  # Prompt user for the API key
    video_url = generate_video(runway_api_key, image_url)  # Pass the image URL to video generation
    print(f"Generated Image URL: {image_url}")
    print(f"Generated Video URL: {video_url}")
else:
    print("Image generation failed. Skipping video generation.")

# Step 7: Generate a caption for the post
if generator:
    caption_prompt = f"Write a short and catchy caption for this post: {base_prompt}"
    caption_response = generator(caption_prompt, max_length=30)
    caption = caption_response[0]['generated_text']
else:
    caption = f"Check out this amazing image and video based on: '{base_prompt}'!"

print(f"Generated Caption: {caption}")

# Final Output
print("\nWorkflow Complete!")
print(f"Generated Image URL: {image_url}")
print(f"Generated Video URL: {video_url}")
print(f"Generated Caption: {caption}")

@app.route('/api/generate', methods=['POST'])
def generate():
    data = request.json
    base_prompt = data.get('base_prompt')
    use_llama = data.get('use_llama')

    generator = load_llama_model() if use_llama == "yes" else None

    # Modify the prompt for image generation
    if generator:
        enriched_prompt = generator(
            f"Enhance this prompt for image generation: {base_prompt}",
            max_length=50
        )
        modified_prompt = enriched_prompt[0]['generated_text']
    else:
        modified_prompt = f"{base_prompt}, detailed, vibrant, high resolution"
        print("Using fallback prompt modification.")

    # Generate the image
    api_key = os.getenv("STABILITY_API_KEY")
    image_url = generate_image_stability(api_key, modified_prompt)

    # Generate video if image generation is successful
    if image_url:
        runway_api_key = os.getenv("RUNWAY_API_KEY")
        video_url = generate_video(runway_api_key, image_url)  # Pass the image URL to video generation
        return jsonify({"image_url": image_url, "video_url": video_url})
    else:
        return jsonify({"error": "Image generation failed."}), 500

if __name__ == "__main__":
    app.run()
