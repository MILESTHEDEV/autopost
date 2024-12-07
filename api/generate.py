import requests
import os
import time
from runwayml import RunwayML
from flask import Flask, request, jsonify

app = Flask(__name__)

# Step 1: Modify the prompt for image generation
def modify_prompt(base_prompt):
    return f"{base_prompt}, detailed, vibrant, high resolution"

# Step 2: Generate the image using Stability AI
def generate_image_stability(api_key, prompt):
    try:
        print(f"Using API Key: {api_key}")  # Debugging line to check the API key
        response = requests.post(
            "https://api.stability.ai/v2beta/stable-image/generate/ultra",
            headers={
                "authorization": f"Bearer {api_key}",
                "accept": "image/*"  # Set to receive any image format
            },
            files={
                "prompt": (None, prompt),  # Send the prompt as a file
                "output_format": (None, "jpeg"),  # Specify JPEG as the output format
            },
        )

        # Handle the response
        if response.status_code == 200:
            image_url = response.json().get('url')  # Get the image URL from the response
            print(f"Image generated and available at {image_url}")
            return image_url  # Return the image URL
        else:
            print(f"Failed to generate image: {response.status_code}")
            print("Response content:", response.json())  # Print error details
            return None
    except Exception as e:
        print(f"Error during image generation: {e}")
        return None

# Step 4: Generate video from the image using Runway Gen-3 API
def generate_video(runway_api_key, prompt_image_url):
    client = RunwayML(api_key=runway_api_key)  # Initialize the RunwayML client

    # Specify the model and inputs
    model = "gen3a_turbo"
    prompt_text = "The bunny is eating a carrot"  # Describe your video content
    duration = 5  # Video duration in seconds (accepted values: 5 or 10)

    # Create a video generation task
    try:
        task = client.image_to_video.create(
            model=model,
            prompt_image=prompt_image_url,  # Use the image URL here
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
            return task_status.output  # Return the output URLs
        elif task_status.status == "FAILED":
            print("Video generation failed.")
            print("Failure reason:", task_status.failure)
            return None
        else:
            print("Video generation was cancelled.")
            return None

    except Exception as e:
        print(f"Error creating task: {e}")
        return None

@app.route('/api/generate', methods=['POST'])
def generate():
    data = request.json
    base_prompt = data.get('base_prompt')
    modified_prompt = modify_prompt(base_prompt)

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
