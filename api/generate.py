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
                "accept": "image/*",  # Expect an image as response
            },
            files={
                "prompt": (None, prompt),
                "output_format": (None, "webp"),  # Specify desired output format
            },
        )

        # Handle the response
        if response.status_code == 200:
            # Save the image temporarily
            temp_path = "/tmp/generated_image.webp"
            with open(temp_path, "wb") as file:
                file.write(response.content)
            print(f"Image generated and saved temporarily")
            
            # Here you would typically upload this image to a cloud storage
            # and return the URL. For now, we'll return the local path
            return temp_path
        else:
            print(f"Failed to generate image: {response.status_code}")
            print("Response content:", response.json())  # Print error details
            return None
    except Exception as e:
        print(f"Error during image generation: {e}")
        return None

# Step 4: Generate video from the image using Runway Gen-3 API
def generate_video(runway_api_key, image_path):
    client = RunwayML(api_key=runway_api_key)

    try:
        # Read the image file
        with open(image_path, 'rb') as image_file:
            # Create a video generation task
            task = client.image_to_video.create(
                model="gen3a_turbo",
                prompt_image=image_file,
                prompt_text="Transform this image into a dynamic scene",
                duration=5
            )
            
        print(f"Task started successfully. Task ID: {task.id}")

        # Polling for task status
        while True:
            time.sleep(5)
            task_status = client.tasks.retrieve(id=task.id)
            print(f"Current status: {task_status.status}")
            
            if task_status.status in ["SUCCEEDED", "FAILED", "CANCELLED"]:
                break
        
        if task_status.status == "SUCCEEDED":
            print("Video generation succeeded!")
            return task_status.output
        else:
            print("Video generation failed or was cancelled.")
            print("Status:", task_status.status)
            return None

    except Exception as e:
        print(f"Error creating video task: {e}")
        return None

@app.route('/api/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        base_prompt = data.get('base_prompt')
        modified_prompt = modify_prompt(base_prompt)

        # Get API keys from environment variables
        stability_api_key = os.environ.get("STABILITY_API_KEY")
        if not stability_api_key:
            stability_api_key = "sk-MzpLjHNnkypPMHzoAaEcER5k57t1Tfc34YKOW2cIN2MwMjIf"  # Fallback API key
        
        runway_api_key = os.environ.get("RUNWAY_API_KEY")
        
        # Generate the image
        image_path = generate_image_stability(stability_api_key, modified_prompt)

        if not image_path:
            return jsonify({"error": "Image generation failed"}), 500

        # Generate video if image generation is successful
        video_url = generate_video(runway_api_key, image_path)
        
        # Clean up the temporary file
        if os.path.exists(image_path):
            os.remove(image_path)

        if video_url:
            return jsonify({
                "image_path": image_path,
                "video_url": video_url
            })
        else:
            return jsonify({"error": "Video generation failed"}), 500

    except Exception as e:
        print(f"Error in generate endpoint: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
