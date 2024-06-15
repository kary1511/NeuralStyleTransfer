from flask import Flask, request, jsonify, send_from_directory
import os
import subprocess

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
GENERATED_FOLDER = 'generated'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_file():
    content_image = request.files['contentImage']
    style_image = request.files['styleImage']
    
    content_path = os.path.join(UPLOAD_FOLDER, 'content.jpg')
    style_path = os.path.join(UPLOAD_FOLDER, 'style.jpg')
    content_image.save(content_path)
    style_image.save(style_path)
    
    # Run the Jupyter Notebook script
    subprocess.run(['jupyter', 'nbconvert', '--to', 'notebook', '--execute', '--inplace', 'style_transfer.ipynb'])
    
    return jsonify({'generatedImage': 'output.jpg'})

@app.route('/generated/<filename>')
def send_generated(filename):
    return send_from_directory(GENERATED_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
