from flask import Flask, request, jsonify, Response, render_template

import io
import os
import logging
import tempfile
import argparse
import numpy as np
from PIL import Image

from models.Embedding import Embedding
from models.Alignment import Alignment
from models.Blending import Blending

parser = argparse.ArgumentParser(description='Barbershop')

# I/O arguments
parser.add_argument('--input_dir', type=str, default='input/face',
                    help='The directory of the images to be inverted')
parser.add_argument('--output_dir', type=str, default='output',
                    help='The directory to save the latent codes and inversion images')
parser.add_argument('--im_path1', type=str, default='16.png', help='Identity image')
parser.add_argument('--im_path2', type=str, default='15.png', help='Structure image')
parser.add_argument('--im_path3', type=str, default='117.png', help='Appearance image')
parser.add_argument('--sign', type=str, default='realistic', help='realistic or fidelity results')
parser.add_argument('--smooth', type=int, default=1, help='dilation and erosion parameter')
# StyleGAN2 setting
parser.add_argument('--size', type=int, default=1024)
parser.add_argument('--ckpt', type=str, default="pretrained_models/ffhq.pt")
parser.add_argument('--channel_multiplier', type=int, default=2)
parser.add_argument('--latent', type=int, default=512)
parser.add_argument('--n_mlp', type=int, default=8)
# Arguments
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--tile_latent', action='store_true', help='Whether to forcibly tile the same latent N times')
parser.add_argument('--opt_name', type=str, default='adam', help='Optimizer to use in projected gradient descent')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate to use during optimization')
parser.add_argument('--lr_schedule', type=str, default='fixed', help='fixed, linear1cycledrop, linear1cycle')
parser.add_argument('--save_intermediate', action='store_true',
                    help='Whether to store and save intermediate HR and LR images during optimization')
parser.add_argument('--save_interval', type=int, default=400, help='Latent checkpoint interval')
parser.add_argument('--verbose', action='store_true', help='Print loss information')
parser.add_argument('--seg_ckpt', type=str, default='pretrained_models/seg.pth')
# Embedding loss options
parser.add_argument('--percept_lambda', type=float, default=1.0, help='Perceptual loss multiplier factor')
parser.add_argument('--l2_lambda', type=float, default=1.0, help='L2 loss multiplier factor')
parser.add_argument('--p_norm_lambda', type=float, default=0.001, help='P-norm Regularizer multiplier factor')
parser.add_argument('--l_F_lambda', type=float, default=0.5, help='L_F loss multiplier factor')
parser.add_argument('--W_steps', type=int, default=10, help='Number of W space optimization steps')
parser.add_argument('--FS_steps', type=int, default=50, help='Number of FS space optimization steps')
# Alignment loss options
parser.add_argument('--ce_lambda', type=float, default=1.0, help='cross entropy loss multiplier factor')
parser.add_argument('--style_lambda', type=str, default=4e4, help='style loss multiplier factor')
parser.add_argument('--align_steps1', type=int, default=30, help='')
parser.add_argument('--align_steps2', type=int, default=30, help='')
# Blend loss options
parser.add_argument('--face_lambda', type=float, default=2, help='')
parser.add_argument('--hair_lambda', type=str, default=10.0, help='')
parser.add_argument('--blend_steps', type=int, default=100, help='')

args = parser.parse_args()
ii2s = Embedding(args)
# model_parser = get_parser()
# model_args, _ = model_parser.parse_known_args()
# hair_fast = HairFast(model_args)

def resize_image(image, target_size=(1024, 1024)):
    """Resize the image to the target size (e.g., 1024x1024)."""
    image = image.resize(target_size, Image.LANCZOS)
    return image

def ensure_rgb(image):
    """Ensure the image has 3 channels (RGB). If the image has an alpha channel (RGBA), it will be converted to RGB."""
    if image.mode == 'RGBA':  # Check if the image has an alpha channel
        image = image.convert('RGB')  # Remove alpha channel and convert to RGB
    return image

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/wig_stick', methods=['POST'])
def wig_stick():
    # Check if the post request has the files
    if 'face' not in request.files or 'shape' not in request.files:
        return jsonify({"message": "Missing one or more required files (face, shape, color)"}), 400

    # Get the files from the request
    face_file = request.files['face']
    shape_file = request.files['shape']

    # If color is part of the input, handle it
    color_file = shape_file  # Using shape as color if color is not provided

    # Check if files are valid (you can add other checks here like file extensions)
    if face_file.filename == '' or shape_file.filename == '':
        return jsonify({"message": "No selected file"}), 400

    try:
        # Log the file size and content type
        logging.info(f"Received face file: {face_file.filename}, MIME type: {face_file.content_type}, size: {len(face_file.read())} bytes")
        face_file.stream.seek(0)  # Reset the pointer after logging the file size

        logging.info(f"Received shape file: {shape_file.filename}, MIME type: {shape_file.content_type}, size: {len(shape_file.read())} bytes")
        shape_file.stream.seek(0)  # Reset the pointer

        if color_file:
            logging.info(f"Received color file: {color_file.filename}, MIME type: {color_file.content_type}, size: {len(color_file.read())} bytes")
            color_file.stream.seek(0)  # Reset the pointer

        # Try opening the files
        face_image = Image.open(io.BytesIO(face_file.read()))
        shape_image = Image.open(io.BytesIO(shape_file.read()))
        color_image = shape_image

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f1, \
            tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f2:
            face_image.save(f1.name)
            shape_image.save(f2.name)

            im_path1 = f1.name
            im_path2 = f2.name

            im_set = {im_path1, im_path2, im_path2}

            ii2s.invert_images_in_W([*im_set])
            ii2s.invert_images_in_FS([*im_set])
            align = Alignment(args)
            final_pil_image = align.align_images_2(im_path1, im_path2, sign=args.sign, align_more_region=False, smooth=args.smooth)

        # Resize all images to a consistent size (1024x1024)
        # target_size = (1024, 1024)  # Resize to a fixed size like 1024x1024
        # face_image = resize_image(ensure_rgb(face_image), target_size)
        # shape_image = resize_image(ensure_rgb(shape_image), target_size)
        # if color_file:
        #     color_image = resize_image(ensure_rgb(color_image), target_size)

        # # Log the resized image dimensions
        # logging.info(f"Resized Face image size: {face_image.size}, Shape image size: {shape_image.size}")
        # if color_file:
        #     logging.info(f"Resized Color image size: {color_image.size}")

        # # Process the images here (e.g., hair swapping)
        # # final_tensor = hair_fast.swap(face_image, shape_image, color_image)
        # final_tensor = hair_fast.swap(face_image, shape_image, color_image, align=True)

        # # Convert tensor to PIL Image
        # final_image = final_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        # final_image = (final_image * 255).astype(np.uint8)
        # final_pil_image = Image.fromarray(final_image)

        # Convert the output image to a byte stream
        img_byte_arr = io.BytesIO()
        final_pil_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        # Return the final image as a response
        return Response(img_byte_arr, mimetype='image/png')

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify({"message": "Error in hair swapping", "error": str(e)}), 500
    
@app.route('/get_wig', methods=['POST'])
def get_wig():
    # Only check for 'face' since shape is loaded from path
    if 'face' not in request.files:
        return jsonify({"message": "Missing required file: face"}), 400

    face_file = request.files['face']
    choose_file = request.form.get('shape', '1')
    align = request.form.get('align', '1')
    align = align=='1'
    files = os.listdir("static/wig")
    if face_file.filename == '' or f"{choose_file}.png" not in files:
        return jsonify({"message": "No selected face file"}), 400

    try:
        # Open input images
        face_image = Image.open(io.BytesIO(face_file.read()))
        shape_image = Image.open(f"static/wig/{choose_file}.png")

        # Resize images
        target_size = (1024, 1024)
        face_image = resize_image(ensure_rgb(face_image), target_size)
        shape_image = resize_image(ensure_rgb(shape_image), target_size)
        color_image = shape_image  # Use shape as color too

        # Call your hair swap function
        # final_tensor = hair_fast.swap(face_image, shape_image, color_image, align=align)

        # # Convert tensor to PIL
        # final_image = final_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        # final_image = (final_image * 255).astype(np.uint8)
        # final_pil_image = Image.fromarray(final_image)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f1, \
            tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f2:
            face_image.save(f1.name)
            shape_image.save(f2.name)

            im_path1 = f1.name
            im_path2 = f2.name

            im_set = {im_path1, im_path2, im_path2}

            ii2s.invert_images_in_W([*im_set])
            ii2s.invert_images_in_FS([*im_set])
            align = Alignment(args)
            final_pil_image = align.align_images_2(im_path1, im_path2, sign=args.sign, align_more_region=False, smooth=args.smooth)

            # Prepare response
            img_byte_arr = io.BytesIO()
            final_pil_image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)

            return Response(img_byte_arr, mimetype='image/png')

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify({"message": "Error in hair swapping", "error": str(e)}), 500
    
@app.route('/web', methods=['GET'])
def web():
    wig_dir = 'static/wig'
    files = [os.path.splitext(f)[0] for f in os.listdir(wig_dir) if f.lower().endswith(('.png'))]
    return render_template('web.html', wig_files=files)

@app.route('/temp', methods=['GET'])
def temp():
    wig_dir = 'static/wig'
    files = [os.path.splitext(f)[0] for f in os.listdir(wig_dir) if f.lower().endswith(('.png'))]
    return render_template('temp.html', wig_files=files)

@app.route('/camera', methods=['GET'])
def camera():
    wig_dir = 'static/wig'
    files = [os.path.splitext(f)[0] for f in os.listdir(wig_dir) if f.lower().endswith(('.png'))]
    return render_template('camera.html', wig_files=files)

@app.route('/get_wig_list', methods=['GET'])
def get_wig_list():
    wig_dir = 'static/wig'
    domain = request.host_url.rstrip('/')
    files = [dict(id=os.path.splitext(f)[0], url=f"{domain}/{wig_dir}/{f}") for f in os.listdir(wig_dir) if f.lower().endswith(('.png'))]
    return files    

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000,debug=True)