import os
import argparse
import logging
import cv2

def preprocess_image(input_path, output_path, size):
    try:
        img = cv2.imread(input_path)
        img = cv2.resize(img, size)
        cv2.imwrite(output_path, img)
        logging.info(f"Processed {input_path} -> {output_path}")
    except Exception as e:
        logging.error(f"Failed to process {input_path}: {e}")

def batch_preprocess(input_dir, output_dir, size):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if fname.lower().endswith('.jpg'):
            input_path = os.path.join(input_dir, fname)
            output_path = os.path.join(output_dir, fname)
            preprocess_image(input_path, output_path, size)

def main():
    parser = argparse.ArgumentParser(description='Preprocess JPEG images (resize, normalize).')
    parser.add_argument('--input_dir', required=True, help='Input directory with JPEGs')
    parser.add_argument('--output_dir', required=True, help='Output directory for processed images')
    parser.add_argument('--size', type=int, nargs=2, default=[224, 224], help='Target size (width height)')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    batch_preprocess(args.input_dir, args.output_dir, tuple(args.size))

if __name__ == '__main__':
    main() 