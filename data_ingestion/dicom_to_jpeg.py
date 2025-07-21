import os
import argparse
import logging
import pydicom
import cv2
import numpy as np

def convert_dicom_to_jpeg(dicom_path, jpeg_path):
    try:
        ds = pydicom.dcmread(dicom_path)
        img = ds.pixel_array
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = np.uint8(img)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(jpeg_path, img)
        logging.info(f"Converted {dicom_path} -> {jpeg_path}")
    except Exception as e:
        logging.error(f"Failed to convert {dicom_path}: {e}")

def batch_convert(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if fname.lower().endswith('.dcm'):
            dicom_path = os.path.join(input_dir, fname)
            jpeg_path = os.path.join(output_dir, fname.replace('.dcm', '.jpg'))
            convert_dicom_to_jpeg(dicom_path, jpeg_path)

def main():
    parser = argparse.ArgumentParser(description='Convert DICOMs to JPEGs.')
    parser.add_argument('--input_dir', required=True, help='Input directory with DICOM files')
    parser.add_argument('--output_dir', required=True, help='Output directory for JPEGs')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    batch_convert(args.input_dir, args.output_dir)

if __name__ == '__main__':
    main() 