#!/usr/bin/env python

import sys
import os
from PIL import Image, ExifTags

def get_orientation(exif):
    for tag, value in exif.items():
        if tag == 'Orientation':
            if value is not None:
                return {
                    1: 'Horizontal (normal)',
                    2: 'Mirrored horizontal',
                    3: 'Rotated 180°',
                    4: 'Mirrored vertical',
                    5: 'Mirrored horizontal then rotated 90° CCW',
                    6: 'Rotated 90° CW',
                    7: 'Mirrored horizontal then rotated 90° CW',
                    8: 'Rotated 90° CCW',
                }.get(value, 'Unknown')
    return None

def get_exif_data(img):
    exif_data = {}
    exif_raw = img._getexif()
    if exif_raw:
        for tag, value in exif_raw.items():
            decoded = ExifTags.TAGS.get(tag, tag)
            exif_data[decoded] = value
    return exif_data

def get_gps_info(exif_data):
    gps_info = exif_data.get('GPSInfo', None)
    if gps_info:
        gps_data = {}
        for key in gps_info.keys():
            decoded = ExifTags.GPSTAGS.get(key, key)
            gps_data[decoded] = gps_info[key]
        return gps_data
    return None

def convert_to_degrees(value):
    d = float(value[0])
    m = float(value[1])
    s = float(value[2])
    return d + (m / 60.0) + (s / 3600.0)

def get_lat_lon(gps_data):
    lat = lon = None
    if 'GPSLatitude' in gps_data and 'GPSLatitudeRef' in gps_data and \
       'GPSLongitude' in gps_data and 'GPSLongitudeRef' in gps_data:

        lat = convert_to_degrees(gps_data['GPSLatitude'])
        if gps_data['GPSLatitudeRef'] != 'N':
            lat = -lat

        lon = convert_to_degrees(gps_data['GPSLongitude'])
        if gps_data['GPSLongitudeRef'] != 'E':
            lon = -lon

    return lat, lon

# Skeletal functions as per your request

def convert_to_bitmap(img):
    # Placeholder for converting image to bitmap
    print(" - Converting image to bitmap...")
    # Convert image to black and white (1-bit pixels)
    return img.convert('1')

def vectorize_image(bitmap):
    # Placeholder for vectorizing the bitmap image
    print(" - Vectorizing image...")
    # For now, just return a placeholder vector
    vector = []
    return vector

def deduplicate_image(vector):
    # Placeholder for deduplicating the image data
    print(" - Deduplicating image...")
    # For now, just return the vector as is
    return vector

def connectivity_analysis(vector):
    # Placeholder for performing connectivity analysis
    print(" - Performing connectivity analysis...")
    # For now, just return a placeholder result
    return None

def solution(vector):
    # Placeholder for final solution based on analysis
    print(" - Computing final solution...")
    # For now, just return a placeholder
    return None

def main():
    if len(sys.argv) != 2:
        print("Usage: python app.py <path_to_images>")
        sys.exit(1)

    path = sys.argv[1]

    if not os.path.exists(path):
        print(f"Error: Path '{path}' does not exist.")
        sys.exit(1)

    if not os.path.isdir(path):
        print(f"Error: Path '{path}' is not a directory.")
        sys.exit(1)

    files = os.listdir(path)
    image_count = 0

    for filename in files:
        fullpath = os.path.join(path, filename)

        if os.path.isfile(fullpath):
            try:
                with Image.open(fullpath) as img:
                    img.verify()  # Verify that it's an image

                # Re-open the image for processing after verify
                with Image.open(fullpath) as img:
                    image_count += 1

                    # Get image size
                    width, height = img.size

                    # Get image format
                    img_format = img.format

                    # Get color mode
                    color_mode = img.mode

                    # Get file size
                    file_size = os.path.getsize(fullpath)

                    # Initialize rotation to None
                    rotation = None

                    # Try to get EXIF data
                    exif = get_exif_data(img)
                    if exif:
                        rotation = get_orientation(exif)

                        # Additional EXIF data
                        dpi = img.info.get('dpi', (None, None))
                        make = exif.get('Make', 'Unknown')
                        model = exif.get('Model', 'Unknown')
                        date_time = exif.get('DateTime', 'Unknown')
                        iso = exif.get('ISOSpeedRatings', 'Unknown')
                        exposure_time = exif.get('ExposureTime', 'Unknown')
                        f_number = exif.get('FNumber', 'Unknown')

                        # GPS Data
                        gps_data = get_gps_info(exif)
                        if gps_data:
                            latitude, longitude = get_lat_lon(gps_data)
                        else:
                            latitude = longitude = None
                    else:
                        dpi = (None, None)
                        make = model = date_time = iso = exposure_time = f_number = None
                        latitude = longitude = None

                    # Print image statistics
                    print(f"Image: {filename}")
                    print(f" - Resolution: {width}x{height}")
                    print(f" - Format: {img_format}")
                    print(f" - Color Mode: {color_mode}")
                    print(f" - File Size: {file_size} bytes")
                    if dpi[0] and dpi[1]:
                        print(f" - DPI: {dpi[0]}x{dpi[1]}")
                    else:
                        print(" - DPI: Not available")
                    if rotation:
                        print(f" - Rotation: {rotation}")
                        if rotation != 'Horizontal (normal)':
                            print(f"*** Incompatible Rotation ***")
                    else:
                        print(" - Rotation: Not available")
                    if width > height:
                        print(" - Orientation: Landscape")
                    else:
                        print(" - Orientation: Portrait")
                        print("*** Incompatible Orientation: THIS IS A PORTRAIT IMAGE ***")
                    print(f" - Camera Make: {make}")
                    print(f" - Camera Model: {model}")
                    print(f" - Date/Time Taken: {date_time}")
                    print(f" - ISO Speed: {iso}")
                    print(f" - Exposure Time: {exposure_time}")
                    print(f" - F-Number: {f_number}")
                    if latitude and longitude:
                        print(f" - GPS Coordinates: ({latitude}, {longitude})")
                    else:
                        print(" - GPS Coordinates: Not available")

                    # New functionality
                    # Convert image to bitmap
                    bitmap = convert_to_bitmap(img)
                    # Vectorize the bitmap image
                    vector = vectorize_image(bitmap)
                    # Deduplicate the vectorized image data
                    deduped_vector = deduplicate_image(vector)
                    # Perform connectivity analysis
                    connectivity_result = connectivity_analysis(deduped_vector)
                    # Compute final solution
                    final_result = solution(deduped_vector)

            except Exception as e:
                print(f"Error processing image '{filename}':")
                raise e
                # The file is not an image or cannot be opened
                # Optionally, print or log the exception
                pass

    print(f"Total number of images found: {image_count}")

if __name__ == "__main__":
    main()