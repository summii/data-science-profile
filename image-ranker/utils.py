import os


def get_image_file_paths(image_root_path="data"):
    
    image_file_paths = []
    
    
    for root, dirs, filenames in os.walk(image_root_path):
        
        filenames = sorted(filenames)
        
        for filename in filenames:
            input_path = os.path.abspath(root)
            file_path = os.path.join(input_path, filename)
            
            file_extension = filename.split(".")[-1]
            if file_extension.lower() in ("png", "jpg", "jpeg"):
                image_file_paths.append(file_path)
                
        break
        
    return image_file_paths