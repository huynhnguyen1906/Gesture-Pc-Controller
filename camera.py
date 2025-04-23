import cv2

def get_available_cameras():
    """
    Detect available cameras in the system
    Returns a list of available cameras with their corresponding indexes
    """
    available_cameras = []
    max_cameras_to_check = 10  # Limit the number of cameras to check
    
    print("Searching for available cameras...")
    
    for i in range(max_cameras_to_check):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Using CAP_DSHOW to speed up connection on Windows
        if cap.isOpened():
            # Read a frame to verify the camera is working
            ret, frame = cap.read()
            if ret:
                # Get camera information if possible
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                available_cameras.append({
                    'index': i,
                    'width': width,
                    'height': height,
                    'fps': fps,
                    'name': f"Camera {i}"
                })
            cap.release()
    
    return available_cameras

def select_camera():
    """
    Display list of available cameras and allow user to select one
    Returns information about the selected camera
    """
    cameras = get_available_cameras()
    
    if not cameras:
        print("No cameras found in the system!")
        return {'index': 0, 'width': 640, 'height': 480, 'fps': 30}
    
    print("\n=== AVAILABLE CAMERAS ===")
    for i, camera in enumerate(cameras):
        print(f"{i+1}. {camera['name']} ({camera['width']}x{camera['height']}, {camera['fps']} FPS)")
    
    choice = -1
    while choice < 0 or choice >= len(cameras):
        try:
            choice = int(input(f"\nSelect camera (1-{len(cameras)}): ")) - 1
            if choice < 0 or choice >= len(cameras):
                print(f"Please choose a number from 1 to {len(cameras)}")
        except ValueError:
            print("Please enter a valid number")
    
    selected_camera = cameras[choice]
    print(f"Selected: {selected_camera['name']} ({selected_camera['width']}x{selected_camera['height']}, {selected_camera['fps']} FPS)")
    
    # Ask user if they want to increase resolution
    try:
        use_high_res = input("Do you want to use higher resolution? (y/n): ").lower() == 'y'
        if use_high_res:
            res_options = [
                (1280, 720),
                (1920, 1080),
            ]
            
            print("\nAvailable resolutions:")
            for i, res in enumerate(res_options):
                print(f"{i+1}. {res[0]}x{res[1]}")
                
            res_choice = -1
            while res_choice < 0 or res_choice >= len(res_options):
                try:
                    res_choice = int(input(f"Select resolution (1-{len(res_options)}): ")) - 1
                    if res_choice < 0 or res_choice >= len(res_options):
                        print(f"Please choose a number from 1 to {len(res_options)}")
                except ValueError:
                    print("Please enter a valid number")
            
            selected_camera['width'] = res_options[res_choice][0]
            selected_camera['height'] = res_options[res_choice][1]
    except Exception as e:
        print(f"Error setting resolution: {e}")
        print("Using default resolution.")
    
    return selected_camera

def initialize_camera(camera_index, width, height):
    """Initialize the camera with the given index and resolution."""
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 60)  # Request 60 FPS
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce camera buffer

    if not cap.isOpened():
        raise ValueError(f"Error: Cannot open camera with index {camera_index}.")

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = int(cap.get(cv2.CAP_PROP_FPS))

    return cap, actual_width, actual_height, actual_fps