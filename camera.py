import cv2

def get_available_cameras():
    """
    Detect available cameras in the system
    Returns a list of available cameras with their corresponding indexes
    """
    available_cameras = []
    max_cameras_to_check = 20  # Increased limit to detect more cameras including virtual ones
    
    print("Searching for available cameras...")
    print("(This may take a few moments as we check for all camera devices...)")
    
    # First try to get camera names through DirectShow (Windows only)
    try:
        import subprocess
        import re
        import os
        
        # Use PowerShell to query DirectShow devices (works better for virtual cameras like DroidCam)
        if os.name == 'nt':  # Windows only
            try:
                cmd = "powershell -Command \"& {Get-CimInstance Win32_PnPEntity | Where-Object { $_.PNPClass -eq 'Camera' -or $_.PNPClass -eq 'Image' } | Select-Object Name | Format-Table -HideTableHeaders}\""
                result = subprocess.check_output(cmd, shell=True, text=True)
                device_names = [line.strip() for line in result.split('\n') if line.strip()]
                print(f"Found {len(device_names)} camera devices via system query:")
                for i, name in enumerate(device_names):
                    print(f"  - {name}")
            except Exception as e:
                print(f"Could not query camera devices via system: {e}")
                device_names = []
        else:
            device_names = []
    except Exception:
        device_names = []
    
    for i in range(max_cameras_to_check):
        # Try multiple backends to ensure we catch virtual cameras
        # First try DirectShow on Windows
        connected = False
        frame_read = False
        backend_used = None
        
        # Try multiple backends in order of reliability
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"), 
            (cv2.CAP_MSMF, "Media Foundation"),
            (0, "Default")  # 0 is default backend
        ]
        
        for backend_id, backend_name in backends:
            if backend_id == 0:  # Default backend
                cap = cv2.VideoCapture(i)
            else:
                try:
                    cap = cv2.VideoCapture(i, backend_id)
                except Exception:
                    continue
            
            if cap.isOpened():
                connected = True
                backend_used = backend_name
                
                # Read frame with retry and longer timeout for virtual cameras
                for attempt in range(3):  # Try up to 3 times
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        frame_read = True
                        break
                    # Small delay between attempts
                    import time
                    time.sleep(0.2)
                
                # Even if we can't read a frame, we still include the camera if opened
                # This is more permissive and helps with some virtual cameras
                
                # Get camera information
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS)) if int(cap.get(cv2.CAP_PROP_FPS)) > 0 else 30
                
                # Try to get camera name
                # For DroidCam and similar apps, try to identify them by name or resolution pattern
                camera_name = None
                
                # First check if we have device names from system query
                if i < len(device_names):
                    camera_name = device_names[i]
                    # Look for common virtual camera apps in the name
                    is_virtual = any(app.lower() in camera_name.lower() for app in 
                                    ["droidcam", "iriun", "epoccam", "virtual", "obs"]) 
                
                # If no name from system, create a descriptive one
                if not camera_name:
                    camera_name = f"Camera {i}"
                    if backend_used:
                        camera_name += f" ({backend_used})"
                    is_virtual = "virtual" in camera_name.lower() or not frame_read                # Attempt to detect if camera is virtual based on various heuristics
                if not 'is_virtual' in locals():
                    # Common resolutions for phone cameras acting as webcams
                    virtual_cam_resolutions = [(640, 480), (1280, 720), (480, 640), (720, 1280)]
                    is_virtual = (width, height) in virtual_cam_resolutions and fps >= 25
                
                # Properties to help user identify the right camera
                status = "Ready" if frame_read else "Connected (No frame yet)"
                
                available_cameras.append({
                    'index': i,
                    'width': width,
                    'height': height,
                    'fps': fps,
                    'name': camera_name,
                    'is_virtual': is_virtual,
                    'status': status,
                    'backend': backend_used
                })
                
                # If we got a valid camera with this backend, no need to try others
                break
                
            cap.release()
    
    # Sort cameras: put working cameras first, then virtual cameras (likely to be DroidCam)
    available_cameras.sort(key=lambda x: (0 if x['status'] == "Ready" else 1, 
                                         0 if x['is_virtual'] else 1))
    
    return available_cameras

def select_camera():
    """
    Display list of available cameras and allow user to select one
    Returns information about the selected camera
    """
    cameras = get_available_cameras()
    print("\n=== AVAILABLE CAMERAS ===")
    if cameras:
        for i, camera in enumerate(cameras):
            # Create a more descriptive display for each camera
            virtual_tag = "[VIRTUAL] " if camera.get('is_virtual', False) else ""
            status_tag = f"[{camera.get('status', 'Unknown')}]"
            backend_info = f"Backend: {camera.get('backend', 'Unknown')}" if 'backend' in camera else ""
            
            # Format camera name to highlight potential DroidCam or virtual cameras
            camera_name = camera['name']
            if "droidcam" in camera_name.lower():
                camera_name = f"DroidCam: {camera_name}"
            
            print(f"{i+1}. {virtual_tag}{camera_name} ({camera['width']}x{camera['height']}, {camera['fps']} FPS) {status_tag} {backend_info}")
    else:
        print("No local cameras detected.")
        # No additional options - only show detected cameras
    max_choice = len(cameras)
    
    if max_choice == 0:
        print("\nNo cameras detected. Please make sure your camera is connected properly.")
        print("Using default camera (0) as fallback.")
        return {'index': 0, 'width': 640, 'height': 480, 'fps': 30, 'name': 'Default Camera'}
    
    choice = -1
    while choice < 0 or choice >= max_choice:
        try:
            choice = int(input(f"\nSelect camera (1-{max_choice}): ")) - 1
            if choice < 0 or choice >= max_choice:
                print(f"Please choose a number from 1 to {max_choice}")
        except ValueError:
            print("Please enter a valid number")    # Normal local camera selection (including DroidCam)
    selected_camera = cameras[choice]
    print(f"Selected: {selected_camera['name']} ({selected_camera['width']}x{selected_camera['height']}, {selected_camera['fps']} FPS)")
      # Ask user to customize resolution
    try:
        # Different resolution options for DroidCam vs regular cameras
        is_droidcam = 'droidcam' in selected_camera.get('name', '').lower()
        
        # Determine if this is a virtual camera (like DroidCam)
        is_virtual = selected_camera.get('is_virtual', False) or is_droidcam
        
        if is_virtual:
            print("\n=== DROIDCAM/VIRTUAL CAMERA RESOLUTION SETTINGS ===")
            print("Note: DroidCam supports various resolutions depending on your phone model and settings.")
        else:
            print("\n=== CAMERA RESOLUTION SETTINGS ===")
        
        # Check if user wants to change resolution
        change_res = input("Do you want to customize resolution? (y/n): ").lower() == 'y'
        if change_res:            # Different resolution options based on camera type
            # For DroidCam, display preferred resolutions that it actually supports
            if is_virtual or is_droidcam:
                # Test the camera to determine what resolutions it actually supports
                # This approach ensures we only show resolutions that will work
                test_cap = None
                supported_resolutions = []
                
                try:
                    # Try to open the camera to test resolutions
                    test_cap = cv2.VideoCapture(selected_camera['index'])
                    
                    if test_cap.isOpened():
                        # Test each resolution
                        test_resolutions = [
                            (640, 480, "640x480 (Default - Most compatible)"),
                            (800, 600, "800x600 (Good compatibility)"),
                            (960, 720, "960x720 (DroidCam specific)"),
                            (1024, 768, "1024x768 (Extended compatibility)"),
                            (1280, 720, "1280x720 (HD)"),
                            (1920, 1080, "1920x1080 (Full HD)"),
                        ]
                        
                        print("\nTesting supported resolutions for DroidCam...")
                        
                        for w, h, desc in test_resolutions:
                            # Try to set resolution
                            test_cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                            test_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                            
                            # Give it time to apply
                            import time
                            time.sleep(0.2)
                            
                            # Check actual resolution
                            actual_w = int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            actual_h = int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            
                            # Try to read a frame with this resolution
                            frame_success = False
                            for _ in range(2):
                                try:
                                    ret, frame = test_cap.read()
                                    if ret and frame is not None and frame.size > 0:
                                        frame_success = True
                                        break
                                except Exception:
                                    pass
                                time.sleep(0.2)
                            
                            # Only add if the resolution was actually applied and we could read a frame
                            if frame_success:
                                # Use the actual resolution values, not the requested ones
                                res_desc = f"{actual_w}x{actual_h} ({desc.split('(')[1]}"
                                supported_resolutions.append((actual_w, actual_h, res_desc))
                                print(f"✓ Supported: {res_desc}")
                            else:
                                print(f"✗ Not supported: {w}x{h}")
                except Exception as e:
                    print(f"Error testing resolutions: {e}")
                finally:
                    if test_cap is not None:
                        test_cap.release()
                
                # Add current resolution if not already in the list
                current_res = (selected_camera['width'], selected_camera['height'])
                if not any(res[:2] == current_res for res in supported_resolutions):
                    supported_resolutions.append((current_res[0], current_res[1], 
                                               f"{current_res[0]}x{current_res[1]} (Current)"))
                
                # If we couldn't detect any supported resolutions, fall back to defaults
                if not supported_resolutions:
                    print("Couldn't detect supported resolutions, using defaults")
                    res_options = [
                        (640, 480, "640x480 (Default - Recommended for stability)"),
                        (selected_camera['width'], selected_camera['height'], f"{selected_camera['width']}x{selected_camera['height']} (Current)")
                    ]
                else:
                    res_options = supported_resolutions
            else:
                # For standard cameras
                res_options = [
                    (640, 480, "640x480 (Standard)"),
                    (800, 600, "800x600 (SVGA)"),
                    (1280, 720, "1280x720 (HD)"),
                    (1920, 1080, "1920x1080 (Full HD)"),
                    (selected_camera['width'], selected_camera['height'], f"{selected_camera['width']}x{selected_camera['height']} (Current)")
                ]
            
            # Remove duplicates from resolution options
            unique_res = []
            unique_values = set()
            for width, height, desc in res_options:
                if (width, height) not in unique_values:
                    unique_values.add((width, height))
                    unique_res.append((width, height, desc))
            
            print("\nAvailable resolutions:")
            for i, (width, height, desc) in enumerate(unique_res):
                print(f"{i+1}. {desc}")
                
            res_choice = -1
            while res_choice < 0 or res_choice >= len(unique_res):
                try:
                    res_choice = int(input(f"Select resolution (1-{len(unique_res)}): ")) - 1
                    if res_choice < 0 or res_choice >= len(unique_res):
                        print(f"Please choose a number from 1 to {len(unique_res)}")
                except ValueError:
                    print("Please enter a valid number")
            
            selected_camera['width'] = unique_res[res_choice][0]
            selected_camera['height'] = unique_res[res_choice][1]
            
            print(f"Selected resolution: {selected_camera['width']}x{selected_camera['height']}")
    except Exception as e:
        print(f"Error setting resolution: {e}")
        print("Using default resolution.")
    
    return selected_camera

def initialize_camera(camera_index, width, height):
    """
    Initialize the camera with the given index and resolution.
    Enhanced to better handle DroidCam and other virtual camera sources.
    """
    # For local cameras, try multiple backends
    print(f"Connecting to camera {camera_index}...")
    print("Trying multiple connection methods to ensure compatibility...")
    
    cap = None
    backends = [
        (cv2.CAP_DSHOW, "DirectShow (good for most webcams)"),
        (cv2.CAP_MSMF, "Media Foundation (better for some virtual cameras)"),
        (0, "Default (fallback)")
    ]
    
    for backend_id, backend_name in backends:
        print(f"Trying {backend_name}...")
        try:
            if backend_id == 0:  # Default backend
                test_cap = cv2.VideoCapture(camera_index)
            else:
                test_cap = cv2.VideoCapture(camera_index, backend_id)
            
            if test_cap.isOpened():
                # Try to read a frame to verify connection
                success = False
                for attempt in range(3):  # Multiple attempts for virtual cameras
                    ret, frame = test_cap.read()
                    if ret and frame is not None:
                        success = True
                        break
                    import time
                    time.sleep(0.5)  # Give virtual cameras more time to initialize
                
                if success or backend_id == 0:  # Accept default backend even without frame
                    print(f"Success with {backend_name}!")
                    cap = test_cap
                    break
                else:
                    test_cap.release()
        except Exception as e:
            print(f"Failed with {backend_name}: {str(e)}")
    
    # If no backend worked, try one more time with default
    if cap is None:
        print("All backends failed, trying one more time with default settings...")
        cap = cv2.VideoCapture(camera_index)

    # Final check if connection was successful
    if not cap.isOpened():
        raise ValueError(f"Error: Cannot open camera with source {camera_index}. If you're using DroidCam or another virtual camera, make sure it's properly set up and running.")
    
    # Configure camera settings
    # Note: We handle both normal and virtual cameras differently
    is_virtual_cam = False
    is_droidcam = False
    
    # Determine if this is a virtual camera like DroidCam
    device_name = "Unknown"
    try:
        # Check camera name for clues
        camera_name = "Unknown"
        if hasattr(cap, 'getBackendName'):
            backend_name = cap.getBackendName()
            camera_name = backend_name
        
        # Check if this is DroidCam or other virtual camera
        if "DroidCam" in camera_name:
            is_virtual_cam = True
            is_droidcam = True
            device_name = camera_name
        elif any(name.lower() in camera_name.lower() for name in ["Virtual", "OBS", "IP", "EpocCam", "Iriun"]):
            is_virtual_cam = True
            device_name = camera_name
    except Exception:
        pass
    
    # Use a robust approach for setting camera resolution, especially for DroidCam
    # The key is to try multiple approaches and have good fallbacks
    
    # Store original properties before changing
    try:
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = int(cap.get(cv2.CAP_PROP_FPS)) if int(cap.get(cv2.CAP_PROP_FPS)) > 0 else 30
    except Exception:
        original_width = 640
        original_height = 480
        original_fps = 30
    
    # Try to set resolution with appropriate approach based on camera type
    if is_droidcam:
        print(f"Detected DroidCam: {device_name}")
        print(f"Carefully applying requested resolution: {width}x{height}")
        
        # For DroidCam, we need a carefully staged approach
        import time
        
        # First try closing and reopening with new backend to avoid bugs
        camera_index_backup = camera_index
        cap.release()
        
        # Try different backends and approaches
        success = False
        for backend_id in [cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_MSMF]:
            try:
                if backend_id == cv2.CAP_ANY:
                    cap = cv2.VideoCapture(camera_index_backup)
                else:
                    cap = cv2.VideoCapture(camera_index_backup, backend_id)
                
                if cap.isOpened():
                    # DroidCam typically needs settings applied in this specific order
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Increase buffer for smoother capture
                    time.sleep(0.2)
                    cap.set(cv2.CAP_PROP_FPS, 30)  # Force 30 FPS for stability
                    time.sleep(0.2)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    time.sleep(0.2)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    time.sleep(0.5)  # Give it time to apply settings
                    
                    # Try to read a frame to verify settings worked
                    for _ in range(3):
                        try:
                            ret, frame = cap.read()
                            if ret and frame is not None and frame.size > 0:
                                success = True
                                break
                        except Exception:
                            pass
                        time.sleep(0.3)
                    
                    if success:
                        break
                    else:
                        cap.release()  # Try next backend
            except Exception as e:
                print(f"Error with backend {backend_id}: {e}")
        
        # If all backends failed, reopen with default
        if not cap.isOpened():
            print("Trying one more time with default settings...")
            cap = cv2.VideoCapture(camera_index_backup)
            # Use safe settings
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
        # Check what resolution we actually got
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if actual_width != width or actual_height != height:
            print(f"Note: DroidCam adjusted to resolution: {actual_width}x{actual_height}")
            print("This is the highest resolution DroidCam can provide.")
            
    # For other virtual cameras
    elif is_virtual_cam:
        print(f"Detected virtual camera: {device_name}")
        print("Using optimized settings for virtual cameras...")
        
        # Staged approach with verification
        resolutions_to_try = [
            (width, height),  # First try the requested resolution
            (1280, 720),      # Then try HD
            (800, 600),       # Then try intermediate
            (640, 480)        # Finally try the safest option
        ]
        
        for test_width, test_height in resolutions_to_try:
            # Try to set resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, test_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, test_height)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Give it time to apply settings
            import time
            time.sleep(0.3)
            
            # Try to read a frame to check if settings work
            try:
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    print(f"Successfully set resolution to {test_width}x{test_height}")
                    break
            except Exception:
                print(f"Failed with resolution {test_width}x{test_height}, trying next option...")
    
    # For regular physical cameras
    else:
        print(f"Setting up physical camera with resolution {width}x{height}")
        # Standard cameras are usually more reliable with resolution settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 60)  # Request 60 FPS
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce camera buffer
    
    # Get the actual properties the camera is using
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # If we couldn't get valid FPS (some cameras return 0), use a default value
    if actual_fps <= 0:
        actual_fps = 30
    
    # Test reading a frame with a robust error handling approach
    frame_read = False
    max_retries = 5  # More retries for virtual cameras
    
    # For DroidCam, we need special handling for high resolutions
    if is_droidcam and (actual_width > 1280 or actual_height > 720):
        print("High resolution detected with DroidCam. Using special handling...")
        
        # Special approach for high-resolution DroidCam - reopen with explicit settings
        import time
        backup_index = camera_index
        
        try:
            # Close current connection
            cap.release()
            time.sleep(0.5)
            
            # Try to open with DirectShow for high-res with DroidCam
            cap = cv2.VideoCapture(backup_index, cv2.CAP_DSHOW)
            
            # Set properties in this specific order which works better for DroidCam
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
            time.sleep(0.2)
            
            # First set the height, which sometimes works better with DroidCam
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, actual_height)
            time.sleep(0.2)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, actual_width)
            time.sleep(0.2)
            cap.set(cv2.CAP_PROP_FPS, 30)  # Force 30FPS for high-res
            time.sleep(0.5)  # Generous wait time for settings to apply
            
            # Try to read frames with retries
            for attempt in range(max_retries):
                try:
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
                        frame_read = True
                        break
                except Exception as e:
                    print(f"Retry attempt {attempt+1}/{max_retries}: {e}")
                time.sleep(0.5)
        
        except Exception as e:
            print(f"Error in special high-res handling: {e}")
            frame_read = False
    else:
        # Standard frame reading with retries for other cameras
        for attempt in range(max_retries):
            try:
                ret, frame = cap.read()
                if ret and frame is not None:
                    if frame.size > 0:  # Verify frame is not empty
                        frame_read = True
                        break
                    else:
                        print(f"Attempt {attempt+1}: Empty frame received")
                else:
                    print(f"Attempt {attempt+1}: Failed to read frame")
            except cv2.error as e:
                print(f"OpenCV error on attempt {attempt+1}: {e}")
            except Exception as e:
                print(f"General error on attempt {attempt+1}: {e}")
            
            import time
            time.sleep(0.5)  # Half second delay between attempts
    
    # If we still couldn't read a frame, try a last-resort fallback
    if not frame_read:
        print("\nWARNING: Could not read frame with the selected resolution.")
        print("Trying emergency fallback to 640x480...")
        
        # Release and reopen with lowest compatible settings
        camera_index_backup = camera_index
        cap.release()
        import time
        time.sleep(1.0)  # Give more time to reset
        
        # Try different backends with basic resolution
        for backend in [None, cv2.CAP_DSHOW, cv2.CAP_MSMF]:
            try:
                if backend is None:
                    cap = cv2.VideoCapture(camera_index_backup)
                else:
                    cap = cv2.VideoCapture(camera_index_backup, backend)
                
                if cap.isOpened():
                    # Set to most basic settings guaranteed to work
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    time.sleep(0.5)
                    
                    # Try reading
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        frame_read = True
                        # Update actual dimensions after fallback
                        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
                        if actual_fps <= 0:
                            actual_fps = 30
                            
                        print(f"Emergency fallback successful with resolution: {actual_width}x{actual_height}")
                        break
                    else:
                        cap.release()
            except Exception as e:
                print(f"Fallback error with backend {backend}: {e}")
                if cap.isOpened():
                    cap.release()
    
    # If absolutely nothing worked, provide a clear error message
    if not cap.isOpened():
        raise ValueError("Fatal error: Could not establish a working connection with any camera.")
    
    # If we got this far but still couldn't read a frame, show warning but continue
    if not frame_read:
        print("\nWARNING: Camera connected but could not read frames.")
        print("The application will continue, but may not function correctly.")
        print("Try restarting DroidCam or reconnecting your camera.")
    
    # Print final camera information
    print(f"\nCamera successfully initialized with resolution: {actual_width}x{actual_height} @ {actual_fps} FPS")
    
    if is_droidcam:
        print("Camera type: DroidCam virtual camera")
    elif is_virtual_cam:
        print("Camera type: Virtual camera")
    else:
        print(f"Camera type: Standard physical camera (index {camera_index})")
    
    return cap, actual_width, actual_height, actual_fps