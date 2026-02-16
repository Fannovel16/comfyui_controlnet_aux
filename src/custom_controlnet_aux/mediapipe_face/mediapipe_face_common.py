from typing import Mapping
import warnings
import numpy

try:
    import mediapipe as mp
    from mediapipe.framework.formats import landmark_pb2
except ImportError:
    warnings.warn(
        "The module 'mediapipe' is not installed. The package will have limited functionality. Please install it using the command: pip install 'mediapipe'"
    )
    mp = None

if mp:
    # Check MediaPipe version and API compatibility
    USE_NEW_API = False
    try:
        # Try to access new API (MediaPipe 0.10.32+)
        mp.tasks.vision.FaceLandmarker
        USE_NEW_API = True
    except AttributeError:
        USE_NEW_API = False

    if USE_NEW_API:
        # New API imports
        mp_drawing = mp.tasks.vision.drawing_utils
        mp_face_detection = mp.tasks.vision.FaceDetector
        mp_face_mesh = None  # Will be handled by wrapper
        mp_face_connections = mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
        mp_hand_connections = None
        mp_body_connections = None
        
        DrawingSpec = mp.tasks.vision.drawing_utils.DrawingSpec
        PoseLandmark = None
    else:
        # Old API imports
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_face_detection = mp.solutions.face_detection
        mp_face_mesh = mp.solutions.face_mesh
        mp_face_connections = mp.solutions.face_mesh_connections.FACEMESH_TESSELATION
        mp_hand_connections = mp.solutions.hands_connections.HAND_CONNECTIONS
        mp_body_connections = mp.solutions.pose_connections.POSE_CONNECTIONS

        DrawingSpec = mp.solutions.drawing_styles.DrawingSpec
        PoseLandmark = mp.solutions.drawing_styles.PoseLandmark

    min_face_size_pixels: int = 64
    f_thick = 2
    f_rad = 1
    right_iris_draw = DrawingSpec(color=(10, 200, 250), thickness=f_thick, circle_radius=f_rad)
    right_eye_draw = DrawingSpec(color=(10, 200, 180), thickness=f_thick, circle_radius=f_rad)
    right_eyebrow_draw = DrawingSpec(color=(10, 220, 180), thickness=f_thick, circle_radius=f_rad)
    left_iris_draw = DrawingSpec(color=(250, 200, 10), thickness=f_thick, circle_radius=f_rad)
    left_eye_draw = DrawingSpec(color=(180, 200, 10), thickness=f_thick, circle_radius=f_rad)
    left_eyebrow_draw = DrawingSpec(color=(180, 220, 10), thickness=f_thick, circle_radius=f_rad)
    mouth_draw = DrawingSpec(color=(10, 180, 10), thickness=f_thick, circle_radius=f_rad)
    head_draw = DrawingSpec(color=(10, 200, 10), thickness=f_thick, circle_radius=f_rad)

    # Face connection specifications
    face_connection_spec = {}
    
    if USE_NEW_API:
        # New API face mesh connections - use tuples instead of Connection objects for hashability
        for edge in mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL:
            face_connection_spec[(edge.start, edge.end)] = head_draw
        for edge in mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE:
            face_connection_spec[(edge.start, edge.end)] = left_eye_draw
        for edge in mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW:
            face_connection_spec[(edge.start, edge.end)] = left_eyebrow_draw
        for edge in mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE:
            face_connection_spec[(edge.start, edge.end)] = right_eye_draw
        for edge in mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW:
            face_connection_spec[(edge.start, edge.end)] = right_eyebrow_draw
        for edge in mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS:
            face_connection_spec[(edge.start, edge.end)] = mouth_draw
        iris_landmark_spec = {468: right_iris_draw, 473: left_iris_draw}
    else:
        # Old API face mesh connections
        for edge in mp_face_mesh.FACEMESH_FACE_OVAL:
            face_connection_spec[edge] = head_draw
        for edge in mp_face_mesh.FACEMESH_LEFT_EYE:
            face_connection_spec[edge] = left_eye_draw
        for edge in mp_face_mesh.FACEMESH_LEFT_EYEBROW:
            face_connection_spec[edge] = left_eyebrow_draw
        for edge in mp_face_mesh.FACEMESH_RIGHT_EYE:
            face_connection_spec[edge] = right_eye_draw
        for edge in mp_face_mesh.FACEMESH_RIGHT_EYEBROW:
            face_connection_spec[edge] = right_eyebrow_draw
        for edge in mp_face_mesh.FACEMESH_LIPS:
            face_connection_spec[edge] = mouth_draw
        iris_landmark_spec = {468: right_iris_draw, 473: left_iris_draw}


class FaceMeshWrapper:
    """Wrapper for MediaPipe FaceMesh that provides backward compatibility."""
    
    def __init__(self, static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        
        if USE_NEW_API:
            self._init_new_api()
        else:
            self._init_old_api()
    
    def _init_new_api(self):
        """Initialize the new MediaPipe API."""
        try:
            # Create face landmarker options
            base_options = mp.tasks.BaseOptions(model_asset_path="face_landmarker.task")
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=self.max_num_faces,
                min_face_detection_confidence=self.min_detection_confidence
            )
            self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        except Exception as e:
            warnings.warn(f"Could not initialize new MediaPipe API: {e}. Falling back to old API.")
            self._init_old_api()
    
    def _init_old_api(self):
        """Initialize the old MediaPipe API."""
        self.facemesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=self.static_image_mode,
            max_num_faces=self.max_num_faces,
            refine_landmarks=self.refine_landmarks,
            min_detection_confidence=self.min_detection_confidence,
        )
    
    def process(self, image):
        """Process an image and return face landmarks."""
        if USE_NEW_API and hasattr(self, 'landmarker'):
            return self._process_new_api(image)
        else:
            return self._process_old_api(image)
    
    def _process_new_api(self, image):
        """Process image using new MediaPipe API."""
        try:
            # Convert numpy array to MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            
            # Process the image
            detection_result = self.landmarker.detect(mp_image)
            
            # Convert result to old API format
            result = MediaPipeResultWrapper(detection_result)
            return result
            
        except Exception as e:
            warnings.warn(f"Error in new API processing: {e}. Falling back to old API.")
            return self._process_old_api(image)
    
    def _process_old_api(self, image):
        """Process image using old MediaPipe API."""
        return self.facemesh.process(image)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if USE_NEW_API and hasattr(self, 'landmarker'):
            self.landmarker.close()
        elif hasattr(self, 'facemesh'):
            self.facemesh.close()


class MediaPipeResultWrapper:
    """Wrapper for MediaPipe detection results to maintain compatibility."""
    
    def __init__(self, detection_result):
        self.detection_result = detection_result
        self.multi_face_landmarks = []
        
        if USE_NEW_API and detection_result.face_landmarks:
            for face_landmarks in detection_result.face_landmarks:
                # Convert to old API format
                landmark_list = landmark_pb2.NormalizedLandmarkList()
                for landmark in face_landmarks:
                    normalized_landmark = landmark_pb2.NormalizedLandmark(
                        x=landmark.x,
                        y=landmark.y,
                        z=landmark.z,
                        visibility=getattr(landmark, 'visibility', 1.0),
                        presence=getattr(landmark, 'presence', 1.0)
                    )
                    landmark_list.landmark.append(normalized_landmark)
                self.multi_face_landmarks.append(landmark_list)
        elif not USE_NEW_API:
            # For old API, just pass through
            self.multi_face_landmarks = detection_result.multi_face_landmarks


def draw_pupils(image, landmark_list, drawing_spec, halfwidth: int = 2):
    """We have a custom function to draw the pupils because the mp.draw_landmarks method requires a parameter for all
    landmarks.  Until our PR is merged into mediapipe, we need this separate method."""
    if len(image.shape) != 3:
        raise ValueError("Input image must be H,W,C.")
    image_rows, image_cols, image_channels = image.shape
    if image_channels != 3:  # BGR channels
        raise ValueError('Input image must contain three channel bgr data.')
    for idx, landmark in enumerate(landmark_list.landmark):
        if (
                (landmark.HasField('visibility') and landmark.visibility < 0.9) or
                (landmark.HasField('presence') and landmark.presence < 0.5)
        ):
            continue
        if landmark.x >= 1.0 or landmark.x < 0 or landmark.y >= 1.0 or landmark.y < 0:
            continue
        image_x = int(image_cols*landmark.x)
        image_y = int(image_rows*landmark.y)
        draw_color = None
        if isinstance(drawing_spec, Mapping):
            if drawing_spec.get(idx) is None:
                continue
            else:
                draw_color = drawing_spec[idx].color
        elif isinstance(drawing_spec, DrawingSpec):
            draw_color = drawing_spec.color
        image[image_y-halfwidth:image_y+halfwidth, image_x-halfwidth:image_x+halfwidth, :] = draw_color


def reverse_channels(image):
    """Given a numpy array in RGB form, convert to BGR.  Will also convert from BGR to RGB."""
    # im[:,:,::-1] is a neat hack to convert BGR to RGB by reversing the indexing order.
    # im[:,:,::[2,1,0]] would also work but makes a copy of the data.
    return image[:, :, ::-1]


def generate_annotation(
        img_rgb,
        max_faces: int,
        min_confidence: float
):
    """
    Find up to 'max_faces' inside the provided input image.
    If min_face_size_pixels is provided and nonzero it will be used to filter faces that occupy less than this many
    pixels in the image.
    """
    try:
        with FaceMeshWrapper(
                static_image_mode=True,
                max_num_faces=max_faces,
                refine_landmarks=True,
                min_detection_confidence=min_confidence,
        ) as facemesh:
            img_height, img_width, img_channels = img_rgb.shape
            assert(img_channels == 3)

            results = facemesh.process(img_rgb)

            if results is None or not results.multi_face_landmarks:
                print("No faces detected in controlnet image for Mediapipe face annotator.")
                return numpy.zeros_like(img_rgb)

            # Filter faces that are too small
            filtered_landmarks = []
            for lm in results.multi_face_landmarks:
                landmarks = lm.landmark
                face_rect = [
                    landmarks[0].x,
                    landmarks[0].y,
                    landmarks[0].x,
                    landmarks[0].y,
                ]  # Left, up, right, down.
                for i in range(len(landmarks)):
                    face_rect[0] = min(face_rect[0], landmarks[i].x)
                    face_rect[1] = min(face_rect[1], landmarks[i].y)
                    face_rect[2] = max(face_rect[2], landmarks[i].x)
                    face_rect[3] = max(face_rect[3], landmarks[i].y)
                if min_face_size_pixels > 0:
                    face_width = abs(face_rect[2] - face_rect[0])
                    face_height = abs(face_rect[3] - face_rect[1])
                    face_width_pixels = face_width * img_width
                    face_height_pixels = face_height * img_height
                    face_size = min(face_width_pixels, face_height_pixels)
                    if face_size >= min_face_size_pixels:
                        filtered_landmarks.append(lm)
                else:
                    filtered_landmarks.append(lm)

            # Annotations are drawn in BGR for some reason, but we don't need to flip a zero-filled image at the start.
            empty = numpy.zeros_like(img_rgb)

            # Draw detected faces:
            for face_landmarks in filtered_landmarks:
                mp_drawing.draw_landmarks(
                    empty,
                    face_landmarks,
                    connections=face_connection_spec.keys(),
                    landmark_drawing_spec=None,
                    connection_drawing_spec=face_connection_spec
                )
                draw_pupils(empty, face_landmarks, iris_landmark_spec, 2)

            # Flip BGR back to RGB.
            empty = reverse_channels(empty).copy()

            return empty
            
    except Exception as e:
        warnings.warn(f"Error in generate_annotation: {e}")
        return numpy.zeros_like(img_rgb)