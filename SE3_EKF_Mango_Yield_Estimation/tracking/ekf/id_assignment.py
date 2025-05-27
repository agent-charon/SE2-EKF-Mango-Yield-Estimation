import numpy as np
from scipy.optimize import linear_sum_assignment # For Hungarian algorithm
from .extended_kalman_filter import ExtendedKalmanFilter # To type hint Track object

def calculate_iou(box1_xyxy, box2_xyxy):
    """
    Calculates IoU between two bounding boxes [x1, y1, x2, y2].
    """
    x1_inter = max(box1_xyxy[0], box2_xyxy[0])
    y1_inter = max(box1_xyxy[1], box2_xyxy[1])
    x2_inter = min(box1_xyxy[2], box2_xyxy[2])
    y2_inter = min(box1_xyxy[3], box2_xyxy[3])

    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height

    box1_area = (box1_xyxy[2] - box1_xyxy[0]) * (box1_xyxy[3] - box1_xyxy[1])
    box2_area = (box2_xyxy[2] - box2_xyxy[0]) * (box2_xyxy[3] - box2_xyxy[1])
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area


class IDAssigner:
    def __init__(self, iou_threshold=0.3, max_age=30, min_hits_to_confirm=3, 
                 ekf_config_path="configs/tracking.yaml"):
        """
        Manages track IDs, matching detections to existing tracks.
        Paper Sec III-D.4: Paired Trackers, Unpaired Trackers, Unassigned Detection Boxes.

        Args:
            iou_threshold (float): Minimum IoU to match a detection to a track.
            max_age (int): Max frames a track can be 'lost' before being deleted.
            min_hits_to_confirm (int): Min consecutive hits to confirm a tentative track.
            ekf_config_path (str): Path to EKF config for new tracks.
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age # Corresponds to EKF.time_since_update
        self.min_hits_to_confirm = min_hits_to_confirm
        self.ekf_config_path = ekf_config_path

        self.tracks = [] # List of active ExtendedKalmanFilter objects
        self.next_track_id = 0

    def process_detections(self, detections_data):
        """
        Processes new detections and updates tracks.
        Args:
            detections_data (list of tuples): Each tuple is (bbox_xyxy, observed_theta, score, class_id).
                                              bbox_xyxy = [x1,y1,x2,y2]
        Returns:
            list of tuples: Active tracks with their current state: (track_id, bbox_xyxy, theta, status)
                            status can be 'confirmed', 'tentative'
        """
        # 1. Predict new locations for existing tracks
        for track in self.tracks:
            track.predict()

        # 2. Match detections to tracks
        # Detections: list of (bbox, theta, score, class_id)
        # Tracks: list of EKF objects
        
        matched_indices = [] # Tuples of (det_idx, track_idx)
        unmatched_detections_indices = list(range(len(detections_data)))
        unmatched_tracks_indices = list(range(len(self.tracks)))

        if detections_data and self.tracks:
            cost_matrix = np.full((len(detections_data), len(self.tracks)), np.inf)
            for d_idx, det_data in enumerate(detections_data):
                det_bbox = det_data[0] # bbox_xyxy
                for t_idx, track in enumerate(self.tracks):
                    # Compare det_bbox with track's predicted bbox
                    track_pred_bbox = track.get_predicted_bbox_xyxy()
                    iou = calculate_iou(det_bbox, track_pred_bbox)
                    if iou >= self.iou_threshold:
                        cost_matrix[d_idx, t_idx] = 1.0 - iou # Use 1-IoU as cost for Hungarian

            # Hungarian algorithm for optimal assignment
            # row_ind are detection indices, col_ind are track indices
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            current_matches = []
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] <= (1.0 - self.iou_threshold): # Valid match
                    matched_indices.append((r, c))
                    if r in unmatched_detections_indices: unmatched_detections_indices.remove(r)
                    if c in unmatched_tracks_indices: unmatched_tracks_indices.remove(c)
        
        # 3. Update Matched Tracks ("Paired Trackers")
        for det_idx, track_idx in matched_indices:
            detection_data = detections_data[det_idx]
            track = self.tracks[track_idx]
            
            # Convert detection bbox_xyxy and obs_theta to measurement format [Cx,Cy,W,H,Theta_obs]
            x1,y1,x2,y2 = detection_data[0]
            obs_theta = detection_data[1]
            cx = (x1+x2)/2; cy = (y1+y2)/2
            w = x2-x1; h = y2-y1
            measurement = np.array([cx,cy,w,h,obs_theta], dtype=np.float32)
            
            track.update(measurement)

        # 4. Handle Unmatched Tracks ("Unpaired Trackers" - possibly lost)
        tracks_to_remove = []
        for track_idx in unmatched_tracks_indices:
            track = self.tracks[track_idx]
            if track.time_since_update > self.max_age:
                tracks_to_remove.append(track)
        for track in tracks_to_remove:
            self.tracks.remove(track)

        # 5. Create New Tracks for Unmatched Detections ("Unassigned Detection Boxes" -> new tracks)
        for det_idx in unmatched_detections_indices:
            detection_data = detections_data[det_idx]
            bbox_xyxy = detection_data[0]
            obs_theta = detection_data[1]
            # score = detection_data[2] # Could use score to filter new tracks
            
            new_track = ExtendedKalmanFilter(track_id=self.next_track_id,
                                             initial_bbox_xyxy=bbox_xyxy,
                                             initial_theta=obs_theta,
                                             config_path=self.ekf_config_path)
            self.tracks.append(new_track)
            self.next_track_id += 1
            
        # 6. Return current state of active tracks
        active_track_states = []
        for track in self.tracks:
            # Only return confirmed tracks or those that meet some criteria
            status = 'confirmed' if track.hits >= self.min_hits_to_confirm else 'tentative'
            # if status == 'confirmed' or track.time_since_update < 1: # Show recently updated tentative
            current_bbox = track.get_current_state_bbox_xyxy()
            current_theta = track.x_hat[StateVector.THETA] # Get current estimated theta
            active_track_states.append({
                "id": track.track_id,
                "bbox_xyxy": current_bbox,
                "theta": current_theta,
                "age": track.age,
                "hits": track.hits,
                "time_since_update": track.time_since_update,
                "status": status
            })
        return active_track_states

    def reset(self):
        self.tracks = []
        self.next_track_id = 0

    # ... (rest of the IDAssigner class) ...

if __name__ == '__main__':
    # Create a dummy tracking.yaml if not present for testing
    dummy_config_content_yaml = """
ekf:
  delta_t: 1.0
  initial_covariance_diag:
    pos_tl_x: 10.0 # Corrected: colon, not semicolon
    pos_tl_y: 10.0
    pos_br_x: 10.0
    pos_br_y: 10.0
    vel_tl_x: 1.0
    vel_tl_y: 1.0
    vel_br_x: 1.0
    vel_br_y: 1.0
    angle_theta: 0.1
    angle_delta_theta: 0.01
  process_noise_diag:
    q_x1: 0.1
    q_y1: 0.1
    q_x2: 0.1
    q_y2: 0.1
    q_vx1: 0.01
    q_vy1: 0.01
    q_vx2: 0.01
    q_vy2: 0.01
    q_theta: 0.001
    q_delta_theta: 0.0001
  measurement_noise_diag:
    center_x_r: 1.0
    center_y_r: 1.0
    width_r: 1.0
    height_r: 1.0
    obs_theta_r: 0.01
  # Added id_assignment part for completeness, as IDAssigner itself uses these
  id_assignment:
    iou_threshold: 0.3 
    max_age: 5        
    min_hits_to_confirm: 1
"""
    test_config_path = "temp_test_tracking_config_id.yaml"
    # Ensure the directory for the temp config exists if it's nested
    os.makedirs(os.path.dirname(test_config_path), exist_ok=True) # Good practice
    with open(test_config_path, "w") as f:
        f.write(dummy_config_content_yaml) # No need to replace semicolons now

    assigner = IDAssigner(iou_threshold=0.3, max_age=5, min_hits_to_confirm=1, 
                          ekf_config_path=test_config_path)

    # ... (rest of the IDAssigner example usage) ...
    
    # Clean up dummy config
    import os
    if os.path.exists(test_config_path): os.remove(test_config_path)