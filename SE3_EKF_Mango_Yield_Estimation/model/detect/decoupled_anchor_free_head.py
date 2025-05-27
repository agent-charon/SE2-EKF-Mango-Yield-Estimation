import torch
import torch.nn as nn
import math

# Assuming Conv is defined as in FPNPANNeck or BasicConv from backbone
class Conv(nn.Module): # Simplified Conv from FPNPANNeck
    def __init__(self, c1, c2, k=1, s=1, p=0, act=nn.SiLU()):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = act if act is not None else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DecoupledAnchorFreeHead(nn.Module):
    def __init__(self, num_classes, in_channels_list, strides=[8, 16, 32], 
                 reg_max=16, num_head_convs=2, head_conv_channels=None):
        """
        Decoupled Anchor-Free Detection Head (similar to YOLOv8).
        Paper: "YOLOv8 adopts a decoupled head, eliminating the objectness branch and
                focusing on classification and regression. The Anchor-Free approach,
                which identifies object centers and estimates bounding box dimensions..."

        Args:
            num_classes (int): Number of detection classes.
            in_channels_list (list of int): Input channels from FPN/PAN for P3, P4, P5.
            strides (list of int): Strides for P3, P4, P5 feature maps.
            reg_max (int): Used in DFL (Distribution Focal Loss) for box regression.
                           Predicts (reg_max + 1) values for each side of the box.
            num_head_convs (int): Number of conv layers in cls and reg branches.
            head_conv_channels (int or None): Channels for head convs. If None, uses input channels.
        """
        super().__init__()
        self.num_classes = num_classes
        self.in_channels_list = in_channels_list
        self.strides = strides
        self.reg_max = reg_max # For DFL, predicts distribution for each t,b,l,r
        self.num_outputs_reg = 4 * (self.reg_max + 1) # For DFL: tlbr * (distribution bins)
                                                      # If not DFL, reg is just 4 (e.g. tlbr offsets or cxcywh)
                                                      # YOLOv8 head outputs box (4 * reg_max) and cls (num_classes)
                                                      # The "4 * reg_max" is because it predicts a distribution for each of l,t,r,b.
                                                      # Each distribution has `reg_max` bins (actually `reg_max + 1` values, but often referred to as `reg_max` in config)
                                                      # Total is 4 * (reg_max).
        self.num_outputs_cls = num_classes

        self.cls_branches = nn.ModuleList()
        self.reg_branches = nn.ModuleList()

        for i, in_c in enumerate(in_channels_list):
            current_head_conv_c = head_conv_channels if head_conv_channels else in_c
            
            # Classification branch
            cls_convs = []
            for _ in range(num_head_convs):
                cls_convs.append(Conv(in_c if not cls_convs else current_head_conv_c, 
                                      current_head_conv_c, k=3, p=1))
            cls_convs.append(nn.Conv2d(current_head_conv_c, self.num_outputs_cls, 1)) # Final cls pred
            self.cls_branches.append(nn.Sequential(*cls_convs))

            # Regression branch
            reg_convs = []
            for _ in range(num_head_convs):
                reg_convs.append(Conv(in_c if not reg_convs else current_head_conv_c,
                                      current_head_conv_c, k=3, p=1))
            reg_convs.append(nn.Conv2d(current_head_conv_c, self.num_outputs_reg, 1)) # Final reg pred
            self.reg_branches.append(nn.Sequential(*reg_convs))
        
        self.initialize_biases()


    def initialize_biases(self):
        # Initialize biases for classification layers for stability, similar to RetinaNet/FCOS
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None and m.out_channels == self.num_classes: # Assuming last conv in cls_branches
                    # Prior for foreground ( ব্যস্ত )
                    prior_prob = 0.01
                    bias_value = -math.log((1 - prior_prob) / prior_prob)
                    torch.nn.init.constant_(m.bias, bias_value)


    def forward(self, features_list):
        # features_list: [P3_out_neck, P4_out_neck, P5_out_neck]
        cls_outputs = []
        reg_outputs = []

        for i, features in enumerate(features_list):
            cls_pred = self.cls_branches[i](features) # (B, num_classes, H, W)
            reg_pred = self.reg_branches[i](features) # (B, 4 * reg_max, H, W)
            
            cls_outputs.append(cls_pred)
            reg_outputs.append(reg_pred)
            
        # During inference, these raw outputs need to be decoded:
        # 1. Reshape: (B, C, H, W) -> (B, H*W, C)
        # 2. For reg_pred with DFL: (B, H*W, 4, reg_max) -> apply softmax to get distribution, then expectation for box coords.
        # 3. Combine with anchor points (implicit in anchor-free based on grid cell location + stride).
        # 4. Apply sigmoid to cls_outputs for probabilities.
        # This forward only returns raw conv outputs. Decoding is usually part of post-processing or loss computation.

        return cls_outputs, reg_outputs # List of tensors for each FPN level


    def decode_outputs(self, cls_preds_levels, reg_preds_levels, image_shape, conf_thresh=0.25, nms_thresh=0.45):
        """
        Decodes raw head outputs into bounding boxes and scores.
        This is a simplified decoder. A full YOLOv8 decoder is more involved.
        Args:
            cls_preds_levels: List of [B, num_classes, H, W] tensors.
            reg_preds_levels: List of [B, 4*reg_max, H, W] tensors.
            image_shape: (original_height, original_width) of the input image for scaling boxes.
            conf_thresh: Confidence threshold for filtering.
            nms_thresh: NMS threshold.
        Returns:
            List of detections for each image in the batch.
            Each detection: [x1, y1, x2, y2, score, class_id]
        """
        batch_size = cls_preds_levels[0].shape[0]
        all_detections_batch = [[] for _ in range(batch_size)]
        
        # Precompute anchor points (centers of grid cells) for each level
        anchor_points_levels, stride_tensor_levels = self._make_anchor_points(cls_preds_levels)

        for i in range(len(self.strides)): # For each FPN level
            cls_pred = cls_preds_levels[i] # (B, num_classes, H, W)
            reg_pred = reg_preds_levels[i] # (B, 4*reg_max, H, W)
            anchor_points = anchor_points_levels[i] # (H*W, 2)
            stride_tensor = stride_tensor_levels[i] # (H*W, 1)

            # Reshape and permute
            # Class predictions
            cls_pred = cls_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes) # (B, H*W, num_classes)
            cls_scores = cls_pred.sigmoid()

            # Regression predictions (DFL decoding)
            # reg_pred is (B, 4*reg_max, H, W)
            reg_pred = reg_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4, self.reg_max) # (B, H*W, 4, reg_max)
            
            # DFL: apply softmax and compute expectation
            # Project tensor for DFL
            project = torch.arange(self.reg_max, device=reg_pred.device, dtype=torch.float32)
            
            box_dist = reg_pred.softmax(dim=3) # Softmax over reg_max dimension
            # box_dist shape: (B, H*W, 4, reg_max)
            # project shape: (reg_max)
            # We need to multiply and sum: (B, H*W, 4, reg_max) * (1,1,1,reg_max) -> sum over last dim
            # Result should be (B, H*W, 4) representing distances (dist_left, dist_top, dist_right, dist_bottom)
            # Corrected DFL projection:
            # box_reg_decoded = (box_dist * project.view(1,1,1,-1)).sum(dim=3) # (B, H*W, 4)
            # Simpler interpretation for skeleton (common in some anchor-free):
            # Assume reg_pred after conv is already the delta_ltrb values, not DFL logits.
            # For actual DFL, the above is needed.
            # For this skeleton, let's assume reg_pred is direct (t,b,l,r) distances if not using DFL:
            # If not using DFL, reg_pred would be (B, 4, H, W)
            # For DFL, the reg_pred shape (B, 4*reg_max, H,W) is correct.
            
            # Let's implement the DFL expectation for box coordinates
            # project needs to be (1, 1, 1, reg_max) for broadcasting
            box_reg_dist_scaled = (box_dist * project.view(1, 1, 1, self.reg_max)).sum(dim=3) # (B, H*W, 4)
            
            # Convert distances to (x1, y1, x2, y2)
            # box_reg_dist_scaled is (dist_left, dist_top, dist_right, dist_bottom) * stride
            # anchor_points is (center_x, center_y)
            # x1 = anchor_x - dist_left
            # y1 = anchor_y - dist_top
            # x2 = anchor_x + dist_right
            # y2 = anchor_y + dist_bottom
            
            # anchor_points shape: (H*W, 2) -> expand for batch
            # box_reg_dist_scaled shape: (B, H*W, 4)
            # Stride tensor needed for scaling distances
            
            # Expand anchor_points and stride_tensor for batch
            anchor_points_batch = anchor_points.unsqueeze(0).repeat(batch_size, 1, 1) # (B, H*W, 2)
            stride_tensor_batch = stride_tensor.unsqueeze(0).repeat(batch_size, 1, 1) # (B, H*W, 1)
            
            # Decoded box distances (ltrb) * stride
            box_ltrb_scaled = box_reg_dist_scaled * stride_tensor_batch # (B, H*W, 4)

            # Calculate x1, y1, x2, y2
            x1y1 = anchor_points_batch - box_ltrb_scaled[..., :2] # Subtract left, top
            x2y2 = anchor_points_batch + box_ltrb_scaled[..., 2:] # Add right, bottom
            decoded_boxes = torch.cat((x1y1, x2y2), dim=2) # (B, H*W, 4) in x1,y1,x2,y2 format

            # Filter based on confidence and class
            for b_idx in range(batch_size):
                boxes_b = decoded_boxes[b_idx] # (H*W, 4)
                scores_b = cls_scores[b_idx]   # (H*W, num_classes)

                # Iterate through classes (excluding background if any)
                for class_id in range(self.num_classes):
                    class_scores = scores_b[:, class_id]
                    confident_mask = class_scores >= conf_thresh
                    
                    if confident_mask.any():
                        selected_boxes = boxes_b[confident_mask]
                        selected_scores = class_scores[confident_mask]
                        
                        # Store [x1, y1, x2, y2, score, class_id]
                        for box_idx in range(selected_boxes.shape[0]):
                            det = [
                                selected_boxes[box_idx, 0].item(), selected_boxes[box_idx, 1].item(),
                                selected_boxes[box_idx, 2].item(), selected_boxes[box_idx, 3].item(),
                                selected_scores[box_idx].item(),
                                float(class_id)
                            ]
                            all_detections_batch[b_idx].append(det)
        
        # Apply NMS per image
        final_batch_results = []
        for b_idx in range(batch_size):
            detections = torch.tensor(all_detections_batch[b_idx], device=cls_preds_levels[0].device)
            if detections.shape[0] == 0:
                final_batch_results.append(torch.empty((0,6), device=detections.device))
                continue

            # torchvision.ops.nms requires boxes (N,4) and scores (N)
            # NMS is typically done per-class or multi-class NMS.
            # For simplicity here, let's do NMS on all boxes if scores are high enough.
            # A more correct NMS is class-aware.
            
            # Using torchvision.ops.batched_nms for class-aware NMS:
            nms_boxes = detections[:, :4]
            nms_scores = detections[:, 4]
            nms_class_ids = detections[:, 5] # NMS is usually per class
            
            if nms_boxes.numel() > 0:
                # For simplicity in this skeleton, let's assume a single class and apply NMS
                # For multi-class, loop through classes or use batched_nms
                keep_indices = torchvision.ops.nms(nms_boxes, nms_scores, nms_thresh)
                final_detections = detections[keep_indices]
                final_batch_results.append(final_detections)
            else:
                 final_batch_results.append(torch.empty((0,6), device=detections.device))


        return final_batch_results # List of tensors, each [N_det, 6]

    def _make_anchor_points(self, features_list):
        """ Helper to create anchor points for each FPN level """
        anchor_points_all_levels = []
        stride_tensor_all_levels = []
        for i, feat_map in enumerate(features_list):
            _, _, h, w = feat_map.shape
            stride = self.strides[i]
            
            # Create grid
            shift_y, shift_x = torch.meshgrid(torch.arange(h, device=feat_map.device), 
                                              torch.arange(w, device=feat_map.device), indexing='ij')
            # Grid cells centers
            # Anchor points are (x_center, y_center) for each grid cell
            anchor_points_x = (shift_x + 0.5) * stride
            anchor_points_y = (shift_y + 0.5) * stride
            
            anchor_points = torch.stack((anchor_points_x, anchor_points_y), dim=-1).reshape(-1, 2) # (H*W, 2)
            stride_tensor = torch.full((h * w, 1), stride, device=feat_map.device, dtype=torch.float32)

            anchor_points_all_levels.append(anchor_points)
            stride_tensor_all_levels.append(stride_tensor)
        return anchor_points_all_levels, stride_tensor_all_levels


if __name__ == '__main__':
    num_mango_classes = 1 # Just 'mango'
    # Input channels from a YOLOv8s-like Neck
    # neck_outs = [128, 256, 512] from FPNPANNeck example
    in_channels = [128, 256, 512] 
    strides = [8, 16, 32]
    reg_max_val = 16 # For DFL

    head = DecoupledAnchorFreeHead(num_classes=num_mango_classes,
                                   in_channels_list=in_channels,
                                   strides=strides,
                                   reg_max=reg_max_val,
                                   num_head_convs=2) # Typical 2 convs in head branches
    print(head)

    # Dummy inputs from Neck (batch_size, channels, H, W)
    dummy_n3 = torch.randn(1, in_channels[0], 80, 80) # Stride 8
    dummy_n4 = torch.randn(1, in_channels[1], 40, 40) # Stride 16
    dummy_n5 = torch.randn(1, in_channels[2], 20, 20) # Stride 32
    neck_features = [dummy_n3, dummy_n4, dummy_n5]

    try:
        cls_preds, reg_preds = head(neck_features)
        print("\nRaw outputs per level:")
        for i in range(len(cls_preds)):
            print(f"Level {i} (Stride {strides[i]}):")
            print(f"  Cls pred shape: {cls_preds[i].shape}") # Expected (B, num_classes, H_i, W_i)
            print(f"  Reg pred shape: {reg_preds[i].shape}") # Expected (B, 4*reg_max_val, H_i, W_i)
        
        # Example of how decoding might be called (needs torchvision for NMS in this skeleton)
        # import torchvision
        # print("\nAttempting to decode outputs (requires torchvision for NMS):")
        # detections = head.decode_outputs(cls_preds, reg_preds, image_shape=(640,640))
        # if detections and detections[0] is not None:
        #     print(f"Decoded detections for image 0: {detections[0].shape}")
        # else:
        #     print("No detections after decoding or NMS requires torchvision.")

    except ImportError:
        print("torchvision not found, skipping NMS part of decode_outputs example.")
    except Exception as e:
        print(f"Error during head forward pass or decode: {e}")
        print("This skeleton is conceptual and may need adjustments for real use.")
