import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming BasicConv and C2fModule/C3Module might be used here as well,
# or simpler conv blocks for the neck.
# For simplicity, let's define a light Conv block for the neck.
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, act=nn.SiLU()):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = act

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class FPNPANNeck(nn.Module):
    def __init__(self, p3_channels, p4_channels, p5_channels, neck_out_channels_list):
        """
        Implements FPN (Top-Down) and PAN (Bottom-Up) fusion.
        Paper: "YOLOv8 uses Feature Pyramid Network (FPN) and Path Aggregation Network (PAN)
                for multi-scale feature fusion."

        Args:
            p3_channels (int): Number of channels from backbone's P3 output (/8).
            p4_channels (int): Number of channels from backbone's P4 output (/16).
            p5_channels (int): Number of channels from backbone's P5 output (/32).
            neck_out_channels_list (list of int): List of output channels for N3, N4, N5
                                                  (the features fed to the detection head).
                                                  Usually, these are all the same, e.g., [256, 256, 256].
        """
        super().__init__()

        if len(neck_out_channels_list) != 3:
            raise ValueError("neck_out_channels_list must have 3 elements for P3, P4, P5 level outputs.")
        
        # Latent channels for FPN/PAN internal features, often same as output channels
        # For YOLOv8, these are often related to the backbone stage channels or a fixed size.
        # Let's use the first neck_out_channel as a common latent dim for simplicity.
        # Or, could be specific for each level.
        # For YOLOv8 style, often the channel counts in neck are consistent, e.g. 256 or 128 for 's'
        
        # For simplicity, let's assume the neck_out_channels are consistent.
        # e.g. if neck_out_channels_list = [128,128,128], then internal_c = 128.
        # This needs to match what the C2f/C3 blocks in the neck expect.
        # Ultralytics YOLOv8 neck is quite specific with channel numbers.
        
        # Let's make internal channels match the target output channels for simplicity for now
        # Or use a fixed intermediate channel size.
        # For YOLOv8s, neck channels might be like 128, 256, 512 (matching backbone outputs sometimes)
        # or all reduced to a common size like 128 or 256 for the C2f blocks in the neck.
        
        # For this skeleton, let's assume neck_out_channels_list[0] is the common channel dim in neck blocks.
        # This is a simplification.
        internal_c = neck_out_channels_list[0] # Example: 128 for a 's' model

        # --- FPN (Top-Down Path) ---
        # P5 goes into FPN top
        self.fpn_p5_conv = Conv(p5_channels, internal_c, k=1) # Reduce P5 channels
        # P4 is combined with upsampled P5
        self.fpn_p4_conv1 = Conv(p4_channels, internal_c, k=1) # Reduce P4 channels
        self.fpn_p4_csp = C2fModule(internal_c * 2, internal_c, n_bottlenecks=1) # Or C3, n_bottlenecks often related to depth_multiple
        # P3 is combined with upsampled (P4_out_fpn)
        self.fpn_p3_conv1 = Conv(p3_channels, internal_c, k=1) # Reduce P3 channels
        self.fpn_p3_csp = C2fModule(internal_c * 2, internal_c, n_bottlenecks=1)
        
        # FPN outputs (these are inputs to PAN or detection head for N3)
        self.fpn_out_p3_conv = Conv(internal_c, neck_out_channels_list[0], k=3, p=1) # N3 output

        # --- PAN (Bottom-Up Path) ---
        # N3 (from FPN) goes into PAN bottom
        self.pan_n3_downsample_conv = Conv(neck_out_channels_list[0], internal_c, k=3, s=2, p=1) # Downsample N3
        # Combine downsampled N3 with FPN_P4_out
        # FPN_P4_out is `self.fpn_p4_csp` output. Its channel count is `internal_c`.
        self.pan_n4_csp = C2fModule(internal_c * 2, internal_c, n_bottlenecks=1)
        self.pan_out_n4_conv = Conv(internal_c, neck_out_channels_list[1], k=3, p=1) # N4 output

        # N4 (from PAN) goes up
        self.pan_n4_downsample_conv = Conv(neck_out_channels_list[1], internal_c, k=3, s=2, p=1) # Downsample N4
        # Combine downsampled N4 with FPN_P5_out
        # FPN_P5_out is `self.fpn_p5_conv` output. Its channel count is `internal_c`.
        self.pan_n5_csp = C2fModule(internal_c * 2, internal_c, n_bottlenecks=1)
        self.pan_out_n5_conv = Conv(internal_c, neck_out_channels_list[2], k=3, p=1) # N5 output

    def forward(self, p3_feat, p4_feat, p5_feat):
        # p3_feat: /8 from backbone
        # p4_feat: /16 from backbone
        # p5_feat: /32 from backbone

        # --- FPN Path (Top-Down) ---
        # P5 processing
        fpn_p5_out = self.fpn_p5_conv(p5_feat) # Reduced channels

        # P4 processing
        fpn_p4_upsampled = F.interpolate(fpn_p5_out, size=p4_feat.shape[2:], mode='nearest')
        fpn_p4_latent = self.fpn_p4_conv1(p4_feat) # Reduce P4 channels
        fpn_p4_concat = torch.cat([fpn_p4_latent, fpn_p4_upsampled], dim=1)
        fpn_p4_out = self.fpn_p4_csp(fpn_p4_concat)

        # P3 processing
        fpn_p3_upsampled = F.interpolate(fpn_p4_out, size=p3_feat.shape[2:], mode='nearest')
        fpn_p3_latent = self.fpn_p3_conv1(p3_feat) # Reduce P3 channels
        fpn_p3_concat = torch.cat([fpn_p3_latent, fpn_p3_upsampled], dim=1)
        fpn_p3_out_csp = self.fpn_p3_csp(fpn_p3_concat) # Output for PAN path & N3
        
        # FPN N3 output (to detection head)
        neck_out_n3 = self.fpn_out_p3_conv(fpn_p3_out_csp)

        # --- PAN Path (Bottom-Up) ---
        # N4 processing
        pan_n3_down = self.pan_n3_downsample_conv(fpn_p3_out_csp) # Downsample FPN's P3 output
        pan_n4_concat = torch.cat([pan_n3_down, fpn_p4_out], dim=1) # Concat with FPN's P4 stage output
        pan_n4_out_csp = self.pan_n4_csp(pan_n4_concat)
        neck_out_n4 = self.pan_out_n4_conv(pan_n4_out_csp)

        # N5 processing
        pan_n4_down = self.pan_n4_downsample_conv(pan_n4_out_csp) # Downsample PAN's N4 output
        pan_n5_concat = torch.cat([pan_n4_down, fpn_p5_out], dim=1) # Concat with FPN's P5 stage output
        pan_n5_out_csp = self.pan_n5_csp(pan_n5_concat)
        neck_out_n5 = self.pan_out_n5_conv(pan_n5_out_csp)

        return neck_out_n3, neck_out_n4, neck_out_n5 # Outputs for detection heads (/8, /16, /32)

if __name__ == '__main__':
    # Example channels from a YOLOv8s-like backbone
    p3_c, p4_c, p5_c = 128, 256, 512 
    # Neck output channels, often consistent for heads
    neck_outs = [128, 256, 512] # Or could all be 128, or 256 etc.
                                 # This depends on the head design.
                                 # Ultralytics YOLOv8 heads take features of these varying channel sizes.

    neck = FPNPANNeck(p3_channels=p3_c, p4_channels=p4_c, p5_channels=p5_c,
                      neck_out_channels_list=neck_outs)
    print(neck)

    # Dummy inputs (batch_size, channels, H, W)
    dummy_p3 = torch.randn(1, p3_c, 80, 80)
    dummy_p4 = torch.randn(1, p4_c, 40, 40)
    dummy_p5 = torch.randn(1, p5_c, 20, 20)

    try:
        n3, n4, n5 = neck(dummy_p3, dummy_p4, dummy_p5)
        print("N3 output shape:", n3.shape) # Expected: [1, neck_outs[0], 80, 80]
        print("N4 output shape:", n4.shape) # Expected: [1, neck_outs[1], 40, 40]
        print("N5 output shape:", n5.shape) # Expected: [1, neck_outs[2], 20, 20]
    except Exception as e:
        print(f"Error during neck forward pass: {e}")
        print("This skeleton is conceptual and may need adjustments for real use.")