"""
FastMETRO model.
"""
import torch
import numpy as np
from torch import nn
from .transformer import build_transformer
from .position_encoding import build_position_encoding
import torch.nn.functional as F

class FastMETRO_Body_Network(nn.Module):
    """FastMETRO for 3D human pose and mesh reconstruction from a single RGB image"""
    def __init__(self, args, backbone, mesh_sampler, num_joints=14, num_vertices=431):
        """
        Parameters:
            - args: Arguments
            - backbone: CNN Backbone used to extract image features from the given image
            - mesh_sampler: Mesh Sampler used in the coarse-to-fine mesh upsampling
            - num_joints: The number of joint tokens used in the transformer decoder
            - num_vertices: The number of vertex tokens used in the transformer decoder
        """
        super().__init__()
        self.args = args
        self.backbone = backbone
        self.mesh_sampler = mesh_sampler
        self.num_joints = num_joints
        self.num_vertices = num_vertices

        
        
        # the number of transformer layers
        if 'FastMETRO_S' in args.model_name:
            num_enc_layers = 1
            num_dec_layers = 1
        elif 'FastMETRO_M' in args.model_name:
            num_enc_layers = 2
            num_dec_layers = 2
        elif 'FastMETRO_L' in args.model_name:
            num_enc_layers = 3
            num_dec_layers = 3
    

        self.keep_num = int(49 * args.keep_ratio)
        print(f"number of kept tokens: {self.keep_num} out of 49")
        self.token_selector_conv2d = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.token_selector_fc = nn.Linear(128, self.keep_num)
        self.layer_norm = nn.LayerNorm([self.keep_num, 512])

        # configurations for the first transformer
        self.transformer_config_1 = {"model_dim": args.model_dim_1, "dropout": args.transformer_dropout, "nhead": args.transformer_nhead, 
                                     "feedforward_dim": args.feedforward_dim_1, "num_enc_layers": num_enc_layers, "num_dec_layers": num_dec_layers, 
                                     "pos_type": args.pos_type}
        # configurations for the second transformer
        self.transformer_config_2 = {"model_dim": args.model_dim_2, "dropout": args.transformer_dropout, "nhead": args.transformer_nhead,
                                     "feedforward_dim": args.feedforward_dim_2, "num_enc_layers": num_enc_layers, "num_dec_layers": num_dec_layers, 
                                     "pos_type": args.pos_type}

        self.transformer_config_3 = {"model_dim": args.model_dim_3, "dropout": args.transformer_dropout, "nhead": args.transformer_nhead,
                                     "feedforward_dim": args.feedforward_dim_3, "num_enc_layers": num_enc_layers, "num_dec_layers": num_dec_layers, 
                                     "pos_type": args.pos_type}

        # build transformers
        self.transformer_1 = build_transformer(self.transformer_config_1)
        self.transformer_2 = build_transformer(self.transformer_config_2)
        self.transformer_3 = build_transformer(self.transformer_config_3)

        # dimensionality reduction
        self.dim_reduce_enc = nn.Linear(self.transformer_config_1["model_dim"], self.transformer_config_2["model_dim"])
        self.dim_reduce_dec = nn.Linear(self.transformer_config_1["model_dim"], self.transformer_config_2["model_dim"])
        # token embeddings
        self.cam_token_embed = nn.Embedding(1, int(self.transformer_config_1["model_dim"]))
        self.joint_token_embed = nn.Embedding((self.num_joints), int(self.transformer_config_1["model_dim"]))
        self.vertex_token_embed = nn.Embedding((self.num_vertices), int(self.transformer_config_3["model_dim"]))
        # positional encodings
        self.position_encoding_1 = build_position_encoding(pos_type=self.transformer_config_1['pos_type'], hidden_dim=self.transformer_config_1['model_dim'])
        self.position_encoding_2 = build_position_encoding(pos_type=self.transformer_config_2['pos_type'], hidden_dim=self.transformer_config_2['model_dim'])
        self.position_encoding_3 = build_position_encoding(pos_type=self.transformer_config_3['pos_type'], hidden_dim=self.transformer_config_3['model_dim'])
        # estimators
        self.xyz_regressor_j = nn.Linear(self.transformer_config_2["model_dim"], 3)
        self.xyz_regressor_v = nn.Linear(self.transformer_config_3["model_dim"], 3)
        
        # 1x1 Convolution
        if "efficientnet" in self.args.arch:
            args.conv_1x1_dim = 1280
        
        self.conv_1x1 = nn.Conv2d(args.conv_1x1_dim, self.transformer_config_1["model_dim"], kernel_size=1)

        # attention mask
        self.cam_predictor = nn.Linear(self.transformer_config_2["model_dim"], 3)

        adjacency_indices = torch.load('./tore/modeling/data/smpl_431_adjmat_indices.pt')
        adjacency_matrix_value = torch.load('./tore/modeling/data/smpl_431_adjmat_values.pt')
        adjacency_matrix_size = torch.load('./tore/modeling/data/smpl_431_adjmat_size.pt')
        adjacency_matrix = torch.sparse_coo_tensor(adjacency_indices, adjacency_matrix_value, size=adjacency_matrix_size).to_dense()
        temp_mask_2 = (adjacency_matrix == 0)
        self.attention_mask = temp_mask_2
    
    def forward(self, images):
        device = images.device
        batch_size = images.size(0)

        # preparation
        cam_token = self.cam_token_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)

        j_tokens = (self.joint_token_embed.weight).unsqueeze(1).repeat(1, batch_size, 1)
        v_tokens = (self.vertex_token_embed.weight).unsqueeze(1).repeat(1, batch_size, 1)
        j_attention_mask = None
        v_attention_mask = self.attention_mask.to(device)
        
        # extract image features through a CNN backbone
        if "efficientnet" in self.args.arch:
            img_features = self.backbone.extract_endpoints(images)['reduction_6']
        else:
            img_features = self.backbone(images)

        _, _, h, w = img_features.shape
        
        img_features = self.conv_1x1(img_features) # b * 512 * 7 * 7

        token_heatmap = self.token_selector_conv2d(img_features) # b * 512 * 7 * 7 -> b * 128 * 7 * 7
        token_heatmap = nn.GELU()(token_heatmap)
        token_heatmap = torch.flatten(token_heatmap, start_dim=2).permute(0, 2, 1) # b * 128 * 7 * 7 -> b * 49 * 128
        token_heatmap = self.token_selector_fc(token_heatmap) # b * 49 * 128 -> b * 49 * M
        token_heatmap = F.softmax(token_heatmap, dim=1).permute(0, 2, 1) # b * M * 49

        img_features = torch.flatten(img_features, start_dim=2).permute(0, 2, 1) # b * 512 * 7 * 7 -> b * 49 * 512
        img_features = self.layer_norm(torch.matmul(token_heatmap, img_features)) # (b * M * 49) . (b * 49 * 512) -> (b * M * 512)
        img_features = img_features.permute(1, 0, 2) # b * M * 512 -> M * b * 512
        
        
        # positional encodings
        pos_enc_1 = self.position_encoding_1(batch_size, self.keep_num, 1, device).flatten(2).permute(2, 0, 1)
        pos_enc_2 = self.position_encoding_2(batch_size, self.keep_num, 1, device).flatten(2).permute(2, 0, 1)
        pos_enc_3 = self.position_encoding_3(batch_size, self.num_joints, 1, device).flatten(2).permute(2, 0, 1)

        # first transformer encoder-decoder
        cam_features_1, enc_img_features_1, j_features_1 = self.transformer_1(img_features, cam_token, j_tokens, pos_enc_1, attention_mask=j_attention_mask)
        
        # progressive dimensionality reduction
        reduced_cam_features_1 = self.dim_reduce_enc(cam_features_1)
        reduced_enc_img_features_1 = self.dim_reduce_enc(enc_img_features_1)
        reduced_j_features_1 = self.dim_reduce_dec(j_features_1)

        # second transformer encoder-decoder
        cam_features_2, _, j_features_2 = self.transformer_2(reduced_enc_img_features_1, reduced_cam_features_1, reduced_j_features_1, pos_enc_2, attention_mask=j_attention_mask) 
        # estimators
        
        cam_parameter = self.cam_predictor(cam_features_2).view(batch_size, 3)
        
        _, _, v_features = self.transformer_3(j_features_2, cam_features_2, v_tokens, pos_enc_3, attention_mask=v_attention_mask) 

        pred_3d_coordinates_J = self.xyz_regressor_j(j_features_2.transpose(0, 1))
        pred_3d_joints = pred_3d_coordinates_J

        pred_3d_coordinates_V = self.xyz_regressor_v(v_features.transpose(0, 1))
        pred_3d_vertices_coarse = pred_3d_coordinates_V
        
        # coarse-to-fine mesh upsampling
        pred_3d_vertices_mid = self.mesh_sampler.upsample(pred_3d_vertices_coarse, n1=2, n2=1)
        pred_3d_vertices_fine = self.mesh_sampler.upsample(pred_3d_vertices_mid, n1=1, n2=0)

        return cam_parameter, pred_3d_joints, pred_3d_vertices_coarse, pred_3d_vertices_mid, pred_3d_vertices_fine, token_heatmap