# """
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# """

# from __future__ import absolute_import, division, print_function, unicode_literals

# import logging
# import code
# import torch
# from torch import nn
# from .modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder, BertPooler
# from .modeling_bert import BertLayerNorm as LayerNormClass
# import metro.modeling.data.config as cfg

# class METRO_Encoder(BertPreTrainedModel):
#     def __init__(self, config):
#         super(METRO_Encoder, self).__init__(config)
#         self.config = config
#         self.embeddings = BertEmbeddings(config)
#         self.encoder = BertEncoder(config)
#         self.pooler = BertPooler(config)
#         self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
#         self.img_dim = config.img_feature_dim 

#         try:
#             self.use_img_layernorm = config.use_img_layernorm
#         except:
#             self.use_img_layernorm = None

#         self.img_embedding = nn.Linear(self.img_dim, self.config.hidden_size, bias=True)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         if self.use_img_layernorm:
#             self.LayerNorm = LayerNormClass(config.hidden_size, eps=config.img_layer_norm_eps)

#         self.apply(self.init_weights)


#     def _prune_heads(self, heads_to_prune):
#         """ Prunes heads of the model.
#             heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
#             See base class PreTrainedModel
#         """
#         for layer, heads in heads_to_prune.items():
#             self.encoder.layer[layer].attention.prune_heads(heads)

#     def forward(self, img_feats, input_ids=None, token_type_ids=None, attention_mask=None,
#             position_ids=None, head_mask=None):

#         batch_size = len(img_feats)
#         seq_length = len(img_feats[0])
#         input_ids = torch.zeros([batch_size, seq_length],dtype=torch.long).cuda()

#         if position_ids is None:
#             position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
#             position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

#         position_embeddings = self.position_embeddings(position_ids)

#         if attention_mask is None:
#             attention_mask = torch.ones_like(input_ids)

#         if token_type_ids is None:
#             token_type_ids = torch.zeros_like(input_ids)

#         if attention_mask.dim() == 2:
#             extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
#         elif attention_mask.dim() == 3:
#             extended_attention_mask = attention_mask.unsqueeze(1)
#         else:
#             raise NotImplementedError

#         extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
#         extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

#         if head_mask is not None:
#             if head_mask.dim() == 1:
#                 head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
#                 head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
#             elif head_mask.dim() == 2:
#                 head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
#             head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
#         else:
#             head_mask = [None] * self.config.num_hidden_layers

#         # Project input token features to have spcified hidden size
#         img_embedding_output = self.img_embedding(img_feats)

#         # We empirically observe that adding an additional learnable position embedding leads to more stable training
#         embeddings = position_embeddings + img_embedding_output

#         if self.use_img_layernorm:
#             embeddings = self.LayerNorm(embeddings)
#         embeddings = self.dropout(embeddings)

#         encoder_outputs = self.encoder(embeddings,
#                 extended_attention_mask, head_mask=head_mask)
#         sequence_output = encoder_outputs[0]

#         outputs = (sequence_output,)
#         if self.config.output_hidden_states:
#             all_hidden_states = encoder_outputs[1]
#             outputs = outputs + (all_hidden_states,)
#         if self.config.output_attentions:
#             all_attentions = encoder_outputs[-1]
#             outputs = outputs + (all_attentions,)

#         return outputs

# class METRO(BertPreTrainedModel):
#     '''
#     The archtecture of a transformer encoder block we used in METRO
#     '''
#     def __init__(self, config):
#         super(METRO, self).__init__(config)
#         self.config = config
#         self.bert = METRO_Encoder(config)
#         self.cls_head = nn.Linear(config.hidden_size, self.config.output_feature_dim)
#         self.residual = nn.Linear(config.img_feature_dim, self.config.output_feature_dim)
#         self.apply(self.init_weights)

#     def forward(self, img_feats, input_ids=None, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
#             next_sentence_label=None, position_ids=None, head_mask=None):
#         '''
#         # self.bert has three outputs
#         # predictions[0]: output tokens
#         # predictions[1]: all_hidden_states, if enable "self.config.output_hidden_states"
#         # predictions[2]: attentions, if enable "self.config.output_attentions"
#         '''
#         predictions = self.bert(img_feats=img_feats, input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
#                             attention_mask=attention_mask, head_mask=head_mask)

#         # We use "self.cls_head" to perform dimensionality reduction. We don't use it for classification.
#         pred_score = self.cls_head(predictions[0])
#         res_img_feats = self.residual(img_feats)
#         pred_score = pred_score + res_img_feats

#         if self.config.output_attentions and self.config.output_hidden_states:
#             return pred_score, predictions[1], predictions[-1]
#         else:
#             return pred_score

# class METRO_Hand_Network(torch.nn.Module):
#     '''
#     End-to-end METRO network for hand pose and mesh reconstruction from a single image.
#     '''
#     def __init__(self, args, config, backbone, trans_encoder):
#         super(METRO_Hand_Network, self).__init__()
#         self.config = config
#         self.backbone = backbone
#         self.trans_encoder = trans_encoder
#         self.upsampling = torch.nn.Linear(195, 778)
#         self.cam_param_fc = torch.nn.Linear(3, 1)
#         self.cam_param_fc2 = torch.nn.Linear(195+21, 150) 
#         self.cam_param_fc3 = torch.nn.Linear(150, 3)

#     def forward(self, images, mesh_model, mesh_sampler, meta_masks=None, is_train=False):
#         batch_size = images.size(0)
#         # Generate T-pose template mesh
#         template_pose = torch.zeros((1,48))
#         template_pose = template_pose.cuda()
#         template_betas = torch.zeros((1,10)).cuda()
#         template_vertices, template_3d_joints = mesh_model.layer(template_pose, template_betas)
#         template_vertices = template_vertices/1000.0
#         template_3d_joints = template_3d_joints/1000.0

#         template_vertices_sub = mesh_sampler.downsample(template_vertices)

#         # normalize
#         template_root = template_3d_joints[:,cfg.J_NAME.index('Wrist'),:]
#         template_3d_joints = template_3d_joints - template_root[:, None, :]
#         template_vertices = template_vertices - template_root[:, None, :]
#         template_vertices_sub = template_vertices_sub - template_root[:, None, :]
#         num_joints = template_3d_joints.shape[1]

#         # concatinate template joints and template vertices, and then duplicate to batch size
#         ref_vertices = torch.cat([template_3d_joints, template_vertices_sub],dim=1)
#         ref_vertices = ref_vertices.expand(batch_size, -1, -1)

#         # extract global image feature using a CNN backbone
#         image_feat = self.backbone(images)

#         # concatinate image feat and template mesh
#         image_feat = image_feat.view(batch_size, 1, 2048).expand(-1, ref_vertices.shape[-2], -1)
#         features = torch.cat([ref_vertices, image_feat], dim=2)

#         if is_train==True:
#             # apply mask vertex/joint modeling
#             # meta_masks is a tensor of all the masks, randomly generated in dataloader
#             # we pre-define a [MASK] token, which is a floating-value vector with 0.01s
#             constant_tensor = torch.ones_like(features).cuda()*0.01
#             features = features*meta_masks + constant_tensor*(1-meta_masks)     

#         # forward pass
#         if self.config.output_attentions==True:
#             features, hidden_states, att = self.trans_encoder(features)
#         else:
#             features = self.trans_encoder(features)

#         pred_3d_joints = features[:,:num_joints,:]
#         pred_vertices_sub = features[:,num_joints:,:]

#         # learn camera parameters
#         x = self.cam_param_fc(features)
#         x = x.transpose(1,2)
#         x = self.cam_param_fc2(x)
#         x = self.cam_param_fc3(x)
#         cam_param = x.transpose(1,2)
#         cam_param = cam_param.squeeze()

#         temp_transpose = pred_vertices_sub.transpose(1,2)
#         pred_vertices = self.upsampling(temp_transpose)
#         pred_vertices = pred_vertices.transpose(1,2)

#         if self.config.output_attentions==True:
#             return cam_param, pred_3d_joints, pred_vertices_sub, pred_vertices, hidden_states, att
#         else:
#             return cam_param, pred_3d_joints, pred_vertices_sub, pred_vertices

# class METRO_Body_Network(torch.nn.Module):
#     '''
#     End-to-end METRO network for human pose and mesh reconstruction from a single image.
#     '''
#     def __init__(self, args, config, backbone, trans_encoder, mesh_sampler):
#         super(METRO_Body_Network, self).__init__()
#         self.config = config
#         self.config.device = args.device
#         self.backbone = backbone
#         self.trans_encoder = trans_encoder
#         self.upsampling = torch.nn.Linear(431, 1723)
#         self.upsampling2 = torch.nn.Linear(1723, 6890)
#         self.conv_learn_tokens = torch.nn.Conv1d(49,431+14,1)
#         self.cam_param_fc = torch.nn.Linear(3, 1)
#         self.cam_param_fc2 = torch.nn.Linear(431, 250)
#         self.cam_param_fc3 = torch.nn.Linear(250, 3)

#     def forward(self, images, smpl, mesh_sampler, meta_masks=None, is_train=False):
#         batch_size = images.size(0)
#         # Generate T-pose template mesh
#         template_pose = torch.zeros((1,72))
#         template_pose[:,0] = 3.1416 # Rectify "upside down" reference mesh in global coord
#         template_pose = template_pose.cuda(self.config.device)
#         template_betas = torch.zeros((1,10)).cuda(self.config.device)
#         template_vertices = smpl(template_pose, template_betas)

#         # template mesh simplification
#         template_vertices_sub = mesh_sampler.downsample(template_vertices)
#         template_vertices_sub2 = mesh_sampler.downsample(template_vertices_sub, n1=1, n2=2)

#         # template mesh-to-joint regression 
#         template_3d_joints = smpl.get_h36m_joints(template_vertices)
#         template_pelvis = template_3d_joints[:,cfg.H36M_J17_NAME.index('Pelvis'),:]
#         template_3d_joints = template_3d_joints[:,cfg.H36M_J17_TO_J14,:]
#         num_joints = template_3d_joints.shape[1]

#         # normalize
#         template_3d_joints = template_3d_joints - template_pelvis[:, None, :]
#         template_vertices_sub2 = template_vertices_sub2 - template_pelvis[:, None, :]

#         # concatinate template joints and template vertices, and then duplicate to batch size
#         ref_vertices = torch.cat([template_3d_joints, template_vertices_sub2],dim=1)
#         ref_vertices = ref_vertices.expand(batch_size, -1, -1)

#         # extract image feature maps using a CNN backbone
#         image_feat = self.backbone(images)
#         image_feat_newview = image_feat.view(batch_size,2048,-1)
#         image_feat_newview = image_feat_newview.transpose(1,2)
#         # and apply a conv layer to learn image token for each 3d joint/vertex position
#         img_tokens = self.conv_learn_tokens(image_feat_newview)

#         # concatinate image feat and template mesh
#         features = torch.cat([ref_vertices, img_tokens], dim=2)

#         if is_train==True:
#             # apply mask vertex/joint modeling
#             # meta_masks is a tensor of all the masks, randomly generated in dataloader
#             # we pre-define a [MASK] token, which is a floating-value vector with 0.01s
#             constant_tensor = torch.ones_like(features).cuda(self.config.device)*0.01
#             features = features*meta_masks + constant_tensor*(1-meta_masks)            

#         # forward pass
#         if self.config.output_attentions==True:
#             features, hidden_states, att = self.trans_encoder(features)
#         else:
#             features = self.trans_encoder(features)

#         pred_3d_joints = features[:,:num_joints,:]
#         pred_vertices_sub2 = features[:,num_joints:,:]

#         # learn camera parameters
#         x = self.cam_param_fc(pred_vertices_sub2)
#         x = x.transpose(1,2)
#         x = self.cam_param_fc2(x)
#         x = self.cam_param_fc3(x)
#         cam_param = x.transpose(1,2)
#         cam_param = cam_param.squeeze()

#         temp_transpose = pred_vertices_sub2.transpose(1,2)
#         pred_vertices_sub = self.upsampling(temp_transpose)
#         pred_vertices_full = self.upsampling2(pred_vertices_sub)
#         pred_vertices_sub = pred_vertices_sub.transpose(1,2)
#         pred_vertices_full = pred_vertices_full.transpose(1,2)

#         if self.config.output_attentions==True:
#             return cam_param, pred_3d_joints, pred_vertices_sub2, pred_vertices_sub, pred_vertices_full, hidden_states, att
#         else:
#             return cam_param, pred_3d_joints, pred_vertices_sub2, pred_vertices_sub, pred_vertices_full 


# ----------------------------------------------------------------------------------------------
# FastMETRO Official Code
# Copyright (c) POSTECH Algorithmic Machine Intelligence Lab. (P-AMI Lab.) All Rights Reserved 
# Licensed under the MIT license.
# ----------------------------------------------------------------------------------------------

"""
FastMETRO model.
"""
import torch
import numpy as np
from torch import nn
from .transformer import build_transformer
from .position_encoding import build_position_encoding
import torch.nn.functional as F


# ----------------------------------------------------------------------------------------------
# FastMETRO Official Code
# Copyright (c) POSTECH Algorithmic Machine Intelligence Lab. (P-AMI Lab.) All Rights Reserved 
# Licensed under the MIT license.
# ----------------------------------------------------------------------------------------------

"""
FastMETRO model.
"""
import torch
import numpy as np
from torch import nn
from .transformer import build_transformer
from .position_encoding import build_position_encoding

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
                                     "feedforward_dim": args.feedforward_dim_3, "num_enc_layers": 1, "num_dec_layers": num_dec_layers, 
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
        self.conv_1x1 = nn.Conv2d(args.conv_1x1_dim, self.transformer_config_1["model_dim"], kernel_size=1)

        # attention mask
        self.cam_predictor = nn.Linear(self.transformer_config_2["model_dim"], 3)

        adjacency_indices = torch.load('./metro/modeling/data/smpl_431_adjmat_indices.pt')
        adjacency_matrix_value = torch.load('./metro/modeling/data/smpl_431_adjmat_values.pt')
        adjacency_matrix_size = torch.load('./metro/modeling/data/smpl_431_adjmat_size.pt')
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
        img_features = self.backbone(images) # CNN


        # token = self.token_learner(img_features) # B * 1 * 7 * 7
        # MSE loss for this.
        # img_features[:,keep_index,:]


        # 0-1 score.



        # input(22)
        # HRnet
        # torch.Size([B, 2048, 7, 7])


        # 49 / 4
        # torch.Size([25, 2048])
        

        # Resnet-50
        # torch.Size([25, 2048, 7, 7])


        # input("stop")
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
        # print(j_features_2.shape) # 14, B, 128
        # print(cam_features_2.shape) # 1, B, 128
        # print(v_tokens.shape) # 431, B, 64
        # print(pos_enc_3.shape) # 14, B, 64
        # print(v_attention_mask.shape) # 431, 431
        
        _, _, v_features = self.transformer_3(j_features_2, cam_features_2, v_tokens, pos_enc_3, attention_mask=v_attention_mask) 
        # print(v_features.shape)
        # input("22")
        pred_3d_coordinates_J = self.xyz_regressor_j(j_features_2.transpose(0, 1))
        pred_3d_joints = pred_3d_coordinates_J

        pred_3d_coordinates_V = self.xyz_regressor_v(v_features.transpose(0, 1))
        pred_3d_vertices_coarse = pred_3d_coordinates_V
        
        # coarse-to-fine mesh upsampling
        pred_3d_vertices_mid = self.mesh_sampler.upsample(pred_3d_vertices_coarse, n1=2, n2=1)
        pred_3d_vertices_fine = self.mesh_sampler.upsample(pred_3d_vertices_mid, n1=1, n2=0)

        return cam_parameter, pred_3d_joints, pred_3d_vertices_coarse, pred_3d_vertices_mid, pred_3d_vertices_fine