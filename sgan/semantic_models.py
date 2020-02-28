
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from various_length_models import *

'''
ProjectorBlock and LinearAttentionBlock blocks

Reference Code:
https://github.com/SaoYan/LearnToPayAttention/blob/master/blocks.py

'''

class ProjectorBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ProjectorBlock, self).__init__()
        self.op = nn.Conv2d(in_channels=in_features, out_channels=out_features, 
                            kernel_size=1, padding=0, bias=False)

    def forward(self, inputs):
        return self.op(inputs)

class LinearAttentionBlock(nn.Module):
    def __init__(self, in_features, normalize_attn=True):
        super(LinearAttentionBlock, self).__init__()
        self.normalize_attn = normalize_attn
        self.op = nn.Conv2d(in_channels=in_features, out_channels=1, 
                            kernel_size=1, padding=0, bias=False)

    def forward(self, l, g):
        N, C, W, H = l.size()
        c = self.op(l+g) # batch_sizex1xWxH
        if self.normalize_attn:
            a = F.softmax(c.view(N,1,-1), dim=2).view(N,1,W,H)
        else:
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l), l)
        if self.normalize_attn:
            g = g.view(N,C,-1).sum(dim=2) # batch_sizexC
        else:
            g = F.adaptive_avg_pool2d(g, (1,1)).view(N,C)
        return c.view(N,1,W,H), g


class FeatureExtractor(nn.Module):
    def __init__(self, img_size=512, img_embeded_dim=1024, attention=True, normalize_attn=True):
        """
        Select conv3_3, conv4_3, conv5_3 from vgg16 model, pretrained from ImageNet
        layer 14 receptive map [batch,256,128,128]
        layer 21 receptive map [batch,512,64,64]
        layer 28 receptive map [batch,512,32,32]

        Compare Global feature with local feature at layer 14, 21, 28.
        """
        super(feature_extractor,self).__init__()

        self.img_size = img_size
        self.image_embeded_dim = img_embeded_dim
        self.attention = attention
        self.normalize_attn = normalize_attn
        # selected conv layers
        self.select = ['14','21','28','30']
        self.model = torchvision.models.vgg16(pretrained=True).features
        self.dense = nn.Conv2d(in_channels=512, out_channels=512, 
                            kernel_size=int( self.img_size/32), padding=0, bias=True)
        # Projectors & Compatibility functions
        if self.attention:
            self.projector = ProjectorBlock(256, 512)
            self.attn1 = LinearAttentionBlock(in_features=512, normalize_attn=self.normalize_attn)
            self.attn2 = LinearAttentionBlock(in_features=512, normalize_attn=self.normalize_attn)
            self.attn3 = LinearAttentionBlock(in_features=512, normalize_attn=self.normalize_attn)
        # final aggregation layer
        if self.attention:
            self.aggregation = nn.Linear(in_features=512*3, out_features=self.image_embeded_dim, bias=True)
        else:
            self.aggregation = nn.Linear(in_features=512, out_features=self.image_embeded_dim, bias=True)

    def forward(self, x):
        """
        extract multiple local conv feature maps and multiply with logit weights
        """

        features = []
        logits = []
        # collect 3 intermidiate local feature maps and final global feature map
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        # global feature is from last max-pooling layer
        l1 = features[0]
        l2 = features[1]
        l3 = features[2]
        g =  self.dense(features[-1]) # batch_sizex512x1x1

         # pay attention
        if self.attention:
            c1, g1 = self.attn1(self.projector(l1), g)
            c2, g2 = self.attn2(l2, g)
            c3, g3 = self.attn3(l3, g)
            g = torch.cat((g1,g2,g3), dim=1) # batch_sizexC
            # aggregation layer
            x = self.aggregation(g) # batch_sizexnum_classes
        else:
            c1, c2, c3 = None, None, None
            x = self.aggregation(torch.squeeze(g))
        return x, c1, c2, c3


    def fine_tune(self, fine_tune=True):

        """
        define the fine tuning layers
        """
        for paras in self.model.parameters():
            paras.requires_grad = False
        # if fine tuning, only tuning later blocks, keep fundamental blocks unchanged
        for layer in list(self.model.children())[12:]:
            for paras in layer.parameters():
                paras.requires_grad = True



class SemanticAttentionGenerator(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, encoder_h_dim=64,
        decoder_h_dim=128, mlp_dim=1024, num_layers=1, noise_dim=(0, ),
        noise_type='gaussian', noise_mix_type='ped', pooling_type=None,
        pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
        activation='relu', batch_norm=True, neighborhood_size=2.0, grid_size=8, goal_dim=(2,), spatial_dim=True,
        img_size=512, img_embeded_dim=1024):
        super(SemanticAttentionGenerator, self).__init__()

        if pooling_type and pooling_type.lower() == 'none':
            pooling_type = None

        self.spatial_dim=spatial_dim
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.mlp_dim = mlp_dim
        self.encoder_h_dim = encoder_h_dim
        self.decoder_h_dim = decoder_h_dim
        self.embedding_dim = embedding_dim
        self.noise_dim = noise_dim
        self.num_layers = num_layers
        self.noise_type = noise_type
        self.noise_mix_type = noise_mix_type
        self.pooling_type = pooling_type
        self.noise_first_dim = 0
        self.pool_every_timestep = pool_every_timestep
        self.bottleneck_dim = 1024
        self.goal_dim = goal_dim
        self.img_size=img_size
        self.img_embeded_dim=img_embeded_dim # 1024
        
        # initialize feature extractor
        # image feature vector: (batch, self.img_embeded_dim)
        self.feature_extractor = FeatureExtractor(self.img_size, self.img_embeded_dim)
        self.feature_extractor.fine_tune(fine_tune=True)
        
        # After self trajectory embedding, encode embedded vector to hidden vector,
        # Encoder hidden vector: (self.num_layers, batch, self.h_dim)
        self.intention_encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=encoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        self.force_encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=encoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        self.intention_decoder = Decoder(
            pred_len,
            embedding_dim=embedding_dim,
            h_dim=decoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            pool_every_timestep=False,
            dropout=dropout,
            bottleneck_dim=bottleneck_dim,
            activation=activation,
        )
        
        self.force_decoder = Decoder(
            pred_len,
            embedding_dim=embedding_dim,
            h_dim=decoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            pool_every_timestep=pool_every_timestep,
            dropout=dropout,
            bottleneck_dim=bottleneck_dim,
            activation=activation,
            batch_norm=batch_norm,
            pooling_type=pooling_type,
            grid_size=grid_size,
            neighborhood_size=neighborhood_size,
            spatial_dim=2 if spatial_dim else decoder_h_dim
        )

        if pooling_type == 'pool_net':
            self.pool_net = PoolHiddenNet(
                embedding_dim=self.embedding_dim,
                h_dim=encoder_h_dim,
                mlp_dim=mlp_dim,
                bottleneck_dim=bottleneck_dim,
                activation=activation,
                batch_norm=batch_norm
            )
        elif pooling_type == 'spool':
            self.pool_net = SocialPooling(
                h_dim=encoder_h_dim,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout,
                neighborhood_size=neighborhood_size,
                grid_size=grid_size
            )

        
        if self.noise_dim[0] == 0:
            self.noise_dim = None
        else:
            self.noise_first_dim = noise_dim[0]

        # Decoder Hidden
        if pooling_type:
            input_dim = encoder_h_dim + bottleneck_dim + img_embeded_dim # need to add encoded image dims here
        else:
            input_dim = encoder_h_dim  + img_embeded_dim# need to add encoded image dims here
        
        if self.goal_dim[0] == 0:
            self.goal_dim = None
        else:
            self.goal_first_dim = goal_dim[0]

        if self.force_mlp_decoder_needed():
            mlp_decoder_context_dims = [
                input_dim, mlp_dim, decoder_h_dim - self.noise_first_dim
            ]


            self.force_mlp_decoder_context = make_mlp(
                mlp_decoder_context_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )

        if self.intention_mlp_decoder_needed():
            mlp_decoder_context_dims = [
                encoder_h_dim + img_embeded_dim, mlp_dim, decoder_h_dim - self.goal_first_dim # added img_embeded_dim iun first dimension
            ]

            self.intention_mlp_decoder_context = make_mlp(
                mlp_decoder_context_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )

        self.attention_mlp = nn.Linear(2*decoder_h_dim, 2)
        # nn.init.kaiming_normal_(self.attention_mlp.weight)


    def add_noise(self, _input, seq_start_end, user_noise=None, aux_input=None):
        """
        Inputs:
        - _input: Tensor of shape (_, decoder_h_dim - noise_first_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Outputs:
        - decoder_h: Tensor of shape (_, decoder_h_dim)
        """
        if not self.noise_dim:
            return _input

        if self.noise_mix_type == 'global':
            # Randomize one noise per "batch" (multiple trajectories)
            noise_shape = (seq_start_end.size(0), ) + self.noise_dim
        else:
            # Randomize one noise per "traj"
            noise_shape = (_input.size(0), ) + self.noise_dim

        if user_noise is not None:
            z_decoder = user_noise
        else:
            z_decoder = get_noise(noise_shape, self.noise_type, aux_input=aux_input)

        if self.noise_mix_type == 'global':
            _list = []
            for idx, (start, end) in enumerate(seq_start_end):
                start = start.item()
                end = end.item()
                _vec = z_decoder[idx].view(1, -1)
                _to_cat = _vec.repeat(end - start, 1)
                _list.append(torch.cat([_input[start:end], _to_cat], dim=1))
            decoder_h = torch.cat(_list, dim=0)
            return decoder_h

        decoder_h = torch.cat([_input, z_decoder], dim=1)

        return decoder_h

    def add_goal(self, _input, seq_start_end, goal_input=None):
        """
        Inputs:
        - _input: Tensor of shape (_, decoder_h_dim - noise_first_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Outputs:
        - decoder_h: Tensor of shape (_, decoder_h_dim)
        """
        if not self.goal_dim:
            return _input

        goal_shape = (_input.size(0), ) + self.goal_dim

        z_decoder = get_noise(goal_shape, 'inject_goal', aux_input=goal_input)

        decoder_h = torch.cat([_input, z_decoder], dim=1)

        return decoder_h

    def force_mlp_decoder_needed(self):
        if (
            self.noise_dim or self.pooling_type or
            self.encoder_h_dim != self.decoder_h_dim
        ):
            return True
        else:
            return False
    
    def intention_mlp_decoder_needed(self):
        if (
            self.goal_dim or self.encoder_h_dim != self.decoder_h_dim
        ):
            return True
        else:
            return False

    def forward(self, obs_traj, obs_traj_rel, seq_start_end, aux_input=None, user_noise=None, goal_input=None, seq_len=12, image):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        - obs_traj_rel: Tensor of shape (obs_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        - image: Tensor of shape (obs_len,batch,channel, H, W)
        Output:
        - pred_traj_rel: Tensor of shape (self.pred_len, batch, 2)
        """
        batch_size = obs_traj_rel.size(1)
        # Encode seq
        force_final_encoder_h = self.force_encoder(obs_traj_rel)
        intention_final_encoder_h = self.intention_encoder(obs_traj_rel)
        image_final_h = self.feature_extractor(image)

        # Pool States
        if self.pooling_type:
            end_pos = obs_traj[-1, :, :]
            pool_h = self.pool_net(force_final_encoder_h, seq_start_end, end_pos)
            # Construct input hidden states for decoder
            force_mlp_decoder_context_input = torch.cat(
                [force_final_encoder_h.view(-1, self.encoder_h_dim), pool_h], dim=1)
        else:
            force_mlp_decoder_context_input = force_final_encoder_h.view(
                -1, self.encoder_h_dim)

        intention_mlp_decoder_context_input = intention_final_encoder_h.view(-1, self.encoder_h_dim)

        # Add Noise
        if self.force_mlp_decoder_needed():
            noise_input = self.force_mlp_decoder_context(force_mlp_decoder_context_input)
        else:
            noise_input = force_mlp_decoder_context_input
        force_decoder_h = self.add_noise(
            noise_input, seq_start_end, aux_input=aux_input, user_noise=user_noise)
        force_decoder_h = torch.unsqueeze(force_decoder_h, 0)

        force_decoder_c = torch.zeros(
            self.num_layers, batch_size, self.decoder_h_dim
        ).cuda()

        if self.intention_mlp_decoder_needed():
            noise_input = self.intention_mlp_decoder_context(intention_mlp_decoder_context_input)
        else:
            noise_input = intention_mlp_decoder_context_input
        
        intention_decoder_h = self.add_goal(
                noise_input, seq_start_end, goal_input=goal_input)
        intention_decoder_h = torch.unsqueeze(intention_decoder_h, 0)

        intention_decoder_c = torch.zeros(
            self.num_layers, batch_size, self.decoder_h_dim
        ).cuda()

            

        force_state_tuple = (force_decoder_h, force_decoder_c)
        intention_state_tuple = (intention_decoder_h, intention_decoder_c)

        last_pos = obs_traj[-1]
        last_pos_rel = obs_traj_rel[-1]
        # Predict Trajectory

        ret = []
        attention = []
        intent = []
        social = []
        
        for t in range(seq_len):

            intention_rel_pos, intention_state_tuple = self.intention_decoder.step_forward(last_pos, last_pos_rel, intention_state_tuple, seq_start_end)

            intention_pos = intention_rel_pos + obs_traj[0]
            
            if self.spatial_dim:
                force_rel_pos, force_state_tuple = self.force_decoder.step_forward(intention_pos, intention_rel_pos, force_state_tuple, seq_start_end)
            else:
                force_rel_pos, force_state_tuple = self.force_decoder.step_forward(intention_pos, intention_state_tuple[0], force_state_tuple, seq_start_end)

            attention_score = self.attention_mlp(torch.cat([force_state_tuple[0].view(-1, self.decoder_h_dim), intention_state_tuple[0].view(-1, self.decoder_h_dim)], dim=1))

            attention_score = torch.nn.functional.softmax(attention_score, dim=1)

            last_pos_rel = force_rel_pos*attention_score[:, 0].view(-1,1) + intention_rel_pos*attention_score[:,1].view(-1,1)
            #ret.append(last_pos_rel)
            #print (attention_score[0,:])
            ret.append(last_pos_rel)
            attention.append(attention_score)
            intent.append(intention_rel_pos)
            social.append(force_rel_pos)

            last_pos = last_pos_rel+obs_traj[0]

#        force_decoder_out = self.force_decoder(
#            #last_pos,
#            #last_pos_rel,
#            torch.zeros(last_pos.size()).cuda(),   # Start with zero social force
#            torch.zeros(last_pos_rel.size()).cuda(),
#            force_state_tuple,
#            seq_start_end,
#            seq_len=seq_len
#        )
#
#        intention_decoder_out = self.intention_decoder(
#            last_pos,
#            last_pos_rel,
#            intention_state_tuple,
#            seq_start_end,
#            seq_len=seq_len
#        )

        
        return (torch.stack(ret), [torch.stack(attention), torch.stack(intent), torch.stack(social)])