# change the text here
clip_texts = [
    "person walks forward and stops walking", 
    "a person kicking a ball using their right foot",
    "a person playing violin",
    "a person waves goodbye with their right hand",
    "an injured person slowly hobbles forward",
    "a person sadly walks and sits down",
    "a person happily walks and sits down",
    "a person walks up to his table and slowly picks up a mug",
    "a person riding a skateboard",
    "a person jogs forward and stops, then jumps",
    "a person walks forward, then waves their left hand",
    "a person takes a free kick with his left foot, he then celebrates with his arms up in the air",
    "a person slowly walks backwards, he then sits down on a chair",
    "a person juggles three balls while moving side to side with his feet",
    "a person runs forwards and performs a front flip",
    "A person spins in place clockwise two full rotations, slows down, then carefully sits cross-legged on the ground",
    "A person raises their right arm and waves slowly side to side five times, lowers it gently",
    "A person hops on his right foot, he then hops on his left foot",
    "A person performs jumping jacks, he then jogs in place",
	"A person pretends to dive by bending at his waist and holds position briefly, he then mimics swimming strokes by moving arms in forward circles",
    "a person walks forward",
    "a person walks backward",
    "a person jumps",
    "a person does a handstand",
    "a person claps their hands"
    ]

import sys
import os
sys.argv = ['GPT_eval_multi.py']
import options.option_transformer as option_trans
args = option_trans.get_args_parser()

args.dataname = 't2m'
args.resume_pth = 'pretrained/VQVAE/net_last.pth'
args.resume_trans = 'pretrained/VQTransformer_corruption05/net_best_fid.pth'
args.down_t = 2
args.depth = 3
args.block_size = 51
from scipy.ndimage import gaussian_filter
import clip
import torch
import numpy as np
import models.vqvae as vqvae
import models.t2m_trans as trans
import warnings
warnings.filterwarnings('ignore')



def encode_text(text, clip_model, device, choose_layer=-1):
    # Motion Lens modification
    assert choose_layer >= -1 and choose_layer < 12, f"Expected layer -1 to 11, but got {choose_layer}"
    hidden_activations = []

    with torch.no_grad():
        text = clip.tokenize([text], truncate=True).to(device)
        x = clip_model.token_embedding(text).type(clip_model.dtype)  # [batch_size, n_ctx, d_model]


        x = x + clip_model.positional_embedding.type(clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        
        # original implementation
        # x = self.clip.transformer(x)
        # Motion Lens modification
        for i, layer in enumerate(clip_model.transformer.resblocks):
            print(' > processing clip layer ', i+1)
            x = layer(x)
            hidden_activations.append(x)
            print(x.shape)

        # Motion Lens modification
        x = hidden_activations[choose_layer]
        x = x.permute(1,0,2) # LND -> NLD
        x = clip_model.ln_final(x).type(clip_model.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ clip_model.text_projection
 
    return x.float()

## load clip model and datasets
clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False, download_root='./')  # Must set jit=False for training
clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                       args.nb_code,
                       args.code_dim,
                       args.output_emb_width,
                       args.down_t,
                       args.stride_t,
                       args.width,
                       args.depth,
                       args.dilation_growth_rate)


trans_encoder = trans.Text2Motion_Transformer(num_vq=args.nb_code,
                                embed_dim=1024,
                                clip_dim=args.clip_dim,
                                block_size=args.block_size,
                                num_layers=9,
                                n_head=16,
                                drop_out_rate=args.drop_out_rate,
                                fc_rate=args.ff_rate)


print ('loading checkpoint from {}'.format(args.resume_pth))
ckpt = torch.load(args.resume_pth, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.cuda()

print ('loading transformer checkpoint from {}'.format(args.resume_trans))
ckpt = torch.load(args.resume_trans, map_location='cpu')
trans_encoder.load_state_dict(ckpt['trans'], strict=True)
trans_encoder.eval()
trans_encoder.cuda()

mean = torch.from_numpy(np.load('./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy')).cuda()
std = torch.from_numpy(np.load('./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy')).cuda()

from utils.motion_process import recover_from_ric
import visualization.plot_3d_global as plot_3d

# Loop through each motion prompt
for i, clip_text in enumerate(clip_texts):
    print(f"Processing motion prompt: '{clip_text}'")
    
    # Create folder structure for results
    result_folder = f'results/example{i}'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    
    # Loop through each layer from -1 to 11
    for layer in range(-1, 12):
        layer_name = 'final_layer' if layer == -1 else f'layer{layer}'
        layer_folder = os.path.join(result_folder, layer_name)
        if not os.path.exists(layer_folder):
            os.makedirs(layer_folder)
        
        # Tokenize and encode the text prompt for the specified layer
        feat_clip_text = encode_text(clip_text, clip_model, torch.device('cuda'), layer)
        
        # Generate motion indices using the transformer encoder
        index_motion = trans_encoder.sample(feat_clip_text[0:1], False)
        
        # Decode the motion indices to get the predicted pose
        pred_pose = net.forward_decoder(index_motion)
        
        # Recover XYZ coordinates from predicted pose
        pred_xyz = recover_from_ric((pred_pose * std + mean).float(), 22)
        xyz = pred_xyz.reshape(1, -1, 22, 3)
        
        # Save the generated motion as a numpy file
        result_path = os.path.join(layer_folder, 'motion.npy')
        np.save(result_path, xyz.detach().cpu().numpy())
        
        # Visualize the motion and save as a gif
        gif_path = os.path.join(layer_folder, 'example.gif')
        pose_vis = plot_3d.draw_to_batch(xyz.detach().cpu().numpy(), [clip_text], [gif_path])
        
        print(f"Saved motion data to '{result_path}' and visualization to '{gif_path}'")

print("Processing complete.")