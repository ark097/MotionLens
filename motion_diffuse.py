# Replace the encode_text method of MotionTransformer in MotionDiffuse/text2Motion/models/transformer.py
# with the encode text block provided here

class MotionLensTransformer:
    def encode_text(self, text, device, choose_layer=-1):
        # Motion Lens modification
        assert choose_layer >= -1 and choose_layer < 12, f"Expected layer -1 to 11, but got {choose_layer}"
        hidden_activations = []


        with torch.no_grad():
            text = clip.tokenize(text, truncate=True).to(device)
            x = self.clip.token_embedding(text).type(self.clip.dtype)  # [batch_size, n_ctx, d_model]

            x = x + self.clip.positional_embedding.type(self.clip.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            
            # original implementation
            # x = self.clip.transformer(x)
            # Motion Lens modification
            for i, layer in enumerate(self.clip.transformer.resblocks):
                print('processing clip layer ', i+1)
                x = layer(x)
                hidden_activations.append(x)


            # Motion Lens modification
            x = hidden_activations[choose_layer]
            x = self.clip.ln_final(x).type(self.clip.dtype)
            print(x.shape)



# Update the DDPMTrainer's generate methods in MotionDiffuse/text2Motion/trainers/ddpm_trainer.py
class DDPMTrainer:
    def generate_batch(self, caption, m_lens, dim_pose, choose_layer=-1):
        xf_proj, xf_out = self.encoder.encode_text(caption, self.device, choose_layer=choose_layer)
        
        B = len(caption)
        T = min(m_lens.max(), self.encoder.num_frames)
        output = self.diffusion.p_sample_loop(
            self.encoder,
            (B, T, dim_pose),
            clip_denoised=False,
            progress=True,
            model_kwargs={
                'xf_proj': xf_proj,
                'xf_out': xf_out,
                'length': m_lens
            })
        return output

    def generate(self, caption, m_lens, dim_pose, batch_size=1024, choose_layer=-1):
        N = len(caption)
        cur_idx = 0
        self.encoder.eval()
        all_output = []
        while cur_idx < N:
            if cur_idx + batch_size >= N:
                batch_caption = caption[cur_idx:]
                batch_m_lens = m_lens[cur_idx:]
            else:
                batch_caption = caption[cur_idx: cur_idx + batch_size]
                batch_m_lens = m_lens[cur_idx: cur_idx + batch_size]
            output = self.generate_batch(batch_caption, batch_m_lens, dim_pose, choose_layer=choose_layer)
            B = output.shape[0]

            for i in range(B):
                all_output.append(output[i])
            cur_idx += batch_size
        return all_output

