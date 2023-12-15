import torch
import torch.nn as nn
from languagebind import LanguageBindAudio, LanguageBindImage, LanguageBindVideo
from languagebind import LanguageBindAudioProcessor, LanguageBindImageProcessor, LanguageBindVideoProcessor
from languagebind import to_device

class MultiModalEncoder(nn.Module):

    def __init__(self, device,encoding_shape=1024,projected_shape=4096,torch_dtype=torch.float32):
        super(MultiModalEncoder, self).__init__()
        self.device = device
        self.dtype = torch_dtype
        
        # Load models
        self.audio_model = LanguageBindAudio.from_pretrained("LanguageBind/LanguageBind_Audio_FT",torch_dtype=torch.float32).to(device)
        self.video_model = LanguageBindVideo.from_pretrained("LanguageBind/LanguageBind_Video_FT",torch_dtype=torch.float32).to(device)
        self.image_model = LanguageBindImage.from_pretrained("LanguageBind/LanguageBind_Image",torch_dtype=torch.float32).to(device)
        
        # Load processors
        self.audio_processor = LanguageBindAudioProcessor(self.audio_model.config)
        self.video_processor = LanguageBindVideoProcessor(self.video_model.config)
        self.image_processor = LanguageBindImageProcessor(self.image_model.config)

        # Trainable modality-specific tokens // Commented out due to some bugs
        # shape = (1,1, projected_shape)
        # self.start_img = nn.Parameter(torch.rand(shape, dtype=torch_dtype)).to(device)
        # self.end_img = nn.Parameter(torch.rand(shape, dtype=torch_dtype)).to(device)
        # self.start_aud = nn.Parameter(torch.rand(shape, dtype=torch_dtype)).to(device)
        # self.end_aud = nn.Parameter(torch.rand(shape, dtype=torch_dtype)).to(device)
        # self.start_vid = nn.Parameter(torch.rand(shape, dtype=torch_dtype)).to(device)
        # self.end_vid = nn.Parameter(torch.rand(shape, dtype=torch_dtype)).to(device)

        # Projection layers
        # self.img_proj = nn.Linear(encoding_shape, projected_shape,dtype=torch_dtype).to(device)
        # self.video_proj = nn.Linear(encoding_shape, projected_shape,dtype=torch_dtype).to(device)
        # self.audio_proj = nn.Linear(encoding_shape, projected_shape,dtype=torch_dtype).to(device)

    def forward(self, inputs):
        # Initialize a list to keep our encoded representations
        encoding_list = []
        encoding_names = []

        pooled_outputs = {}

        
        # Encode Image
        if 'image' in inputs:
            image_processed = to_device(self.image_processor(inputs['image']), self.device)
            image_encodings = self.image_model.vision_model(**image_processed)
            pooled = image_encodings.pooler_output
            pooled = pooled / pooled.norm(p=2, dim=-1, keepdim=True)
            pooled_outputs['image_pooled'] = pooled
            # image_encodings = self.img_proj(image_encodings.last_hidden_state.to(self.dtype))
            image_encodings = torch.concat([image_encodings.last_hidden_state.to(self.dtype)]*4,-1)
            encoding_list.extend([ image_encodings])
            encoding_names.extend(['image_embedding'])

        # Encode Video
        if 'video' in inputs:
            video_processed = to_device(self.video_processor(inputs['video']), self.device)
            video_encodings = self.video_model.vision_model(**video_processed)
            pooled = video_encodings.pooler_output
            pooled = pooled / pooled.norm(p=2, dim=-1, keepdim=True)
            pooled_outputs['video_pooled'] = pooled
            # Average the last 8 frames
            video_encodings = video_encodings.last_hidden_state.reshape(-1,8,257,1024).mean(1)
            # video_encodings = self.video_proj(video_encodings.last_hidden_state.reshape(-1,8,257,1024).mean(1).to(self.dtype))
            video_encodings = torch.concat([video_encodings.to(self.dtype)]*4,-1)
            # print(video_encodings.shape)
            encoding_list.extend([video_encodings])
            encoding_names.extend(['video_embedding'])

          # Encode Audio
        if 'audio' in inputs:
            audio_processed = to_device(self.audio_processor(inputs['audio']), self.device)
            audio_encodings = self.audio_model.vision_model(**audio_processed)
            pooled = audio_encodings.pooler_output
            pooled = pooled / pooled.norm(p=2, dim=-1, keepdim=True)
            pooled_outputs['audio_pooled'] = pooled
            # audio_encodings = self.audio_proj(audio_encodings.last_hidden_state.to(self.dtype))
            audio_encodings = torch.concat([audio_encodings.last_hidden_state.to(self.dtype)]*4,-1)
            encoding_list.extend([ audio_encodings])
            encoding_names.extend(['audio_embedding'])

        # Concatenate all encodings
        if encoding_list:
            concatenated_encodings = torch.cat(encoding_list, dim=1)
        else:
            concatenated_encodings = None
        
        return {
            'concatenated_encodings': concatenated_encodings,
            **{encoding_name: encoding for encoding_name, encoding in zip(encoding_names, encoding_list)},
            **pooled_outputs

        }
    def freeze_encoders(self,img=True, aud=True, vid=True):
        # Freeze parameters for the audio model encoder
        for param in self.audio_model.parameters():
            param.requires_grad = not aud

        # Freeze parameters for the video model encoder
        for param in self.video_model.parameters():
            param.requires_grad = not vid

        # Freeze parameters for the image model encoder
        for param in self.image_model.parameters():
            param.requires_grad = not img