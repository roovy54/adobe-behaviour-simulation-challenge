import torch
import torch.nn as nn
from languagebind import LanguageBindAudio, LanguageBindImage, LanguageBindVideo
from languagebind import LanguageBindAudioProcessor, LanguageBindImageProcessor, LanguageBindVideoProcessor, LanguageBindImageTokenizer
from languagebind import to_device

class MultiModalEncoder(nn.Module):

    def __init__(self, device,torch_dtype=torch.float32):
        super(MultiModalEncoder, self).__init__()
        self.device = device
        self.dtype = torch_dtype
        
        # Load models
        self.audio_model = LanguageBindAudio.from_pretrained("LanguageBind/LanguageBind_Audio_FT",torch_dtype=torch.float32).to(device)
        self.video_model = LanguageBindVideo.from_pretrained("LanguageBind/LanguageBind_Video_FT",torch_dtype=torch.float32).to(device)
        self.image_model = LanguageBindImage.from_pretrained("LanguageBind/LanguageBind_Image",torch_dtype=torch.float32).to(device)
        self.text_model = self.image_model.text_model
        
        # Load processors
        self.audio_processor = LanguageBindAudioProcessor(self.audio_model.config)
        self.video_processor = LanguageBindVideoProcessor(self.video_model.config)
        self.image_processor = LanguageBindImageProcessor(self.image_model.config)
        self.tokenizer = LanguageBindImageTokenizer.from_pretrained("LanguageBind/LanguageBind_Image")


    def forward(self, inputs):
        # Initialize a list to keep our encoded representations
        returns = {}

        
        # Encode Image
        if 'image' in inputs:
            image_processed = to_device(self.image_processor(inputs['image']), self.device)
            image_encodings = self.image_model.vision_model(**image_processed)
            pooled = image_encodings.pooler_output
            pooled = self.image_model.visual_projection(pooled)
            image_encodings.pooler_output = pooled / pooled.norm(p=2, dim=-1, keepdim=True)
            returns['image'] = image_encodings
            

        # Encode Video
        if 'video' in inputs:
            video_processed = to_device(self.video_processor(inputs['video']), self.device)
            video_encodings = self.video_model.vision_model(**video_processed)
            pooled = video_encodings.pooler_output
            pooled = self.video_model.visual_projection(pooled)
            video_encodings.pooler_output = pooled / pooled.norm(p=2, dim=-1, keepdim=True)
            
            # Average the last 8 frames
            video_encodings.last_hidden_state = video_encodings.last_hidden_state.reshape(-1,8,257,1024).mean(1)
            returns['video'] = video_encodings

          # Encode Audio
        if 'audio' in inputs:
            audio_processed = to_device(self.audio_processor(inputs['audio']), self.device)
            audio_encodings = self.audio_model.vision_model(**audio_processed)
            pooled = audio_encodings.pooler_output
            pooled = self.audio_model.visual_projection(pooled)
            pooled = pooled / pooled.norm(p=2, dim=-1, keepdim=True)
            audio_encodings.pooler_output = pooled
            returns['audio'] = audio_encodings
        
        if 'text' in inputs:
            text_processed = to_device(self.tokenizer(inputs['text'], max_length=77, padding='max_length',
                                             truncation=True, return_tensors='pt'), self.device)
            text_encodings = self.text_model(**text_processed)
            pooled = text_encodings.pooler_output
            text_encodings.pooler_output = pooled / pooled.norm(p=2, dim=-1, keepdim=True)
            returns['text'] = text_encodings
        
        return returns
    
    def freeze_encoders(self,freeze_img=True, freeze_aud=True, freeze_vid=True):
        # Freeze parameters for the audio model encoder
        for param in self.audio_model.parameters():
            param.requires_grad = not freeze_aud

        # Freeze parameters for the video model encoder
        for param in self.video_model.parameters():
            param.requires_grad = not freeze_vid

        # Freeze parameters for the image model encoder
        for param in self.image_model.parameters():
            param.requires_grad = not freeze_img