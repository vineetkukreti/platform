import torch
from transformers import AutoTokenizer, AutoModel

class EmbeddingService:
    cos = torch.nn.CosineSimilarity(dim=0)

    def __init__(self, modelId, device='cpu'):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(modelId, use_fast=True)
        self.model = AutoModel.from_pretrained(modelId).to(self.device)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def embeddings(self, sentences):
        encoded_input = self.tokenizer(sentences, truncation=True, return_tensors='pt').to(self.device)
        
        # Check if input length exceeds model's maximum context length
        max_context_length = len(encoded_input['input_ids'][0])
        max_length_checks = [
            self.model.wpe.num_embeddings if hasattr(self.model, 'wpe') else None,
            self.model.embeddings.position_embeddings.num_embeddings if hasattr(self.model.embeddings, 'position_embeddings') else None
        ]
        for mi in max_length_checks:
            if mi is not None and max_context_length > mi:
                raise Exception(f"This model's maximum context length is {mi} tokens, however you requested {max_context_length} tokens")

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings[0], max_context_length

    def completion(self, text, do_sample=True, temperature=1.3, max_length=2048, **kwargs):
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        out = self.model.generate(input_ids, do_sample=do_sample, temperature=temperature, max_length=max_length, **kwargs) 
        return self.tokenizer.decode(out[0])

    def compare(self, e1, e2):
        return self.cos(e1, e2)
