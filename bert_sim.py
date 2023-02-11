from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import torch.nn as nn

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModelForMaskedLM.from_pretrained("bert-base-chinese")

input1 = "我爱广州"
input2 = "我喜欢广州"


def calcuate_sim(a,b):
    return torch.cosine_similarity(a,b,dim = 0)


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


inputs_a = tokenizer(input1, return_tensors="pt",)
outputs_a = model(**inputs_a, labels=inputs_a["input_ids"],output_hidden_states = True)


inputs_b = tokenizer(input2, return_tensors="pt")
outputs_b = model(**inputs_b, labels=inputs_b["input_ids"],output_hidden_states = True)


Pool = MeanPooling()

for i in range(0,13):
    a_h = Pool(outputs_a["hidden_states"][i],inputs_a["attention_mask"])
    b_h = Pool(outputs_b["hidden_states"][i],inputs_b["attention_mask"])
    sim = calcuate_sim(a_h.flatten(),b_h.flatten())
    print("lay:",i,"cos_sim:",sim.detach().numpy())


