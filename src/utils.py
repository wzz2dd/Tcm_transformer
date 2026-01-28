import torch
import torch.nn.functional as F

def generate_formula_sequence(model, disease_vec, herb_nes_matrix, config, hard=False):
    batch_size = disease_vec.size(0)
    device = disease_vec.device
    
    total_herb_effect = torch.zeros_like(disease_vec) 
    current_input_embed = torch.zeros(batch_size, 1, config['d_model']).to(device)
    
    generated_herbs_indices = []
    generated_dosages = []
    logits_mask = torch.zeros(batch_size, model.num_herbs).to(device)
    
    for t in range(config['max_seq_len']):
        herb_logits, dosage_pred, full_output = model(disease_vec, tgt_seq=current_input_embed)
        
        herb_logits = herb_logits - (logits_mask * 1e9)
        
        if hard:
            chosen_indices = torch.argmax(herb_logits, dim=-1)
            herb_one_hot = F.one_hot(chosen_indices, num_classes=model.num_herbs).float()
        else:
            herb_one_hot = F.gumbel_softmax(herb_logits, tau=1.5, hard=False)
        
        with torch.no_grad():
             current_indices = torch.argmax(herb_one_hot, dim=-1)
             logits_mask.scatter_(1, current_indices.unsqueeze(1), 1.0)
        
        current_herb_nes = torch.matmul(herb_one_hot, herb_nes_matrix)
        weighted_effect = current_herb_nes * dosage_pred
        total_herb_effect = total_herb_effect + weighted_effect
        
        next_herb_embed = torch.matmul(herb_one_hot, model.herb_embedding.weight[:-1]) 
        next_dosage_embed = model.dosage_projector(dosage_pred)
        next_input = next_herb_embed + next_dosage_embed
        
        current_input_embed = torch.cat([current_input_embed, next_input.unsqueeze(1)], dim=1)
        
        if hard:
            generated_herbs_indices.append(chosen_indices)
            generated_dosages.append(dosage_pred)
            
    return total_herb_effect, generated_herbs_indices, generated_dosages