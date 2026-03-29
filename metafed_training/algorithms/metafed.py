import torch
import torch.nn.functional as F
import copy

def metafed_distill(models, data_loaders, device, temperature=3.0, alpha=0.7, epochs=3):
    """Cyclic knowledge distillation between federation models"""
    num_models = len(models)
    
    for cycle in range(epochs):
        for i in range(num_models):
            student = models[i]
            teacher = models[(i - 1) % num_models]
            
            student.train()
            teacher.eval()
            
            optimizer = torch.optim.Adam(student.parameters(), lr=0.0005)
            
            for data, target in data_loaders[i]:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                
                student_out = student(data)
                with torch.no_grad():
                    teacher_out = teacher(data)
                
                # Hard label loss
                hard_loss = F.cross_entropy(student_out, target)
                
                # Soft label loss (knowledge distillation)
                soft_loss = F.kl_div(
                    F.log_softmax(student_out / temperature, dim=1),
                    F.softmax(teacher_out / temperature, dim=1),
                    reduction='batchmean'
                ) * (temperature ** 2)
                
                loss = alpha * hard_loss + (1 - alpha) * soft_loss
                loss.backward()
                optimizer.step()
    
    return models
