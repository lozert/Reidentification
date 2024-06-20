
def count_param(model, show_layers=False):
    if show_layers:
        i = 0
        for name, param in model.named_parameters():
            print(f"{name} " +  str(param.size()))
            i+=1
        
        # train_layer = ['last', 'block8', 'logits']
        for name, param in model.named_parameters():        
            if name.startswith('last') or name.startswith('logits') : 
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Проверка, какие параметры будут обновляться
        for name, param in model.named_parameters():
            print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")

    print(sum([p.numel() for p in model.parameters() if p.requires_grad]))


if __name__ == "__main__":
    from models.inception_resnet_v1 import InceptionResnetV1
    from models.AppendLinear import AppendLinear
    from models.ModifiedInceptionResnetV1 import ModifiedInceptionResnetV1
    
    model_path = "model_weights3.pt"
    # original_model = InceptionResnetV1(pretrained='casia-webface')
    
    # Здесь должен быть ваш оригинальный модельный класс
    model = ModifiedInceptionResnetV1()
    
    
    count_param(model, show_layers=True)