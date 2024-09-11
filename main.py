from training import Training
import torch
import pickle

if __name__ == '__main__':
    torch.cuda.set_device(0)
    device = torch.device('cuda')
    model = 'mlp_fix'
    Dtype = 'noise0.6'
    add_annot = '_lr0.001'
    # add_annot = 'redo'
    # with open('datasets/sample_data0.6.pickle', 'rb') as handle:
    #     data = pickle.load(handle)
    with open('datasets/'+Dtype+'.pickle', 'rb') as handle:
        data = pickle.load(handle)
    # with open('datasets/gpt_emb_mnar_imb_no_nan.pickle', 'rb') as handle:
    #     data = pickle.load(handle)
    # training = Training(data, model = model, hidden_dim=1536, device = device)
    training = Training(data, model = model, input_dim = 49, hidden_dim=1536, device = device)
    # training = Training(data, model = model, input_dim = 1536, hidden_dim=128, device = device)
    best_model, train_loss, val_loss = training.training()
    
    torch.save(best_model, 'outputs/baseline_'+model+'_'+Dtype+add_annot+'.pt')
    with open('outputs/train_losses_'+model+'_'+Dtype+add_annot+'.pickle', 'wb') as handle:
        pickle.dump(train_loss, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('outputs/val_losses_'+model+'_'+Dtype+add_annot+'.pickle', 'wb') as handle:
        pickle.dump(val_loss, handle, protocol=pickle.HIGHEST_PROTOCOL)