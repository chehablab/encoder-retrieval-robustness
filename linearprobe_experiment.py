import os, json
import torch

import linearprobe_pipeline

def experiment_done(encoder_name, dataset_name, checkpoint_path, max_epochs=20):
  encoder_name = encoder_name.replace("/", "_")
  dataset_name = dataset_name.replace("/", "_")
  checkpoint_name = f"{encoder_name}_{dataset_name}.pt"
  checkpoint_filepath = os.path.join(checkpoint_path, checkpoint_name)
  if os.path.exists(checkpoint_filepath):
    checkpoint = torch.load(checkpoint_filepath, weights_only=False)
    epochs = checkpoint['epoch']
    early_stopped = checkpoint['early_stopped'] if 'early_stopped' in checkpoint else False
    print(f"Experiment ran for {epochs} epochs.")
    if epochs == max_epochs:
      return True
    else:
      if early_stopped:
        print("Early stopped.")
        return True
      else:
        return False
      
if __name__ == "__main__":
    config = json.load(open("config.json", "r"))
    chkpt_path = "./chkpt_linearprobe"
    
    for encoder_obj in config['encoders']:
        if encoder_obj['active']:
            for dataset_obj in config['datasets']:
                if dataset_obj['active']:
                    if not experiment_done(encoder_obj['id'], dataset_obj['id'], chkpt_path):
                        linearprobe_pipeline.probe(encoder_obj["id"], dataset_obj['id'],
                                                    batch_size= 64, n_epochs=20, encoder_target_dim=768,
                                                    num_workers= 4, learning_rate= 1e-3, random_state=42,
                                                    chkpt_path= chkpt_path, verbose= True)